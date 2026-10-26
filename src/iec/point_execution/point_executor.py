"""Execution of one measurement point: task, ceiling check, model call, validation."""

from __future__ import annotations

import time

from iec.answer_validator import RunValidator, build_reference
from iec.contracts import (
    ModelConfig,
    PointConditions,
    PointOutcome,
    PointRecord,
    PointStatus,
    RawOutput,
    Vocab,
)
from iec.infrastructure.exceptions import ProviderError
from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry import TelemetryContext, telemetry_step
from iec.model_client import ModelClient, Tokenizer
from iec.prompt_generator import PromptBuilder, RuleGenerator, TextGenerator

_MODULE_NAME = "point_executor"

#: Error code of an adapter timeout. Silence of the model is data, not a failure: the point
#: is recorded as answered with an empty text, and the validator reads it as a collapse
#: from the first word.
PROVIDER_TIMEOUT_CODE = "PROVIDER_TIMEOUT"

#: Finish reason marking the empty answer of a silent point.
_SILENCE_FINISH_REASON = "timeout"

#: Answer reserve above the reference that the ceiling check requires before the call.
#: The reference is measured without the wrapper a model sometimes adds (a greeting, a code
#: fence); without this margin a limit could cut a legitimate answer. A protocol constant.
WRAPPER_RESERVE_TOKENS = 64


def ceiling_verdict(
    config: ModelConfig, prompt_len_tokens: int, response_tokens_needed: int
) -> str | None:
    """Return why the point does not fit the model limits, or ``None`` if it fits.

    Args:
        config: Model configuration with the answer reserve and the declared limits.
        prompt_len_tokens: Tokens of the assembled prompt, measured by the model tokenizer.
        response_tokens_needed: Tokens of the reference, measured by the same tokenizer.

    Returns:
        A human-readable reason, or ``None`` when the point can be executed.
    """
    max_tokens = config.gen_params.max_tokens
    window = config.context_window_tokens
    needed_with_wrapper = response_tokens_needed + WRAPPER_RESERVE_TOKENS
    if window is not None and prompt_len_tokens + max_tokens > window:
        return (
            f"prompt {prompt_len_tokens} + reserve {max_tokens} tokens "
            f"exceeds the context window {window}"
        )
    if needed_with_wrapper > max_tokens:
        return (
            f"the answer needs {response_tokens_needed} tokens plus a wrapper reserve "
            f"of {WRAPPER_RESERVE_TOKENS}, reserve max_tokens={max_tokens} is smaller"
        )
    ceiling = config.response_ceiling_tokens
    if ceiling is not None and needed_with_wrapper > ceiling:
        return (
            f"the answer needs {response_tokens_needed} tokens plus a wrapper reserve "
            f"of {WRAPPER_RESERVE_TOKENS}, model response ceiling {ceiling} is smaller"
        )
    return None


def _telemetry_context(self: PointExecutor, conditions: PointConditions) -> TelemetryContext:
    """Extract the point identity for the telemetry event of point execution."""
    return TelemetryContext(
        model_id=self.model_id,
        m=conditions.m,
        seed_rules=conditions.seed_rules,
        seed_text=conditions.seed_text,
        seed_order=conditions.seed_order,
    )


class PointExecutor:
    """Executor of one point - a thin chain of ready operations.

    The task is built by the generators, the reference by the reference builder, the trial
    outcomes by the validator; the executor passes contracts from step to step and branches
    on the status. A bad answer of the model is data: an empty or garbage answer gets status
    ``ok`` and honest zeros in the trials. A provider failure is not data: the point gets
    ``api_error`` and no trials. An exception of the bench itself is not caught here - the
    caller decides what to do with it.
    """

    def __init__(
        self,
        client: ModelClient,
        tokenizer: Tokenizer,
        vocab: Vocab,
        config: ModelConfig,
    ) -> None:
        """Assemble the executor once per model.

        Args:
            client: Model client built by the builder registry.
            tokenizer: Tokenizer of the same model; it measures both limits.
            vocab: Vocabulary of the measurement.
            config: Model configuration: generation parameters and declared limits.
        """
        self._client = client
        self._tokenizer = tokenizer
        self._config = config
        self._rule_generator = RuleGenerator(vocab)
        self._text_generator = TextGenerator(vocab, tokenizer)
        self._prompt_builder = PromptBuilder()
        self._validator = RunValidator()
        self._logger = get_logger(LogContext(module_name=_MODULE_NAME))

    @property
    def model_id(self) -> str:
        """Return the opaque model identifier of the configuration."""
        return self._config.model_id

    @telemetry_step("point_execute", module_name=_MODULE_NAME, context_extractor=_telemetry_context)
    def execute(self, conditions: PointConditions) -> PointOutcome:
        """Execute one point and return its full outcome.

        Args:
            conditions: The six numbers of the point.

        Returns:
            ``PointOutcome``: the published record and everything built on the way.

        Raises:
            IECError: an exception of the bench itself (inconsistent conditions, a broken
                vocabulary); provider failures are not raised but become ``api_error``.

        Side effects:
            One call of the model unless the point does not fit the model limits.
        """
        started = time.perf_counter()
        rule_table = self._rule_generator.generate(conditions)
        source_text = self._text_generator.generate(conditions, rule_table)
        prompt = self._prompt_builder.build(rule_table, source_text)
        reference = build_reference(rule_table, source_text)
        prompt_len_tokens = self._tokenizer.count_tokens(prompt.text)
        response_tokens_needed = self._tokenizer.count_tokens(reference.text)

        raw_output: RawOutput | None = None
        error_code: str | None = None
        unreachable = ceiling_verdict(self._config, prompt_len_tokens, response_tokens_needed)
        if unreachable is not None:
            self._logger.info(f"point {conditions.identity} is unreachable: {unreachable}")
            status = PointStatus.NA
        else:
            try:
                raw_output = self._client.generate(prompt, self._config.gen_params)
                status = PointStatus.OK
            except ProviderError as error:
                error_code = error.error_code
                if error.error_code == PROVIDER_TIMEOUT_CODE:
                    self._logger.warning(
                        f"point {conditions.identity} gave no answer in time: recorded as "
                        f"a collapse from the first word ({error.error_code})"
                    )
                    raw_output = RawOutput(text="", finish_reason=_SILENCE_FINISH_REASON)
                    status = PointStatus.OK
                else:
                    self._logger.error(
                        f"point {conditions.identity} provider failure: {error.error_code}"
                    )
                    status = PointStatus.API_ERROR

        trial_result = (
            self._validator.validate(raw_output, reference, source_text, rule_table)
            if raw_output is not None
            else None
        )
        usage = None if raw_output is None else raw_output.usage
        record = PointRecord(
            conditions=conditions,
            rules=tuple((rule.source, rule.target) for rule in rule_table.rules),
            source_text=source_text.text,
            answer=None if raw_output is None else raw_output.text,
            status=status,
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            response_tokens=None if usage is None else usage.response_tokens,
        )
        return PointOutcome(
            record=record,
            rule_table=rule_table,
            source_text=source_text,
            prompt=prompt,
            reference=reference,
            prompt_len_tokens=prompt_len_tokens,
            raw_output=raw_output,
            trial_result=trial_result,
            error_code=error_code,
            duration_s=time.perf_counter() - started,
        )
