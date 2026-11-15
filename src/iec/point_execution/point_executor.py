"""Execution of one measurement point: task, ceiling check, model call, record."""

from __future__ import annotations

import time

from iec.answer_validator import build_reference
from iec.contracts import (
    ModelConfig,
    PointConditions,
    PointOutcome,
    PointRecord,
    PointStatus,
    Prompt,
    RawOutput,
    RuleTable,
    SourceText,
    Vocab,
    has_no_words,
)
from iec.infrastructure.exceptions import ProviderError
from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry import (
    POINT_EXECUTION,
    TelemetryHub,
    point_context,
    telemetry_step,
)
from iec.model_client import ModelClient, Tokenizer
from iec.prompt_generator import PromptBuilder, RuleGenerator, TextGenerator

_MODULE_NAME = "point_executor"

#: Error code of a model that stayed silent until the time limit twice: the reliable layer retries
#: a timeout once with the limit doubled, and the point gets no answer, ``api_error``.
PROVIDER_TIMEOUT_CODE = "PROVIDER_TIMEOUT"
#: Error code of a model that answered twice without a word: the point gets ``api_error``.
EMPTY_ANSWER_CODE = "EMPTY_ANSWER"
#: Calls of the model when it answers without a word: the first and one retry.
EMPTY_ANSWER_CALLS = 2

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


class PointExecutor:
    """Executor of one point - a thin chain of ready operations.

    The task is built by the generators, the reference by the reference builder; the executor
    passes contracts from step to step and branches on the status. A bad answer of the model is
    data: an empty or garbage answer gets status ``ok``, and the alignment reads it from the
    record. A provider failure is not data: the point gets ``api_error`` and no answer. An
    exception of the bench itself is not caught here - the caller decides what to do with it.

    ``telemetry`` is the hub that records every execution in the block ``point_execution`` with
    the point key - the model and the conditions; the generators of the task share it.
    """

    telemetry: TelemetryHub

    def __init__(
        self,
        client: ModelClient,
        tokenizer: Tokenizer,
        vocab: Vocab,
        config: ModelConfig,
        *,
        telemetry: TelemetryHub | None = None,
    ) -> None:
        """Assemble the executor once per model.

        Args:
            client: Model client built by the builder registry.
            tokenizer: Tokenizer of the same model; it measures both limits.
            vocab: Vocabulary of the measurement.
            config: Model configuration: generation parameters and declared limits.
            telemetry: Hub that records every execution in the block ``point_execution`` and
                every generation of the task in the block ``task_generation``; without it the
                executor gets a hub of its own that writes the log lines only.

        Raises:
            VocabError: ``RULES_EMPTY_VOCAB`` - the vocabulary is empty (from the rule
                generator).
        """
        self._client = client
        self._tokenizer = tokenizer
        self._config = config
        self.telemetry = telemetry if telemetry is not None else TelemetryHub.log_only()
        self._rule_generator = RuleGenerator(vocab, telemetry=self.telemetry)
        self._text_generator = TextGenerator(vocab, tokenizer, telemetry=self.telemetry)
        self._prompt_builder = PromptBuilder()
        self._logger = get_logger(LogContext(module_name=_MODULE_NAME))

    @property
    def model_id(self) -> str:
        """Return the opaque model identifier of the configuration."""
        return self._config.model_id

    @telemetry_step(
        "point_execute",
        block=POINT_EXECUTION,
        module_name=_MODULE_NAME,
        context_extractor=point_context,
    )
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
            raw_output, status, error_code = self._call_model(prompt, conditions)

        return PointOutcome(
            record=_point_record(conditions, rule_table, source_text, raw_output, status),
            rule_table=rule_table,
            source_text=source_text,
            prompt=prompt,
            reference=reference,
            prompt_len_tokens=prompt_len_tokens,
            raw_output=raw_output,
            error_code=error_code,
            duration_s=time.perf_counter() - started,
        )

    def _call_model(
        self, prompt: Prompt, conditions: PointConditions
    ) -> tuple[RawOutput | None, PointStatus, str | None]:
        """Call the model and turn a failure or an empty answer into the outcome of the point.

        A provider failure, the silence until the doubled time limit (``PROVIDER_TIMEOUT``)
        included, leaves the point without an answer, ``api_error``. An answer without a word
        is asked once more; a second one gives ``api_error`` with ``EMPTY_ANSWER``.

        Args:
            prompt: The assembled prompt of the point.
            conditions: The conditions of the point, for the log lines.

        Returns:
            The raw output (``None`` without an answer), the status of the point and the error
            code of the failure (``None`` when the model answered).

        Side effects:
            One or two calls of the model; a failure and an empty answer are written to the log.
        """
        for call in range(1, EMPTY_ANSWER_CALLS + 1):
            try:
                output = self._client.generate(prompt, self._config.gen_params)
            except ProviderError as error:
                self._logger.error(
                    f"point {conditions.identity} provider failure: {error.error_code}"
                )
                return None, PointStatus.API_ERROR, error.error_code
            if not has_no_words(output.text):
                return output, PointStatus.OK, None
            self._logger.warning(
                f"point {conditions.identity} answered without a word "
                f"(call {call} of {EMPTY_ANSWER_CALLS})"
            )
        return None, PointStatus.API_ERROR, EMPTY_ANSWER_CODE


def _point_record(
    conditions: PointConditions,
    rule_table: RuleTable,
    source_text: SourceText,
    raw_output: RawOutput | None,
    status: PointStatus,
) -> PointRecord:
    """Assemble the published record of a point from its task and the answer of the model.

    Args:
        conditions: The conditions of the point.
        rule_table: The rule list in the order shown to the model.
        source_text: The source text of the task.
        raw_output: The raw output of the model; ``None`` when the model was not called or did
            not answer.
        status: The status of the point.

    Returns:
        The record; without an answer the answer and the provider counters are ``None``.
    """
    usage = None if raw_output is None else raw_output.usage
    return PointRecord(
        conditions=conditions,
        rules=tuple((rule.source, rule.target) for rule in rule_table.rules),
        source_text=source_text.text,
        answer=None if raw_output is None else raw_output.text,
        status=status,
        prompt_tokens=None if usage is None else usage.prompt_tokens,
        response_tokens=None if usage is None else usage.response_tokens,
    )
