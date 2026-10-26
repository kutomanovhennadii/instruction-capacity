"""A deterministic provider without network and external libraries."""

from __future__ import annotations

from dataclasses import dataclass

from iec.contracts import GenParams, ModelConfig, Prompt, RawOutput, TokenUsage
from iec.infrastructure.telemetry import TelemetryContext, telemetry_step

DEFAULT_MOCK_TOKENIZER_NAME = "mock"
MOCK_SINGLE_TOKEN_MAX_CHARS = 16


def _mock_telemetry_context(
    client: MockModelClient,
    prompt: Prompt,
    params: GenParams,
) -> TelemetryContext:
    """Extract the call context of the mock model."""
    del prompt, params
    return TelemetryContext(model_id=client.model_id)


@dataclass(frozen=True, slots=True)
class MockModelClient:
    """A mock implementation of ``ModelClient`` that returns an exact echo of the prompt.

    The call does not touch the network and keeps no state of previous requests. The
    result is fully determined by the input ``Prompt``; the parameters are accepted to
    follow the common contract and are reflected only in the technical snapshot ``raw``.
    """

    model_id: str

    @telemetry_step(
        "model_generate",
        module_name="mock_provider",
        context_extractor=_mock_telemetry_context,
    )
    def generate(self, prompt: Prompt, params: GenParams) -> RawOutput:
        """Return a deterministic raw answer without network side effects."""
        # The answer is an exact echo of the prompt, so both counters equal its length;
        # translation into contract fields is the duty of the adapter, here it degenerates
        # to an assignment.
        token_count = len(prompt.text.split())
        return RawOutput(
            text=prompt.text,
            usage=TokenUsage(prompt_tokens=token_count, response_tokens=token_count),
            finish_reason="stop",
            latency_s=0.0,
            raw={"provider": "mock", "seed": params.seed},
        )


@dataclass(frozen=True, slots=True)
class MockTokenizer:
    """A simple tokenizer for checking the scaffold; it does not imitate BPE."""

    name: str = DEFAULT_MOCK_TOKENIZER_NAME
    single_token_max_chars: int = MOCK_SINGLE_TOKEN_MAX_CHARS

    def is_single_token(self, word: str) -> bool:
        """Apply a deterministic rule to the given string (as is).

        A leading separator space is allowed and does not count toward the length - real
        BPE vocabularies store a unit like ``"▁lake"`` this way; inner spaces are not
        allowed.
        """
        probe = word[1:] if word.startswith(" ") else word
        return (
            bool(probe)
            and not any(character.isspace() for character in probe)
            and len(probe) <= self.single_token_max_chars
        )

    def count_tokens(self, text: str) -> int:
        """Count whitespace-separated items as mock tokens."""
        return len(text.split())


def build_mock_provider(config: ModelConfig) -> tuple[MockModelClient, MockTokenizer]:
    """Assemble the mock model and tokenizer from the configuration.

    Args:
        config: Model configuration; the mock reads ``model_id`` and ``tokenizer_name``.

    Returns:
        A deterministic pair without heavy resources and network side effects.
    """
    tokenizer_name = config.tokenizer_name or DEFAULT_MOCK_TOKENIZER_NAME
    return MockModelClient(model_id=config.model_id), MockTokenizer(name=tokenizer_name)
