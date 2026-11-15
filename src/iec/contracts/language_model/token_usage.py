"""Token counter of a model call in the uniform fields of the project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens of one model call as counted by the provider.

    Providers name the same two quantities differently (``prompt_tokens`` /
    ``completion_tokens`` for some, ``input_tokens`` / ``output_tokens`` for others).
    Bringing them to one form is the duty of the adapter at the boundary of the model
    layer: foreign names do not leak into the bench, otherwise the cost of a point would be
    computed differently depending on the provider. The field names here are protocol
    roles: what went out in the prompt and what came back as the answer.

    Both fields are optional: a provider may not return the counter, and ``None`` means
    exactly "the provider did not report", not zero tokens. The raw provider counter stays
    available for audit in ``RawOutput.raw``.
    """

    prompt_tokens: int | None = None
    response_tokens: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        """Return a serializable snapshot of the counter."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
        }
