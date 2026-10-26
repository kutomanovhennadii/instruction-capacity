"""The raw model answer and the separated technical fields of the provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iec.contracts.token_usage import TokenUsage


@dataclass(frozen=True, slots=True)
class RawOutput:
    """Result of the model adapter at the boundary with the rest of the bench.

    ``text`` keeps the raw answer without post-processing. ``usage``, ``finish_reason`` and
    ``latency_s`` are technical fields and are not mixed with the subject evaluation.
    ``raw`` allows the type ``Any`` only as an audit snapshot of the external response.

    ``usage`` is the :class:`TokenUsage` contract, not a provider dictionary: the adapter
    translates the provider's counter names into the project fields at the boundary, so
    the cost of a point reads the same regardless of who measured it.
    """

    text: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    raw: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return all fields without changing the raw text and the external response."""
        return {
            "text": self.text,
            "usage": self.usage.as_dict() if self.usage is not None else None,
            "finish_reason": self.finish_reason,
            "latency_s": self.latency_s,
            "raw": self.raw,
        }
