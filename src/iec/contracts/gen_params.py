"""Technical parameters of one model generation call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenParams:
    """Provider-independent generation parameters.

    The contract holds only technical settings of the call. The model adapter is
    responsible for translating them into the format of a particular SDK and for handling
    unsupported parameters.
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: int | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        """Return all parameters in serializable form."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
        }
