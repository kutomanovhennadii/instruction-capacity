"""The assembled prompt, ready to be passed to the model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    """The exact request text and the optional position of a payload task.

    Assembling the prompt is not part of the contract; it lives in the prompt generator.
    ``payload_position`` takes the protocol values ``BEFORE`` and ``AFTER`` or ``None``
    when there is no additional task.
    """

    text: str
    payload_position: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return the exact text and the payload position unchanged."""
        return {
            "text": self.text,
            "payload_position": self.payload_position,
        }
