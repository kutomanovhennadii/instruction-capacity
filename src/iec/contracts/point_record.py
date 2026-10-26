"""Published record of one measurement point."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from iec.contracts.point_conditions import PointConditions
from iec.infrastructure.exceptions import ContractViolationError

#: Version of the measurement protocol: prompt template, text layout, validation rules.
PROTOCOL_VERSION = "0.1"


class PointStatus(StrEnum):
    """Outcome of one point.

    ``ok`` - the model answered; an empty answer and silence until the time limit are
    answers too, and the validator reads them as a collapse from the first word.
    ``na`` - the task does not fit the model limits; the model was not called.
    ``api_error`` - the provider did not answer; the point is not measured.
    """

    OK = "ok"
    NA = "na"
    API_ERROR = "api_error"


@dataclass(frozen=True, slots=True)
class PointRecord:
    """What a published measurement keeps about one point.

    The record holds what was sent to the model and what it answered, plus the conditions
    that regenerate the task. The prompt, the reference and the validation result are not
    stored: the public code rebuilds them from these fields. ``answer`` is ``None`` when
    the model was not called or did not answer; the token counters are ``None`` when the
    provider did not report them.
    """

    conditions: PointConditions
    rules: tuple[tuple[str, str], ...]
    source_text: str
    answer: str | None
    status: PointStatus
    prompt_tokens: int | None
    response_tokens: int | None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as one JSON object of the published format."""
        return {
            **self.conditions.as_dict(),
            "rules": [list(rule) for rule in self.rules],
            "source_text": self.source_text,
            "answer": self.answer,
            "status": self.status.value,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PointRecord:
        """Read a record of the published format; unknown keys are ignored.

        Raises:
            ContractViolationError: ``POINT_RECORD_INVALID`` - a required key is missing,
                a value has a wrong type or the status is unknown.
        """
        try:
            return cls(
                conditions=PointConditions(
                    D=int(raw["D"]),
                    n=int(raw["n"]),
                    m=int(raw["m"]),
                    seed_rules=int(raw["seed_rules"]),
                    seed_text=int(raw["seed_text"]),
                    seed_order=int(raw["seed_order"]),
                ),
                rules=tuple((str(source), str(target)) for source, target in raw["rules"]),
                source_text=str(raw["source_text"]),
                answer=None if raw.get("answer") is None else str(raw["answer"]),
                status=PointStatus(raw["status"]),
                prompt_tokens=_optional_int(raw.get("prompt_tokens")),
                response_tokens=_optional_int(raw.get("response_tokens")),
            )
        except (KeyError, ValueError, TypeError) as cause:
            raise ContractViolationError.from_cause(
                f"not a point record of the published format: {cause!r}",
                cause,
                error_code="POINT_RECORD_INVALID",
            ) from cause


def _optional_int(value: Any) -> int | None:
    """Return ``None`` as is and any other value as an integer."""
    return None if value is None else int(value)
