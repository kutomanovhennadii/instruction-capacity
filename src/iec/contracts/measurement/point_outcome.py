"""Full result of executing one point: the record plus everything built on the way."""

from __future__ import annotations

from dataclasses import dataclass

from iec.contracts.measurement.point_record import PointRecord
from iec.contracts.task.prompt import Prompt
from iec.contracts.language_model.raw_output import RawOutput
from iec.contracts.task.reference import Reference
from iec.contracts.task.rule_table import RuleTable
from iec.contracts.task.source_text import SourceText


@dataclass(frozen=True, slots=True)
class PointOutcome:
    """What the point executor returns to its caller.

    ``record`` is the published part. The rest serves the caller's own bookkeeping and is
    never published: the rule table and the source text with the replacement markup, the
    assembled prompt, the reference, the measured prompt length, the raw provider output,
    the provider error code and the duration. ``raw_output`` is ``None`` when the model was not
    called or did not answer.
    """

    record: PointRecord
    rule_table: RuleTable
    source_text: SourceText
    prompt: Prompt
    reference: Reference
    prompt_len_tokens: int
    raw_output: RawOutput | None
    error_code: str | None
    duration_s: float
