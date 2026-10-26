"""Full result of executing one point: the record plus everything built on the way."""

from __future__ import annotations

from dataclasses import dataclass

from iec.contracts.point_record import PointRecord
from iec.contracts.prompt import Prompt
from iec.contracts.raw_output import RawOutput
from iec.contracts.reference import Reference
from iec.contracts.rule_table import RuleTable
from iec.contracts.source_text import SourceText
from iec.contracts.trial_result import TrialResult


@dataclass(frozen=True, slots=True)
class PointOutcome:
    """What the point executor returns to its caller.

    ``record`` is the published part. The rest serves the caller's own bookkeeping and is
    never published: the rule table and the source text with the replacement markup, the
    assembled prompt, the reference, the measured prompt length, the raw provider output,
    the validation result, the provider error code and the duration. ``raw_output`` and
    ``trial_result`` are ``None`` when the model was not called or did not answer.
    """

    record: PointRecord
    rule_table: RuleTable
    source_text: SourceText
    prompt: Prompt
    reference: Reference
    prompt_len_tokens: int
    raw_output: RawOutput | None
    trial_result: TrialResult | None
    error_code: str | None
    duration_s: float
