"""Blocks of observation and their levels: how much of every block reaches the log and the sinks.

A block of observation is a named group of working operations: the call of a model, the
generation of a task, a fit, the build of the passport. The configuration gives every block a
level, as a logger has one: an event reaches the log and the sinks when its severity is not
below the level of its block. A successful call has the severity ``info`` and a failed one
``error``; the level ``off`` lets nothing through, and the operation then runs without timing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from iec.infrastructure.telemetry.telemetry_events import TelemetryStatus

#: The blocks of observation of the open part; the closed part passes its own to its builder.
MODEL_CALL = "model_call"
TASK_GENERATION = "task_generation"
VOCAB_BUILD = "vocab_build"
POINT_EXECUTION = "point_execution"
FIT = "fit"
PASSPORT_BUILD = "passport_build"
OPEN_PART_BLOCKS: frozenset[str] = frozenset(
    {MODEL_CALL, TASK_GENERATION, VOCAB_BUILD, POINT_EXECUTION, FIT, PASSPORT_BUILD}
)

#: Severity of an event by its outcome, on the scale of the standard ``logging`` levels.
EVENT_SEVERITY: Mapping[TelemetryStatus, int] = {
    TelemetryStatus.SUCCESS: logging.INFO,
    TelemetryStatus.FAILURE: logging.ERROR,
}


class ObservationLevel(StrEnum):
    """Level of a block of observation, named like the levels of a logger."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    OFF = "off"

    def admits(self, status: TelemetryStatus) -> bool:
        """Tell whether an event with this outcome passes the level."""
        return EVENT_SEVERITY[status] >= _THRESHOLD[self]


#: Lowest severity that passes each level; ``off`` is above every severity.
_THRESHOLD: Mapping[ObservationLevel, int] = {
    ObservationLevel.DEBUG: logging.DEBUG,
    ObservationLevel.INFO: logging.INFO,
    ObservationLevel.WARNING: logging.WARNING,
    ObservationLevel.ERROR: logging.ERROR,
    ObservationLevel.OFF: logging.CRITICAL + 1,
}


@dataclass(frozen=True, slots=True)
class ObservationLevels:
    """Levels of the blocks of observation: one per listed block and a default for the rest."""

    default: ObservationLevel = ObservationLevel.INFO
    blocks: Mapping[str, ObservationLevel] = field(default_factory=dict)

    def level_of(self, block: str) -> ObservationLevel:
        """Return the level of the block, or the default level when the block is not listed."""
        return self.blocks.get(block, self.default)
