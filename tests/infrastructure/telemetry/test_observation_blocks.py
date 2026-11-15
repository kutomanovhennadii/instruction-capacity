"""Mirror tests of the blocks of observation and their levels."""

import pytest

from iec.infrastructure.telemetry.observation_blocks import (
    MODEL_CALL,
    OPEN_PART_BLOCKS,
    ObservationLevel,
    ObservationLevels,
)
from iec.infrastructure.telemetry.telemetry_events import TelemetryStatus


@pytest.mark.parametrize(
    ("level", "success", "failure"),
    [
        (ObservationLevel.DEBUG, True, True),
        (ObservationLevel.INFO, True, True),
        (ObservationLevel.WARNING, False, True),
        (ObservationLevel.ERROR, False, True),
        (ObservationLevel.OFF, False, False),
    ],
)
def test_level_admits_events_by_their_severity(
    level: ObservationLevel, success: bool, failure: bool
) -> None:
    assert level.admits(TelemetryStatus.SUCCESS) is success
    assert level.admits(TelemetryStatus.FAILURE) is failure


def test_unlisted_block_takes_the_default_level() -> None:
    levels = ObservationLevels(
        default=ObservationLevel.WARNING, blocks={MODEL_CALL: ObservationLevel.OFF}
    )

    assert levels.level_of(MODEL_CALL) is ObservationLevel.OFF
    assert levels.level_of("fit") is ObservationLevel.WARNING


def test_default_levels_observe_every_block_at_info() -> None:
    assert ObservationLevels().level_of(MODEL_CALL) is ObservationLevel.INFO


def test_open_part_blocks() -> None:
    assert OPEN_PART_BLOCKS == {
        "model_call",
        "task_generation",
        "vocab_build",
        "point_execution",
        "fit",
        "passport_build",
    }


def test_level_values_are_the_names_of_the_configuration() -> None:
    assert [level.value for level in ObservationLevel] == [
        "debug",
        "info",
        "warning",
        "error",
        "off",
    ]
