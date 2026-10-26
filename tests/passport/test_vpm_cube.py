"""Run cube V × P × M: density level, the cell of a node, the passport bins, the weights."""

import numpy as np
import pytest

from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.passport.measured_points import MeasuredPoint
from iec.passport.vpm_cube import (
    VpmCube,
    build_layout,
    cell_balanced_weights,
    density_level,
    density_p,
)


def _point(
    D: int,  # noqa: N803 - protocol axis name
    n: int,
    m: int,
    status: PointStatus = PointStatus.OK,
    *,
    seed: int = 1,
) -> MeasuredPoint:
    record = PointRecord(
        conditions=PointConditions(
            D=D, n=n, m=m, seed_rules=seed, seed_text=seed + 1, seed_order=0
        ),
        rules=(),
        source_text="",
        answer="",
        status=status,
        prompt_tokens=None,
        response_tokens=None,
    )
    return MeasuredPoint(record=record, trials=(), first_fail_kind_3_line=None)


def test_the_density_level_grows_with_the_density_and_is_zero_at_one() -> None:
    """``⌊log₂(m/D)⌋``: ``p = 1`` - 0, ``[1/2, 1)`` - −1, ``[1/4, 1/2)`` - −2."""
    assert density_level(8, 8) == 0
    assert density_level(2, 1) == -1
    assert density_level(3, 2) == -1  # p = 2/3
    assert density_level(12, 5) == -2  # p ~ 0.42
    assert density_level(4, 1) == -2  # p = 1/4
    assert density_level(16, 1) == -4
    assert density_p(0) == (1.0, 1.0)
    assert density_p(-2) == (0.25, 0.5)


def test_cells_of_agrees_with_cell_of_row_by_row() -> None:
    cube = VpmCube(reference_total_words=20)
    nodes = np.array([[4, 5, 1], [8, 10, 2], [12, 5, 5], [20, 10, 3]], dtype=np.int64)
    rows = [tuple(row) for row in cube.cells_of(nodes).tolist()]
    assert rows == [cube.cell_of(*row) for row in nodes.tolist()]


def test_a_node_with_more_rules_than_words_is_refused() -> None:
    with pytest.raises(ContractViolationError) as error:
        VpmCube(reference_total_words=20).cell_of(4, 5, 8)
    assert error.value.error_code == "RUN_CUBE_INVALID"


def test_the_layout_starts_at_twenty_words_and_has_no_ceiling() -> None:
    layout = build_layout(
        (_point(16, 20, 1), _point(16, 5, 8), _point(64, 40, 1, PointStatus.NA))
    )
    assert layout.volume_bounds == ((1, 20), (21, 40), (41, 80), (81, 160), (161, 320))
    assert layout.density_bounds == ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16))
    assert layout.row_bounds == ((1, 1), (2, 2), (3, 4), (5, 8))
    # The reference density bin holds ⌈4/1⌉ = 4: bin 3 counted from p = 1, [1/4, 1/2).
    assert layout.density_reference == 3
    assert layout.volume_bin(320) == 5


def test_no_ok_point_gives_no_layout() -> None:
    with pytest.raises(PassportError) as error:
        build_layout((_point(4, 5, 1, PointStatus.NA),))
    assert error.value.error_code == "PASSPORT_INPUT_MISSING"


def test_weights_balance_the_cells_and_average_one() -> None:
    points = (_point(4, 5, 1, seed=1), _point(4, 5, 1, seed=3), _point(8, 10, 2))
    result = cell_balanced_weights(points, VpmCube(20))
    assert result is not None
    weights, summary = result
    # A cell of two points: 1/2 each, a cell of one: 1; the mean 2/3 makes 0.75, 0.75, 1.5.
    assert [weights[point.key] for point in points] == pytest.approx([0.75, 0.75, 1.5])
    assert (summary["cells"], summary["runs"], summary["max_occupancy"]) == (2, 3, 2)
    assert summary["n_eff"] == pytest.approx(9.0 / (2 * 0.75**2 + 1.5**2))


def test_no_points_give_no_weights() -> None:
    assert cell_balanced_weights((), VpmCube(20)) is None
