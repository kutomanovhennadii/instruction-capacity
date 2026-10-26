"""Area of analysis from the points: cube bounds, cell states, safe points, the printed cube."""

import pytest

from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.infrastructure.exceptions import PassportError
from iec.passport.analysis_area import Cell, CellState, build_area_cube, render_cube, safe_points
from iec.passport.measured_points import MeasuredPoint


def _point(
    D: int,  # noqa: N803 - protocol axis name
    n: int,
    m: int,
    status: PointStatus = PointStatus.OK,
    *,
    seed: int = 1,
    kind_3: int | None = None,
) -> MeasuredPoint:
    record = PointRecord(
        conditions=PointConditions(
            D=D, n=n, m=m, seed_rules=seed, seed_text=seed + 1, seed_order=0
        ),
        rules=(),
        source_text="",
        answer="" if status is PointStatus.OK else None,
        status=status,
        prompt_tokens=None,
        response_tokens=None,
    )
    return MeasuredPoint(record=record, trials=(), first_fail_kind_3_line=kind_3)


def _three(
    D: int,  # noqa: N803 - protocol axis name
    n: int,
    m: int,
    status: PointStatus = PointStatus.OK,
    *,
    kind_3: int | None = None,
) -> list[MeasuredPoint]:
    return [_point(D, n, m, status, seed=seed, kind_3=kind_3) for seed in (1, 3, 5)]


def test_the_axes_double_from_the_start_step_to_the_largest_executed_point() -> None:
    refused = _point(64, 40, 8, PointStatus.API_ERROR)
    cube = build_area_cube([*_three(4, 5, 1), *_three(16, 10, 4), refused])
    assert cube.bounds.div == ((1, 4), (5, 8), (9, 16))
    assert cube.bounds.row == ((1, 1), (2, 2), (3, 4))
    assert cube.bounds.rep == ((1, 5), (6, 10))


def test_three_open_points_keep_their_cell_ordinary() -> None:
    cube = build_area_cube(_three(4, 5, 1))
    assert cube.state_of(Cell(1, 1, 1)) is CellState.ORDINARY
    assert cube.count_of(Cell(1, 1, 1)) == 3


def test_a_collapse_in_every_answer_makes_a_boundary_cell() -> None:
    cube = build_area_cube([*_three(4, 5, 1), *_three(8, 5, 1, kind_3=1)])
    assert cube.state_of(Cell(2, 1, 1)) is CellState.BOUNDARY
    assert cube.state_of(Cell(1, 1, 1)) is CellState.ORDINARY


def test_a_partial_collapse_is_a_measurement_not_a_boundary() -> None:
    mixed = [
        _point(8, 5, 1, seed=1, kind_3=1),
        _point(8, 5, 1, seed=3),
        _point(8, 5, 1, seed=5, kind_3=1),
    ]
    cube = build_area_cube([*_three(4, 5, 1), *mixed])
    assert cube.state_of(Cell(2, 1, 1)) is CellState.ORDINARY


def test_a_task_beyond_the_model_limits_makes_a_boundary_cell() -> None:
    cube = build_area_cube([*_three(4, 5, 1), *_three(8, 5, 1, PointStatus.NA)])
    assert cube.state_of(Cell(2, 1, 1)) is CellState.BOUNDARY


def test_three_refusals_without_an_answer_make_a_boundary_cell() -> None:
    refused = _three(8, 5, 1, PointStatus.API_ERROR)
    cube = build_area_cube([*_three(4, 5, 1), *refused, *_three(16, 5, 1)])
    assert cube.state_of(Cell(2, 1, 1)) is CellState.BOUNDARY


def test_a_representative_measured_fewer_than_three_times_is_unreachable() -> None:
    cube = build_area_cube([*_three(4, 5, 1), _point(8, 5, 1), _point(8, 5, 1, seed=3)])
    assert cube.state_of(Cell(2, 1, 1)) is CellState.UNREACHABLE


def test_only_the_points_of_ordinary_cells_are_safe() -> None:
    open_points = _three(4, 5, 1)
    collapsed = _three(8, 5, 1, kind_3=1)
    cube = build_area_cube([*open_points, *collapsed])
    assert safe_points([*open_points, *collapsed], cube) == tuple(open_points)


def test_the_printed_cube_shows_counts_and_marks_unreachable_cells() -> None:
    cube = build_area_cube([*_three(4, 5, 1), *_three(8, 5, 2)])
    assert render_cube(cube) == "\n".join(
        [
            "bin cube: point counts by bins D × m × n",
            "",
            "m ∈ 1",
            "D\\n 1-5",
            "1-4   3",
            "5-8   x",
            "",
            "m ∈ 2",
            "D\\n 1-5",
            "1-4   x",
            "5-8   3",
        ]
    )


def test_no_executed_point_is_an_error() -> None:
    with pytest.raises(PassportError) as error:
        build_area_cube([_point(4, 5, 1, PointStatus.API_ERROR)])
    assert error.value.error_code == "PASSPORT_INPUT_MISSING"
