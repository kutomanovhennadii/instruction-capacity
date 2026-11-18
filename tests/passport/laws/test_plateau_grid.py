"""Tests of the law of the length × word-number grid."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iec.passport.laws.forms.hyperbolic_forms import AsymptoticHyperbolicForm, SlantedHyperbolicForm
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.laws.plateau_grid import fit_plateau_law, fit_vs_law, vs_curve
from iec.passport.trials.layout.axis_bins import (
    vs_grid_cells,
)
from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds
from iec.passport.laws.gls_tools import (
    STATUS_INSUFFICIENT,
    STATUS_MEASURED,
    delta_se,
)

#: The surface of the synthetic data: ``κ_S, λ_S, μ₀, κ_V, ω_S, u_*, ω_V``.
_SURFACE = np.array([-2.0, 6.0, 3.0, -1.5, 1.0, 1.5, 0.5])
#: The form of a hyperbolic plateau and the candidate forms of the plateau of P_sync.
_HYPERBOLA = SlantedHyperbolicForm()
_FORMS = build_law_assignment(
    load_law_assignment_config(Path(__file__).resolve().parents[3] / "config" / "passport_laws.yaml")
).forms_of("P_sync", "grid_plateau")
#: The slices of the synthetic surfaces are two-asymptote hyperbolas; the tests of the grid take this
#: form alone - the choice of the form of the slices is tested in ``test_grid_slices``.
_SLICES = (AsymptoticHyperbolicForm(),)


def _vs_curve(params: np.ndarray, xu: np.ndarray) -> np.ndarray:
    """The surface by the seven numbers of the old coding: the slices, then the plateau hyperbola."""
    return vs_curve(AsymptoticHyperbolicForm(), params[[0, 1, 4]], _HYPERBOLA, params[[2, 3, 5, 6]], xu)


def plateau_level(mu_0: float, kappa_v: float, u_star: float, omega_v: float, u: float) -> float:
    """The plateau hyperbola at ``u``."""
    return float(_HYPERBOLA.value(np.array([mu_0, kappa_v, u_star, omega_v]), np.array([u]))[0])


def _layout_of(volume_top: int, density_top: int, rows_top: int) -> BinLayout:
    """Length from 20 words, density and rules from one; the reference density bin holds 4."""
    return BinLayout(
        volume_bounds=doubling_bounds(20, volume_top),
        density_bounds=doubling_bounds(1, density_top),
        row_bounds=doubling_bounds(1, rows_top),
        density_reference=3,
    )


def _layout() -> BinLayout:
    """Length 1-20, 21-40, 41-80; density up to 16 (reference bin 3); rules up to 8."""
    return _layout_of(80, 16, 8)


# --- The approximation block -------------------------------------------------------------


def _synthetic(layout, drop: tuple[str, ...] = ()) -> tuple[dict[str, float], dict]:
    """Build the cells ``θ`` exactly on the surface at the centres of mass, and the centres."""
    abscissae: dict[str, dict[str, float]] = {}
    theta: dict[str, float] = {}
    for volume, position in vs_grid_cells(layout):
        name = f"vs_{volume}_{position}"
        coords = {"x": 2.0 ** (position - 0.25), "u": (volume - 1) - 0.1}
        abscissae[name] = coords
        if name not in drop:
            point = np.array([[math.log2(coords["x"]), coords["u"]]])
            theta[name] = float(_vs_curve(_SURFACE, point)[0])
    return theta, abscissae


# --- The grid law by length slices -------------------------------------------------------


def _grid_only(layout, values):
    """``θ`` of the grid cells by a function ``(x, u, slice number) → value``, synthetic centres."""
    _, abscissae = _synthetic(layout)
    theta = {}
    for volume, position in vs_grid_cells(layout):
        coords = abscissae[f"vs_{volume}_{position}"]
        theta[f"vs_{volume}_{position}"] = values(coords["x"], coords["u"], volume)
    return theta, abscissae


def _free_plateaus(plateaus: Mapping[int, float]) -> Any:
    """A surface whose plateau is its own in every length slice: ``a = −2``, ``b = 6``, ``W_S = 1``."""

    def surface(x: float, u: float, volume: int) -> float:
        del u
        params = np.array([-2.0, 6.0, plateaus[volume], 0.0, 1.0, 1.0, 0.5])
        return float(_vs_curve(params, np.array([[math.log2(x), 0.0]]))[0])

    return surface


#: A surface whose plateau is a hyperbola in the length: ``c₀ = 4``, ``c₁ = −1.5``,
#: ``u_V = 1.5``, ``W_V = 0.5``.
_HYPERBOLIC = np.array([-2.0, 6.0, 4.0, -1.5, 1.0, 1.5, 0.5])


def _hyperbolic_surface(x: float, u: float, volume: int) -> float:
    """The surface ``_HYPERBOLIC`` at a cell: the plateau is the hyperbola itself."""
    del volume
    return float(_vs_curve(_HYPERBOLIC, np.array([[math.log2(x), u]]))[0])


def test_the_plateau_by_length_recovers_free_levels() -> None:
    """Free plateaus of six slices: the fit returns each at shared κ_S, λ_S, ω_S; the plateaus are
    published minus the shift δ_VS - the horizontal asymptote of the plateau hyperbola through
    them."""
    layout = _layout_of(640, 64, 32)
    plateaus = {1: 4.0, 2: 3.8, 3: 3.0, 4: 1.0, 5: -0.5, 6: -2.0}
    theta, abscissae = _grid_only(layout, _free_plateaus(plateaus))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, _SLICES, _FORMS)
    assert law["status"] == STATUS_MEASURED
    bias = law["plateau"].shift()
    assert law["slice_law"].shared.tolist() == pytest.approx(
        [-2.0, 6.0, 1.0], abs=1e-5
    )
    assert all(row["determined"] for row in law["slices"])
    assert [row["mu_v"] + bias for row in law["slices"]] == pytest.approx(
        list(plateaus.values()), abs=1e-5
    )
    assert law["chi2"] < 1e-8
    assert [row["cells"] for row in law["slices"]] == [2, 3, 4, 5, 6, 7]
    assert len(law["plateau"].bins) == 6


def test_a_slice_whose_cells_all_lie_on_the_slope_is_undetermined() -> None:
    """A slice with its plateau far above its cells: the plateau is not seen, no number."""
    layout = _layout_of(640, 64, 32)
    # The corner of slice 1 is left of its cells.
    plateaus = {1: 40.0, 2: 3.8, 3: 3.0, 4: 1.0, 5: -0.5, 6: -2.0}
    theta, abscissae = _grid_only(layout, _free_plateaus(plateaus))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, _SLICES, _FORMS)
    assert law["status"] == STATUS_MEASURED
    first, *rest = law["slices"]
    assert first["determined"] is False
    assert "the level of the slice is not seen" in first["reason"]
    assert all(row["determined"] for row in rest)
    bias = law["plateau"].shift()
    assert [row["mu_v"] + bias for row in rest] == pytest.approx([3.8, 3.0, 1.0, -0.5, -2.0], abs=1e-4)


def test_fewer_than_three_determined_slices_give_no_plateau() -> None:
    layout = _layout()
    theta, abscissae = _synthetic(layout, drop=("vs_1_1", "vs_1_2"))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, _SLICES, _FORMS)
    assert law["status"] == STATUS_INSUFFICIENT
    assert [row["determined"] for row in law["slices"]] == [False, False]


def test_four_determined_slices_draw_a_law_of_at_most_three_parameters() -> None:
    """Four determined slices: the hyperbola of four parameters does not compete, a smaller form
    goes through them and the zero is defined (decision 5 of the passport standard)."""
    layout = _layout_of(160, 64, 32)
    plateaus = {1: 4.0, 2: 3.0, 3: 1.0, 4: 0.5}
    theta, abscissae = _grid_only(layout, _free_plateaus(plateaus))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, _SLICES, _FORMS)
    assert law["status"] == STATUS_MEASURED
    assert all(row["determined"] for row in law["slices"])
    assert len(law["plateau"].form.parameters) <= 3


def test_on_a_hyperbolic_plateau_the_levels_lie_on_the_law() -> None:
    """The plateau of the surface is the hyperbola itself: the plateaus of the slices lie on
    it, the plateau hyperbola returns it, and the shift is its horizontal asymptote μ₀ = 4."""
    layout = _layout_of(640, 64, 32)
    theta, abscissae = _grid_only(layout, _hyperbolic_surface)
    names = list(theta)
    covariance = 0.01 * np.eye(len(names))
    law = fit_vs_law(theta, covariance, names, abscissae, layout, _SLICES, _FORMS)
    assert law["status"] == STATUS_MEASURED
    plateau = law["plateau"]
    assert plateau.form.name == "slanted_hyperbolic"
    assert plateau.shift() == pytest.approx(4.0, abs=1e-5)
    assert law["slice_law"].shared[1] == pytest.approx(6.0, abs=1e-5)
    assert plateau.params[1:] == pytest.approx([-1.5, 1.5, 0.5], abs=1e-4)
    for row in law["slices"]:
        assert row["determined"]
        expected = float(plateau_level(4.0, -1.5, 1.5, 0.5, row["u"])) - 4.0
        assert row["mu_v"] == pytest.approx(expected, abs=1e-5)
        assert row["law"] == pytest.approx(row["mu_v"], abs=1e-5)
        assert row["se"] > 0.0
        assert row["se_fit"] > 0.0
    assert plateau.choice.fit.held == ()
    assert plateau.choice.fit.chi2 < 1e-8
    assert len(plateau.bins) == 6
    assert delta_se(plateau.rows[1], covariance) > 0.0


def test_the_grid_law_sensitivity_is_the_first_order_response() -> None:
    """Both steps: moving one cell by δ moves the parameters by their rows times δ."""
    layout = _layout_of(640, 64, 32)
    theta, abscissae = _grid_only(layout, _hyperbolic_surface)
    names = list(theta)
    covariance = 0.01 * np.eye(len(names))
    base = fit_vs_law(theta, covariance, names, abscissae, layout, _SLICES, _FORMS)
    delta = 1e-3
    moved = dict(theta)
    moved[names[5]] += delta
    shifted = fit_vs_law(moved, covariance, names, abscissae, layout, _SLICES, _FORMS)
    change = np.concatenate(
        [
            shifted["slices_params"] - base["slices_params"],
            shifted["plateau"].params - base["plateau"].params,
        ]
    )
    rows = np.vstack([base["slices_rows"], base["plateau"].rows])
    assert change == pytest.approx(rows[:, 5] * delta, abs=1e-5)


def test_a_sharp_plateau_corner_on_a_slice_is_held() -> None:
    """The plateaus of the eight length slices of the published measurement (``ū``, ``μ_v`` in
    the coding of the fit, their own SE): the data want a corner sharper than the bound, and it
    settles on the slice 161-320 words. The fit is a cone in ``(u_V, W_V)`` there: an optimizer
    that keeps the parameters strictly inside the range creeps to the bound of the width and does
    not converge. The law must be drawn, with the width held on its lower bound.
    """
    u = np.array([1.950, 2.787, 3.777, 4.703, 5.692, 6.627, 7.541, 8.549])
    c = np.array([14.330, 13.416, 14.385, 12.231, 10.919, 9.091, 7.852, 6.485])
    se = np.array([1.180, 0.708, 0.796, 0.601, 0.542, 0.592, 0.734, 1.051])
    law = fit_plateau_law(u, c, np.eye(len(u)), np.diag(se**2), [_HYPERBOLA])
    assert not isinstance(law, str)
    assert law.choice.fit.held == (3,)
    assert law.params[3] == pytest.approx(0.001, abs=1e-6)
    assert not np.any(law.rows[3])
