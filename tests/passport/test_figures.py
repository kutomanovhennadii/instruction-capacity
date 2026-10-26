"""The figures of the passport: every figure of a model is written from the passport alone."""

import json
import math
from pathlib import Path

import numpy as np

from iec.passport.approximation import _vs_curve, approximation_block
from iec.passport.bin_layout import BinLayout, doubling_bounds
from iec.passport.document.figures import FIGURE_NAMES, write_passport_figures
from iec.passport.vspmr_model import VspmrGlmmModel, vs_grid_cells, vspmr_density_p

#: The surface ``κ_S, λ_S, μ₀, κ_V, ω_S, u_*, ω_V`` and the laws of P, M, R of the synthetic data.
_SURFACE = np.array([-2.0, 6.0, 3.0, -1.5, 1.0, 1.5, 0.5])
_P_LAW = (1.0, 3.0, 0.3)  # mu_P, A_P, p_c
_M_LAW = (0.2, 0.3)  # lambda_M, kappa_M
_R_LAW = (0.05, 0.05, 1.2)  # lambda_R, A_R, gamma


def _passport() -> dict:
    """A passport of one model with a measured approximation block on exact synthetic bins."""
    layout = BinLayout(
        volume_bounds=doubling_bounds(20, 320),
        density_bounds=doubling_bounds(1, 64),
        row_bounds=doubling_bounds(1, 32),
        density_reference=3,
    )
    names = list(VspmrGlmmModel().parameter_names(layout))
    abscissae: dict[str, dict[str, float]] = {}
    for volume, position in vs_grid_cells(layout):
        abscissae[f"vs_{volume}_{position}"] = {"x": 2.0 ** (position - 0.25), "u": volume - 1.1}
    for number in range(1, layout.P + 1):
        low, high = vspmr_density_p(layout, number)
        abscissae[f"p_bin_{number}"] = {"p": math.sqrt(low * high)}
    for number in range(2, layout.R + 1):
        low, high = layout.row_bounds[number - 1]
        abscissae[f"m_bin_{number}"] = {"m": math.sqrt(low * high)}
        abscissae[f"r_bin_{number}"] = {"r": math.sqrt(low * high)}
    theta: dict[str, float] = {}
    for name in names:
        coords = abscissae[name]
        if name.startswith("vs_"):
            point = np.array([[math.log2(coords["x"]), coords["u"]]])
            theta[name] = float(_vs_curve(_SURFACE, point)[0])
        elif name.startswith("p_bin_"):
            theta[name] = _P_LAW[0] - _P_LAW[1] * (1.0 - math.exp(-coords["p"] / _P_LAW[2]))
        elif name.startswith("m_bin_"):
            theta[name] = _M_LAW[0] + _M_LAW[1] * math.log2(coords["m"])
        else:
            theta[name] = _R_LAW[0] - _R_LAW[1] * (coords["r"] ** _R_LAW[2] - 1.0)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    return {
        "model": {"id": "mock-model", "experiment_id": "demo"},
        "fit": {"vspmr_glmm_v1": {"se": {name: 0.1 for name in theta}}},
        "passport": {"vspmr_glmm_v1": {"approximation": json.loads(json.dumps(block))}},
    }


def test_every_figure_of_a_measured_model_is_written(tmp_path: Path) -> None:
    written = write_passport_figures(_passport(), tmp_path, "mock-model")
    assert list(written) == [f"vspmr_glmm_v1/{name}" for name in FIGURE_NAMES]
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0


def test_a_model_without_shifts_has_no_figures(tmp_path: Path) -> None:
    """Not all laws measured: there is no shift, and neither state of a figure can be drawn."""
    passport = _passport()
    passport["passport"]["vspmr_glmm_v1"]["approximation"]["shifts"] = None
    assert write_passport_figures(passport, tmp_path, "mock-model") == {}
