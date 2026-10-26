"""The figures of every model of the passport, a folder per model, drawn from the passport alone.

Five pairs of figures, each before and after the shift of its block: the cells of the length ×
position grid by length slice with the slice curves (shared ``κ_S``, ``λ_S``, ``ω_S``, own
plateau ``μ_v``); the plateaus ``μ_v`` by length bin with the plateau hyperbola through them; the
bins of the density, table-size and rule-number axes with the law of the axis. "Before the
shift" is the raw coefficients of the bins - the coding of the fit; "after the shift" is minus
the shift of the block - the law at the zero of its axis, moved into ``α_phys``: for the grid the
horizontal asymptote ``δ_VS = μ₀`` of the plateau hyperbola, for the axes the law at ``p → 0``,
``m = 1``, ``r = 1``. The whiskers are ±1.96 own SE of a point: the SE of the coefficient of a
bin, the SE of a plateau in the coding of the fit; the error of the shift is common to all points
of a block and is not drawn. The bin on which the fit fixed the arbitrary constant of its block
has no error of its own. Nothing is refitted.

The collapse of the answer (the ``collapse`` block) has two pairs of figures of its own, beside
the model folders, drawn the same way as the axes of a model: the bins of the logit of the
collapse by text length and by replacement density with the law of the axis, before and after
the shift of the axis - for the length the law at ``v = 20`` words, for the density the law at
``p → 0``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from iec.passport.bin_layout import REFERENCE_TOTAL_WORDS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: The P, M, R axes: law key, restored rows, coefficient prefix, letter, name, abscissa.
_AXES: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "density",
        "density_profile",
        "p_bin_",
        "P",
        "density of replacements",
        "p = m/D - replacement density (centre of mass of the bin)",
    ),
    (
        "table_size",
        "table_size_profile",
        "m_bin_",
        "M",
        "number of rules in the table",
        "m - rules in the table (centre of mass of the bin)",
    ),
    (
        "rule_position",
        "rule_position_profile",
        "r_bin_",
        "R",
        "rule number",
        "r - number of the rule in the table (centre of mass of the bin)",
    ),
)

#: File names of the figures of a model, in the order of the passport.
FIGURE_NAMES: tuple[str, ...] = (
    "vs-slices-before-shift.png",
    "vs-slices-after-shift.png",
    "plateau-before-shift.png",
    "plateau-after-shift.png",
    "P-before-shift.png",
    "P-after-shift.png",
    "M-before-shift.png",
    "M-after-shift.png",
    "R-before-shift.png",
    "R-after-shift.png",
)

#: The axes of the collapse of the answer, in the form of ``_AXES``.
_COLLAPSE_AXES: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "collapse_length",
        "length_profile",
        "v_bin_",
        "V",
        "collapse by text length",
        "v = D·n - text length, words (centre of mass of the bin)",
    ),
    (
        "collapse_density",
        "density_profile",
        "p_bin_",
        "P",
        "collapse by density",
        "p = m/D - replacement density (centre of mass of the bin)",
    ),
)

#: File names of the collapse figures, in the order of the passport.
COLLAPSE_FIGURE_NAMES: tuple[str, ...] = (
    "collapse-V-before-shift.png",
    "collapse-V-after-shift.png",
    "collapse-P-before-shift.png",
    "collapse-P-after-shift.png",
)

#: The axes of the place of the collapse, in the form of ``_AXES``.
_PLACE_AXES: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "place_word",
        "word_profile",
        "s_bin_",
        "S",
        "collapse place by word number",
        "s - number of the word in the answer (centre of mass of the bin)",
    ),
    (
        "place_density",
        "density_profile",
        "p_bin_",
        "P",
        "collapse place by density",
        "p = m/D - replacement density (centre of mass of the bin)",
    ),
    (
        "place_share",
        "share_profile",
        "q_bin_",
        "Q",
        "collapse place by the share of the text",
        "(s − 1)/v - share of the text before the word (centre of mass of the bin)",
    ),
)

#: File names of the figures of the place of the collapse, in the order of the passport.
PLACE_FIGURE_NAMES: tuple[str, ...] = tuple(
    f"place-{letter}-{state}-shift.png"
    for letter in ("S", "P", "Q")
    for state in ("before", "after")
)

#: Axes drawn on a linear abscissa; the others are on a log₂ abscissa.
_LINEAR_AXES = frozenset({"place_share"})

#: The name of the shift of every axis.
_SHIFT_NAMES = {
    "density": "μ_P",
    "table_size": "λ_M",
    "rule_position": "λ_R",
    "collapse_length": "λ_Vψ",
    "collapse_density": "μ_Pψ",
    "place_word": "μ_SF",
    "place_density": "μ_PF",
    "place_share": "λ_QF",
}
#: Where the curve of a law starts: the zero of its axis, or ``None`` - the first bin.
_CURVE_START: dict[str, float | None] = {
    "density": None,
    "table_size": 1.0,
    "rule_position": 1.0,
    "collapse_length": float(REFERENCE_TOTAL_WORDS),
    "collapse_density": None,
    "place_word": 1.0,
    "place_density": None,
    "place_share": 0.0,
}
#: Half-width of the whiskers in SE.
_HALF = 1.96
_LN2 = math.log(2.0)
_CURVE = "#d62728"
#: The two states of every figure: before and after the shift of its block.
_STATES = ((False, "before"), (True, "after"))


def _value_text(value: float, se: float | None) -> str:
    """Write ``value ± 1.96·SE`` or ``value (held on its bound)``."""
    return f"{value:.4g} (held on its bound)" if se is None else f"{value:.4g} ± {_HALF * se:.2g}"


def _slice_curve(
    kappa_s: float, lambda_s: float, plateau: float, omega_s: float, x: np.ndarray
) -> np.ndarray:
    """A slice: ``[(κ_S·ℓ + λ_S) + μ_v − √((κ_S·ℓ + λ_S − μ_v)² + κ_S²ω_S²)]/2``, ``ℓ = log₂ s``."""
    line = kappa_s * np.log2(x) + lambda_s
    gap = line - plateau
    result: np.ndarray = 0.5 * (
        line + plateau - np.sqrt(gap * gap + kappa_s * kappa_s * omega_s * omega_s)
    )
    return result


def _plateau(mu_0: float, kappa_v: float, u_star: float, omega_v: float, u: np.ndarray) -> np.ndarray:
    """The plateau hyperbola ``μ₀ + (κ_V/2)·[(u − u_*) + √((u − u_*)² + ω_V²)]``."""
    t = u - u_star
    result: np.ndarray = mu_0 + 0.5 * kappa_v * (t + np.sqrt(t * t + omega_v * omega_v))
    return result


def _axis_law(key: str, law: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """The raw law of an axis: ``P = μ_P − A_P(1 − e^(−p/p_c))``, ``M = λ_M + κ_M log₂ m``,
    ``R = λ_R − A_R(r^γ − 1)``; of the collapse ``V_ψ = λ_Vψ + κ_ψ log₂(v/20)`` and
    ``P_ψ = μ_Pψ + A_ψ p^γ_ψ``."""
    if key == "collapse_length":
        return float(law["lambda_Vpsi"]) + float(law["kappa_psi"]) * np.log2(
            x / REFERENCE_TOTAL_WORDS
        )
    if key == "collapse_density":
        return float(law["mu_Ppsi"]) + float(law["A_psi"]) * np.power(x, float(law["gamma_psi"]))
    if key == "place_word":
        return _plateau(
            float(law["mu_SF"]),
            float(law["kappa_SF"]),
            math.log2(float(law["s_star_F"])),
            float(law["omega_SF"]),
            np.log2(x),
        )
    if key == "place_density":
        return float(law["mu_PF"]) + float(law["A_F"]) * np.power(x, float(law["gamma_F"]))
    if key == "place_share":
        return float(law["lambda_QF"]) + float(law["kappa_QF"]) * x
    if key == "density":
        result: np.ndarray = float(law["mu_P"]) - float(law["A_P"]) * (
            1.0 - np.exp(-x / float(law["p_c"]))
        )
        return result
    if key == "table_size":
        return float(law["lambda_M"]) + float(law["kappa_M"]) * np.log2(x)
    return float(law["lambda_R"]) - float(law["A_R"]) * (np.power(x, float(law["gamma"])) - 1.0)


def _law_text(key: str, law: dict[str, Any], shifted: bool) -> str:
    """The formula of the law of an axis with its parameters (± 1.96·SE), raw or minus the shift."""
    se = law.get("se") or {}

    def v(name: str) -> str:
        return _value_text(float(law[name]), se.get(name))

    if key == "collapse_length":
        head = (
            "V_ψ(v) − λ_Vψ = κ_ψ·log₂(v/20)" if shifted else "V_ψ(v) = λ_Vψ + κ_ψ·log₂(v/20)"
        )
        body = f"κ_ψ = {v('kappa_psi')}"
        body = body if shifted else f"λ_Vψ = {v('lambda_Vpsi')}, " + body
    elif key == "collapse_density":
        head = "P_ψ(p) − μ_Pψ = A_ψ·p^γ_ψ" if shifted else "P_ψ(p) = μ_Pψ + A_ψ·p^γ_ψ"
        body = f"A_ψ = {v('A_psi')}, γ_ψ = {v('gamma_psi')}"
        body = body if shifted else f"μ_Pψ = {v('mu_Ppsi')}, " + body
    elif key == "place_word":
        corner = "(κ_SF/2)·[log₂(s/s_*) + √(log₂²(s/s_*) + ω_SF²)]"
        head = f"S_F(s) − μ_SF = {corner}" if shifted else f"S_F(s) = μ_SF + {corner}"
        body = f"κ_SF = {v('kappa_SF')}, s_* = {v('s_star_F')}, ω_SF = {v('omega_SF')}"
        body = body if shifted else f"μ_SF = {v('mu_SF')}, " + body
    elif key == "place_density":
        head = "P_F(p) − μ_PF = A_F·p^γ_F" if shifted else "P_F(p) = μ_PF + A_F·p^γ_F"
        body = f"A_F = {v('A_F')}, γ_F = {v('gamma_F')}"
        body = body if shifted else f"μ_PF = {v('mu_PF')}, " + body
    elif key == "place_share":
        head = "Q_F − λ_QF = κ_QF·(s − 1)/v" if shifted else "Q_F = λ_QF + κ_QF·(s − 1)/v"
        body = f"κ_QF = {v('kappa_QF')}"
        body = body if shifted else f"λ_QF = {v('lambda_QF')}, " + body
    elif key == "density":
        head = "P(p) − μ_P = −A_P·(1 − e^(−p/p_c))" if shifted else "P(p) = μ_P − A_P·(1 − e^(−p/p_c))"
        body = f"A_P = {v('A_P')}, p_c = {v('p_c')}"
        body = body if shifted else f"μ_P = {v('mu_P')}, " + body
    elif key == "table_size":
        head = "M(m) − λ_M = κ_M·log₂ m" if shifted else "M(m) = λ_M + κ_M·log₂ m"
        body = f"κ_M = {v('kappa_M')}" if shifted else f"λ_M = {v('lambda_M')}, κ_M = {v('kappa_M')}"
    else:
        head = "R(r) − λ_R = −A_R·(r^γ − 1)" if shifted else "R(r) = λ_R − A_R·(r^γ − 1)"
        body = f"A_R = {v('A_R')}, γ = {v('gamma')}"
        body = body if shifted else f"λ_R = {v('lambda_R')}, " + body
    return f"{head}:\n{body}"


def _grid_data(block: dict[str, Any], fit_se: dict[str, Any]) -> dict[str, Any] | None:
    """The numbers of the grid figures; raw = published + the shift of the grid.

    ``None`` - the grid law is not measured or it has no cell or no determined slice.
    """
    law = (block.get("laws") or {}).get("vs_grid") or {}
    if law.get("status") != "measured":
        return None
    shift = float(law["delta_VS"])
    cells = [
        {
            **row,
            "raw": float(row["value"]) + shift,
            "raw_se": float(fit_se[f"vs_{row['v_bin']}_{row['s_bin']}"]),
        }
        for row in (block.get("restored") or {}).get("vs_grid") or []
        if row.get("value") is not None
    ]
    slices = [s for s in law.get("slices") or [] if s.get("determined")]
    if not cells or not slices:
        return None
    return {
        "law": law,
        "shift": shift,
        "shift_se": law["se"]["delta_VS"],
        "cells": cells,
        "slices": slices,
    }


def _draw_slices(data: dict[str, Any], shifted: bool, head: str, out: Path) -> None:
    """All length slices on one image: cells joined by a line, slice curves dashed."""
    law, shift = data["law"], data["shift"]
    offset = shift if shifted else 0.0  # displayed = raw − offset
    kappa_s, omega_s = float(law["kappa_S"]), float(law["omega_S"])
    lambda_raw = float(law["lambda_S"]) + shift
    plateau_raw = {s["v_bin"]: float(s["mu_v"]) + shift for s in data["slices"]}
    v_bins = sorted({c["v_bin"] for c in data["cells"]})
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(v_bins)))
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    for color, v in zip(colors, v_bins, strict=True):
        pts = sorted((c for c in data["cells"] if c["v_bin"] == v), key=lambda c: c["x"])
        lo, hi = pts[0]["words"]
        ax.errorbar(
            [c["x"] for c in pts],
            [c["raw"] - offset for c in pts],
            yerr=[_HALF * c["raw_se"] for c in pts],
            fmt="o-",
            lw=1.2,
            color=color,
            ms=5,
            capsize=2,
            elinewidth=0.8,
            label=f"text {lo}–{hi} words",
        )
        if v in plateau_raw:
            x = np.geomspace(1.0, max(c["x"] for c in pts) * 1.2, 200)
            ax.plot(
                x,
                _slice_curve(kappa_s, lambda_raw - offset, plateau_raw[v] - offset, omega_s, x),
                color=color,
                lw=1.6,
                ls="--",
            )
    ax.plot(
        [],
        [],
        color="black",
        lw=1.6,
        ls="--",
        label=f"slice curve: κ_S = {kappa_s:.2f}, ω_S = {omega_s:.2f}, own plateau μ_v",
    )
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("s - word position in the answer")
    ax.set_ylabel(
        "logit, shifted by the asymptote of the plateau"
        if shifted
        else "logit, raw coefficient of the bin"
    )
    ax.set_title(
        f"{head}: cells of the length × position grid by length slice, whiskers ±1.96·SE",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _draw_plateau(data: dict[str, Any], shifted: bool, head: str, out: Path) -> None:
    """The plateaus by length bin with the plateau hyperbola through them."""
    law, shift = data["law"], data["shift"]
    offset = shift if shifted else 0.0
    slices = data["slices"]
    u = np.array([float(s["u"]) for s in slices])
    levels = np.array([float(s["mu_v"]) + shift - offset for s in slices])
    se = np.array([float(s["se_fit"]) for s in slices])
    kappa_v, v_star, omega_v = (float(law[k]) for k in ("kappa_V", "v_star", "omega_V"))
    u_star = math.log2(v_star / REFERENCE_TOTAL_WORDS)
    mu_0 = shift - offset
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.errorbar(
        u + 1.0,
        levels,
        yerr=_HALF * se,
        fmt="o",
        color="black",
        ms=5,
        capsize=3,
        label="plateau of the slice μ_v ± 1.96·SE",
    )
    grid_u = np.linspace(0.0, float(u.max()) + 0.5, 300)
    v_star_se = law["se"].get("v_star")
    corner_range = ""
    if v_star_se is not None:
        # The interval of the corner is that of log₂ v_*: SE(u_*) = SE(v_*)/(ln 2·v_*).
        spread = _HALF * float(v_star_se) / (_LN2 * v_star)
        corner_range = f" ({v_star * 2 ** -spread:.0f}–{v_star * 2**spread:.0f})"
    width = (
        "ω_V → 0"
        if "omega_V" in (law.get("held") or [])
        else f"ω_V = {_value_text(omega_v, law['se'].get('omega_V'))}"
    )
    level = "μ₀ − δ_VS = 0" if shifted else f"μ₀ = {_value_text(mu_0, data['shift_se'])}"
    ax.plot(
        grid_u + 1.0,
        _plateau(mu_0, kappa_v, u_star, omega_v, grid_u),
        color=_CURVE,
        lw=1.5,
        label=(
            f"plateau hyperbola: {level}, κ_V = {_value_text(kappa_v, law['se'].get('kappa_V'))},\n"
            f"corner v* = {v_star:.0f} words{corner_range}, {width}"
        ),
    )
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    ax.set_xticks([s["v_bin"] for s in slices])
    ax.set_xticklabels(
        [f"V{s['v_bin']}\n{s['words'][0]}–{s['words'][1]}" for s in slices],
        fontsize=7,
        rotation=35,
    )
    ax.set_xlabel(
        "length bin V - text length, words (a point stands at the mean length of its slice)"
    )
    ax.set_ylabel(
        "plateau level μ_v, logit, shifted" if shifted else "plateau level μ_v, logit, raw"
    )
    ax.set_title(
        f"{head}: plateau level μ_v by length bin and the plateau hyperbola, whiskers ±1.96·SE",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _draw_axis(
    block: dict[str, Any],
    fit_se: dict[str, Any],
    axis: tuple[str, str, str, str, str, str],
    shifted: bool,
    head: str,
    out: Path,
) -> None:
    """The bins of one axis with its law, before or after the shift of the block."""
    key, rows_key, prefix, letter, name, xlabel = axis
    law = block["laws"][key]
    shift = float(block["shifts"][key]["value"])
    shift_se = float(block["shifts"][key]["se"])
    offset = 0.0 if shifted else shift  # displayed = restored + offset
    rows = [r for r in block["restored"][rows_key] if r.get("value") is not None]
    own = [r for r in rows if f"{prefix}{r['bin']}" in fit_se]
    fixed = [r for r in rows if f"{prefix}{r['bin']}" not in fit_se]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.errorbar(
        [float(r["x"]) for r in own],
        [float(r["value"]) + offset for r in own],
        yerr=[_HALF * float(fit_se[f"{prefix}{r['bin']}"]) for r in own],
        fmt="o",
        color="black",
        ms=5,
        capsize=3,
        label="bin ± 1.96·SE of its coefficient",
    )
    if fixed:
        ax.plot(
            [float(r["x"]) for r in fixed],
            [float(r["value"]) + offset for r in fixed],
            "o",
            mfc="white",
            mec="black",
            ms=6,
            label="bin on which the fit fixed the arbitrary constant, no error of its own",
        )
    x = np.array([float(r["x"]) for r in rows])
    start = _CURVE_START[key]
    low = float(x.min()) if start is None else start
    linear = key in _LINEAR_AXES
    curve_x = (np.linspace if linear else np.geomspace)(low, float(x.max()), 300)
    ax.plot(
        curve_x,
        _axis_law(key, law, curve_x) - (shift if shifted else 0.0),
        color=_CURVE,
        lw=1.5,
        label=_law_text(key, law, shifted),
    )
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    if not shifted:
        ax.axhline(
            shift,
            color=_CURVE,
            lw=0.9,
            ls=":",
            label=f"shift - the law at the zero of the axis: {_SHIFT_NAMES[key]} = "
            f"{_value_text(shift, shift_se)}",
        )
    if not linear:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "logit, shifted by the law at the zero of the axis"
        if shifted
        else "logit, raw coefficient of the bin"
    )
    state = "after the shift" if shifted else "before the shift"
    ax.set_title(
        f"{head}: {name} {letter} - bins and the law, {state}, whiskers ±1.96·SE", fontsize=10
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_passport_figures(
    passport: dict[str, Any], out_dir: Path, title: str
) -> dict[str, Path]:
    """Draw the figures of every model of the passport, a folder per model, and the collapse.

    Args:
        passport: The passport dictionary (the one written to JSON).
        out_dir: Folder of the figures; a model writes into ``out_dir/<model>``, the collapse
            figures lie in ``out_dir`` itself.
        title: The head of every figure title - the measured language model.

    Returns:
        Paths of the figures by ``<model>/<file name>``, the collapse figures by file name.
        A model without the shifts of its blocks (not all laws are measured) has no figures:
        before and after the shift both need the shift; the grid figures need a measured grid
        law with a determined slice, a figure of an axis - a measured law of the axis. The
        collapse figures need the shifts of the ``collapse`` block, the same way.
    """
    written: dict[str, Path] = {}
    fits = passport.get("fit") or {}
    for model, section in (passport.get("passport") or {}).items():
        fit_se = (fits.get(model) or {}).get("se") or {}
        figures = _model_figures(section or {}, fit_se, out_dir / model, title)
        written.update({f"{model}/{name}": path for name, path in figures.items()})
    collapse = passport.get("collapse")
    if isinstance(collapse, dict) and isinstance(collapse.get("shifts"), dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        fit_se = (collapse.get("fit") or {}).get("se") or {}
        for axis in _COLLAPSE_AXES:
            for shifted, state in _STATES:
                name = f"collapse-{axis[3]}-{state}-shift.png"
                written[name] = out_dir / name
                _draw_axis(collapse, fit_se, axis, shifted, title, written[name])
        place = collapse.get("place")
        if isinstance(place, dict) and isinstance(place.get("shifts"), dict):
            place_se = (place.get("fit") or {}).get("se") or {}
            for axis in _PLACE_AXES:
                for shifted, state in _STATES:
                    name = f"place-{axis[3]}-{state}-shift.png"
                    written[name] = out_dir / name
                    _draw_axis(place, place_se, axis, shifted, title, written[name])
    return written


def _model_figures(
    section: dict[str, Any], fit_se: dict[str, Any], out_dir: Path, head: str
) -> dict[str, Path]:
    """Draw the figures of one model section; return their paths by file name."""
    block = section.get("approximation")
    if not isinstance(block, dict) or not isinstance(block.get("shifts"), dict):
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    grid = _grid_data(block, fit_se)
    if grid is not None:
        for kind, draw in (("vs-slices", _draw_slices), ("plateau", _draw_plateau)):
            for shifted, state in _STATES:
                name = f"{kind}-{state}-shift.png"
                written[name] = out_dir / name
                draw(grid, shifted, head, written[name])
    for axis in _AXES:
        if (block["laws"].get(axis[0]) or {}).get("status") != "measured":
            continue
        for shifted, state in _STATES:
            name = f"{axis[3]}-{state}-shift.png"
            written[name] = out_dir / name
            _draw_axis(block, fit_se, axis, shifted, head, written[name])
    return written
