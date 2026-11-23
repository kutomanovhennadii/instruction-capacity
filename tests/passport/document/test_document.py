"""The figures and the document of the passport: what they write and what they say."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from iec.passport.document.charts.figure_plan import load_figure_plan
from iec.passport.document.charts.figure_writer import write_passport_figures
from iec.passport.document.renderer import render_passport, write_documents
from iec.passport.document.text_cells import law_cell
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config

CONFIG = Path(__file__).resolve().parents[3] / "config"
ASSIGNMENT = build_law_assignment(load_law_assignment_config(CONFIG / "passport_laws.yaml"))
PLAN = load_figure_plan(CONFIG / "passport_figures.yaml")


def _axis(reference: float = 1.0) -> list[dict[str, Any]]:
    rows = []
    for center, status in ((reference, "reference"), (2.0, "law"), (4.0, "law"), (8.0, "outlier")):
        rows.append(
            {
                "name": f"b_{center}",
                "center": center,
                "status": status,
                "raw": -0.1 * center,
                "own_half": 0.2,
                "restored": -0.1 * center - 0.5,
                "restored_half": 0.3,
                "law": -0.1 * center,
            }
        )
    return rows


def _grid() -> list[dict[str, Any]]:
    rows = []
    for v_bin in (3, 4):
        for s_bin, x in ((1, 5.0), (2, 15.0), (3, 30.0)):
            rows.append(
                {
                    "name": f"vs_{v_bin}_{s_bin}",
                    "v_bin": v_bin,
                    "s_bin": s_bin,
                    "words": [41, 80],
                    "x": x,
                    "u": 2.0,
                    "status": "law",
                    "raw": 10.0 - x / 10.0,
                    "raw_half": 0.5,
                    "restored": -x / 10.0,
                    "restored_half": 0.6,
                    "law": 10.0 - x / 10.0,
                }
            )
    rows.append(
        {
            "name": "vs_3_4",
            "v_bin": 3,
            "s_bin": 4,
            "words": [41, 80],
            "x": 60.0,
            "u": 2.0,
            "status": "excluded",
        }
    )
    return rows


def _row(key: str, level: float, value: float, status: str = "measured") -> dict[str, Any]:
    return {
        "key": key,
        "level": level,
        "method": "law",
        "value": value,
        "interval": [value / 2.0, value * 2.0],
        "plus_minus": value / 2.0,
        "status": status,
    }


def _fit(sigma2: float | None = 2.0, method: str = "glmm") -> dict[str, Any]:
    return {
        "method": method,
        "sigma2": sigma2,
        "points": 10,
        "trials": 100,
        "failures": 5,
        "parameters": 7,
        "dropped": ["vs_1_1"],
    }


def _law(
    form: str, formula: str, params: dict[str, float], chi2: float, n_points: int
) -> dict[str, Any]:
    """A published law of a block: the form the data chose among four candidates."""
    return {
        "law_form": form,
        "formula": formula,
        "params": params,
        "half": dict.fromkeys(params, 0.1),
        "held": [],
        "chi2": chi2,
        "n_points": n_points,
        "free": len(params),
        "candidates": [
            {"form": name, "converged": True, "chi2": chi2, "free": 2, "criterion": chi2 + 4.0}
            for name in ("linear", "slanted_hyperbolic", "binary_exponential", "saturation")
        ],
    }


HYPERBOLA = "μ + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]"


def _hyperbola(mu: float, kappa: float, corner: float, omega: float) -> dict[str, float]:
    return {"mu": mu, "kappa": kappa, "corner": corner, "omega": omega}


def _sync() -> dict[str, Any]:
    return {
        "model": "VS-PM",
        "weights": "V x P x M",
        "fit": _fit(),
        "status": "measured",
        "laws": {
            # The law of the slices is a law with its chosen form: the shared parameters of the curve.
            "grid": {
                **_law(
                    "asymptotic_hyperbolic",
                    "[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2",
                    {"kappa": -3.0, "lambda": 40.0, "omega": 1.5},
                    10.0,
                    6,
                ),
                "n_parameters": 5,
            },
            "grid_plateau": _law(
                "slanted_hyperbolic", HYPERBOLA, _hyperbola(20.0, -3.0, 5.0, 0.001), 1.0, 2
            ),
            "density": {
                **_law("slanted_hyperbolic", HYPERBOLA, _hyperbola(15.0, -9.0, 9.0, 4.0), 20.0, 3),
                "at_one": -9.0,
            },
            "table_size": _law("linear", "a + b·log₂ m", {"a": 0.4, "b": -0.3}, 2.0, 3),
        },
        "shifts": {
            "delta_VS": {"value": 20.0, "half": 1.0},
            "delta_P": {"value": 15.0, "half": 2.0},
            "delta_M": {"value": 0.4, "half": 0.1},
        },
        "alpha": {"value": 35.0, "half": 2.0},
        "capacities": [
            _row("fatigue_capacity_words", 0.5, 14000.0, "extrapolated"),
        ],
        "bins": {
            "grid": _grid(),
            "slices": [
                {"v_bin": 3, "u": 2.0, "mu_v": -1.0, "se_fit": 0.5, "determined": True},
                {"v_bin": 4, "u": 3.0, "mu_v": -2.0, "se_fit": 0.5, "determined": True},
                {"v_bin": 5, "u": 4.0, "determined": False},
            ],
            "P": _axis(0.25),
            "M": _axis(),
        },
    }


def _rule_word() -> dict[str, Any]:
    return {
        "model": "VS-PMR",
        "weights": "V x P x M",
        "fit": _fit(),
        "status": "measured",
        "laws": {
            "grid": {
                **_law("binary_exponential", "μ − A·2^(γ·ℓ)", {"A": 0.5, "gamma": 0.3}, 300.0, 6),
                "n_parameters": 4,
            },
            "grid_plateau": _law("linear", "a + b·log₂(v/20)", {"a": 13.0, "b": -0.9}, 6.0, 2),
            "density": _law(
                "slanted_hyperbolic", HYPERBOLA, _hyperbola(2.0, -2.0, 7.0, 1.0), 11.0, 3
            ),
            "table_size": _law("linear", "a + b·log₂ m", {"a": 0.3, "b": -0.2}, 1.0, 3),
            "rule_position": _law(
                "slanted_hyperbolic", HYPERBOLA, _hyperbola(-0.4, -0.6, 3.8, 1.0), 8.0, 3
            ),
        },
        "shifts": {
            "mu_0": {"value": 13.0, "half": 0.5},
            "mu_P": {"value": 2.0, "half": 0.3},
            "lambda_M": {"value": 0.3, "half": 0.1},
            "lambda_R": {"value": -0.4, "half": 0.7},
        },
        "alpha": {"value": 15.0, "half": 1.4},
        "capacities": [
            _row("instruction_capacity", 0.5, 32.0),
            {
                "key": "instruction_capacity",
                "level": 0.5,
                "method": "bins",
                "value": 33.0,
                "status": "measured",
            },
            {
                "key": "fatigue_capacity_words",
                "level": 0.5,
                "method": "law",
                "value": None,
                "status": "no_root",
            },
        ],
        "bins": {
            "grid": _grid(),
            "slices": [
                {"v_bin": 3, "u": 2.0, "mu_v": 11.0, "half": 1.0},
                {"v_bin": 4, "u": 3.0, "mu_v": 10.0, "half": 1.0},
            ],
            "P": _axis(0.25),
            "M": _axis(),
            "R": _axis(),
        },
    }


def _background() -> dict[str, Any]:
    pair = {"value": 1.0, "half": 0.5}
    return {
        "model": "V-(s/v)-PM",
        "weights": "V x P x M",
        "fit": _fit(),
        "status": "measured",
        "laws": {
            "V": {
                **_law("linear", "a + b·log₂(v/20)", {"a": 9.4, "b": -0.45}, 3.0, 4),
                "at_20": 9.4,
            },
            "q": {
                **_law("exponential", "c + A·e^(−q/τ)", {"c": -0.2, "A": 1.0, "tau": 0.55}, 2.0, 4),
                "at_0": 0.8,
                "at_1": -0.04,
            },
            "P": {
                **_law(
                    "log_linear_load",
                    "a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))",
                    {"a": 1.1, "b": -0.03, "c": -3.3},
                    5.0,
                    4,
                ),
                "p_top": 0.0009765625,
                "at_top": 1.1,
                "at_one": -2.5,
            },
            "M": _law("linear", "a + b·log₂ m", {"a": 0.12, "b": -0.48}, 1.0, 4),
        },
        "shifts": {"V(20)": pair, "q(0)": pair, "delta_P": pair, "M(1)": pair},
        "alpha": {"value": 11.34, "half": 1.2},
        "edge_rules": 128,
        "edge_words": 10240,
        "capacities": [
            _row("instruction_capacity", 0.5, 6.7e5, "extrapolated"),
            {
                "key": "instruction_capacity",
                "level": 0.5,
                "method": "bins",
                "value": None,
                "status": "beyond_support",
                "lower_bound": 128,
                "reserve_left": 2.2,
            },
            _row("fatigue_capacity_words", 0.5, 5.2e5, "extrapolated"),
        ],
        "bins": {"V": _axis(40.0), "q": _axis(0.05), "P": _axis(0.25), "M": _axis()},
    }


def _passport() -> dict[str, Any]:
    return {
        "model": {"id": "model-x", "experiment_id": "replay"},
        "points": 10,
        "P_sync": _sync(),
        "H": _rule_word(),
        "F": _background(),
    }


def test_all_figures_are_written(tmp_path: Path) -> None:
    written = write_passport_figures(_passport(), tmp_path, ASSIGNMENT, PLAN)
    assert len(written) == 24
    assert all(path.stat().st_size > 0 for path in written)
    assert (tmp_path / "F" / "answer-share-after-correction.png").is_file()
    assert (tmp_path / "H" / "R-before-correction.png").is_file()
    assert (tmp_path / "P_sync" / "plateau-before-correction.png").is_file()


def test_the_figures_of_an_earlier_build_are_removed(tmp_path: Path) -> None:
    stale = tmp_path / "H" / "spread-M.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an earlier build")
    written = write_passport_figures(_passport(), tmp_path, ASSIGNMENT, PLAN)
    assert not stale.exists()
    assert sorted(path.resolve() for path in tmp_path.rglob("*.png")) == sorted(
        path.resolve() for path in written
    )


def test_quantity_without_laws_gets_no_figures(tmp_path: Path) -> None:
    passport = _passport()
    passport["H"] = {"status": "no_laws"}
    written = write_passport_figures(passport, tmp_path, ASSIGNMENT, PLAN)
    assert len(written) == 15
    assert not (tmp_path / "H").exists()


def test_document_marks_a_continuation_and_the_bins() -> None:
    text = render_passport(_passport(), PLAN)
    assert "14,000 (7,000–28,000), continuation of the law" in text
    assert "6.7·10⁵" in text
    assert "at least 128, reserve left 2.20" in text
    assert "none: the law gives no root" in text
    assert "`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`" in text
    assert "figures/F/answer-share-before-correction.png" in text
    assert "odds of an error" not in text
    assert "The capacities are taken at the level 0.5." in text
    assert "Threshold at σ²" not in text
    assert "the reserve `Δ" not in text


def test_every_figure_of_the_document_is_written(tmp_path: Path) -> None:
    path, written = write_documents(_passport(), tmp_path, ASSIGNMENT, PLAN)
    text = path.read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert path.name == "passport.md"
    assert len(links) == len(written) == 24
    assert all((tmp_path / link).is_file() for link in links)


def test_document_says_when_a_quantity_has_no_laws() -> None:
    passport = _passport()
    passport["P_sync"] = {
        "model": "VS-PM",
        "weights": "V x P x M",
        "status": "no_laws",
        "fit": _fit(None, "none"),
    }
    text = render_passport(passport, PLAN)
    assert "There are no laws" in text
    assert "the fit did not converge" in text
    assert "There are no capacities: the quantity has no laws." in text


def test_document_warns_about_a_fit_without_the_shift() -> None:
    passport = _passport()
    passport["P_sync"]["fit"] = _fit(None, "logit")
    text = render_passport(passport, PLAN)
    assert "the logit without the shift of a request" in text
    assert "not comparable" in text


def test_document_has_no_capacities_by_the_product() -> None:
    text = render_passport(_passport(), PLAN)
    assert "By the product" not in text
    assert "Capacities by a rule word" not in text


def test_document_names_a_parameter_of_h_on_a_bound() -> None:
    passport = _passport()
    passport["H"]["laws"]["density"]["held"] = ["omega"]
    assert "on the bound of the search: omega" in render_passport(passport, PLAN)


def test_document_writes_the_exponent_of_a_slice() -> None:
    text = render_passport(_passport(), PLAN)
    # The slices choose their form: the text writes the formula of the chosen form over ℓ = log₂ s.
    assert "`μ − A·2^(γ·ℓ)`" in text
    assert "`A = 0.500 ± 0.100`" in text
    assert "parabola" not in text


def test_document_writes_the_exponent_of_the_share_of_the_answer() -> None:
    text = render_passport(_passport(), PLAN)
    assert "The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`" in text
    assert "`tau = 0.55 ± 0.10`" in text
    assert "A fall of 0.84 over the whole answer" in text


def test_document_names_the_maximum_of_the_density() -> None:
    passport = _passport()
    passport["F"]["shifts"]["delta_P"] = {"value": 1.1, "half": 0.5}
    text = render_passport(passport, PLAN)
    assert "`α_F = V(20) + q(0) + δ_P + M(1) = 11.34 ± 1.20`" in text
    assert "at `p = 0.0010`" in text
    assert "`δ_P` = 1.10 ± 0.50" in text


def _stress() -> dict[str, Any]:
    def row(value: float, half: float) -> dict[str, float]:
        return {"value": value, "half": half}

    def kind(errors: float, length: float, density: float, rules: float) -> dict[str, Any]:
        return {
            "text_errors": row(errors, 0.4),
            "influence": {
                "length": row(length, 2.0),
                "density": row(density, 1.5),
                "rules": row(rules, 0.3),
            },
        }

    return {
        "protocol": {"density": 0.5, "words_per_rule": 50, "point_error": 0.5},
        "status": "measured",
        "support": {"words": 10240.0, "rules": 256.0},
        "words": row(910.0, 40.0),
        "rules": row(18.2, 0.8),
        "combined_error_at_point": 50.0,
        "structure": {
            "synchronization": row(95.7, 1.1),
            "replacements": row(4.3, 1.1),
            "background": row(0.8, 0.2),
        },
        "synchronization": kind(11.0, 63.2, 36.1, 0.7),
        "replacements": kind(1.7, 83.6, 3.7, 12.7),
        "background": kind(0.5, 56.6, 17.1, 26.2),
        "combined_text_errors": row(12.4, 0.9),
    }


def test_document_writes_the_stress_test() -> None:
    text = render_passport({**_passport(), "stress_test": _stress()}, PLAN)
    assert "## The standard stress test" in text
    assert (
        "The standard stress test describes how a model behaves as the complexity of a task with fixed "
        "proportions of the load increases step by step." in text
    )
    assert (
        "with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them "
        "repeated 25 times." in text
    )
    assert "1. The point of 50%: T = 910 ± 40 words, 18.2 ± 0.8 rules." in text
    assert "2. The combined error at the point: 50%." in text
    assert (
        "3. The structure of the error at the point, per 100 errors: synchronization 95.7 ± 1.1, "
        "replacements 4.3 ± 1.1; background over them, per the same 100: 0.8 ± 0.2." in text
    )
    assert (
        "5. Replacements: errors over the text for the median request 1.7 ± 0.4%; influence per 100 "
        "errors of replacements at the point: length 83.6 ± 2.0, density 3.7 ± 1.5, rules 12.7 ± 0.3."
        in text
    )
    assert "7. The combined error over the text for the median request: 12.4 ± 0.9%." in text


def test_document_puts_the_capacities_and_the_stress_test_before_the_quantities() -> None:
    text = render_passport({**_passport(), "stress_test": _stress()}, PLAN)
    assert text.index("## The capacities") < text.index("## The standard stress test")
    assert text.index("## The standard stress test") < text.index("## The formula")
    assert text.index("## The formula") < text.index("\n## P_sync")
    assert "| Capacity | q |" not in text[text.index("## The formula") :]


def test_document_says_when_the_stress_test_has_no_point() -> None:
    stress = {"protocol": _stress()["protocol"], "status": "below_one_rule"}
    text = render_passport({**_passport(), "stress_test": stress}, PLAN)
    assert (
        "There is no point: the combined error of the last word reaches one half already at one rule."
        in text
    )


def test_document_writes_an_influence_that_is_not_defined() -> None:
    stress = _stress()
    stress["background"]["influence"]["length"] = {"value": None, "half": None}
    text = render_passport({**_passport(), "stress_test": stress}, PLAN)
    assert "length not defined" in text


def _spread() -> dict[str, Any]:
    rows = [
        {"bins": [1], "center": 2.0, "share": 0.5, "failing": 40, "sigma2": 8.0, "half": 1.0},
        {"bins": [2], "center": 8.0, "share": 0.5, "failing": 40, "sigma2": 2.0, "half": 0.5},
    ]
    return {"parameters": 3, "loglik_gain": 663.5, "V": rows, "P": rows, "M": rows}


def test_a_quantity_with_the_spread_has_its_table_and_figures(tmp_path: Path) -> None:
    passport = _passport()
    passport["H"]["fit"] = {**_fit(method="glmm_heterogeneous"), "spread": _spread()}
    text = render_passport(passport, PLAN)
    assert "whose variance depends on the load" in text
    assert "GLMM with heterogeneous random-intercept variance" in text
    assert "`2·Δ ln L = 663.5`" in text
    assert "| number of rules | 8 | 0.50 | 2.00 ± 0.50 |" in text
    assert "figures/H/spread-M.png" in text
    path, written = write_documents(passport, tmp_path, ASSIGNMENT, PLAN)
    assert len(written) == 27
    assert (tmp_path / "figures" / "H" / "spread-V.png").is_file()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
    assert len(links) == len(written)


def test_the_table_of_the_capacities_has_no_threshold() -> None:
    text = render_passport(_passport(), PLAN)
    assert "| Capacity | q | By the law | By the bins |" in text
    assert (
        "| fatigue capacity, words | 0.5 | 14,000 (7,000–28,000), continuation of the law | - |"
    ) in text


def test_a_bound_past_the_largest_float_is_written_as_more_than_its_limit() -> None:
    row = {
        "key": "fatigue_capacity_words",
        "level": 0.5,
        "method": "law",
        "value": 2.0e9,
        "interval": [1.0e-290, None],
        "plus_minus": 1.4e12,
        "status": "extrapolated",
    }
    assert law_cell(row) == "2.0·10⁹ (0.0–more than 10³⁰⁸), continuation of the law"
    assert law_cell({**row, "value": None, "interval": [None, None]}) == (
        "more than 10³⁰⁸ (more than 10³⁰⁸–more than 10³⁰⁸), continuation of the law"
    )


def _spread_laws() -> dict[str, Any]:
    return {
        "V": {"form": "line", "params": {"a": 2.2, "b": -0.054}, "chi2": 2.5, "n_points": 7},
        "P": {"form": "line", "params": {"a": 2.19, "b": 0.165}, "chi2": 24.8, "n_points": 11},
        "M": {
            "form": "hyperbola",
            "params": {"c": 0.063, "A": 18.9, "x0": 0.73},
            "chi2": 8.6,
            "n_points": 9,
        },
    }


def test_the_spread_shows_its_approximations(tmp_path: Path) -> None:
    passport = _passport()
    passport["H"]["fit"] = {**_fit(method="glmm_heterogeneous"), "spread": _spread()}
    passport["H"]["spread_laws"] = _spread_laws()
    text = render_passport(passport, PLAN)
    assert "The threshold of a capacity reads the variance" not in text
    assert "serve the eye." in text
    assert (
        "| number of rules | `c + A/(1 + m/m₀)` | `c = 0.06`, `A = 18.90`, `m₀ = 0.73` | "
        "8.6 over 9 groups |"
    ) in text
    assert (
        "| text length, words | `a + b·log₂ v` | `a = 2.20`, `b = −0.054` | 2.5 over 7 groups |"
    ) in text
    _, written = write_documents(passport, tmp_path, ASSIGNMENT, PLAN)
    assert len(written) == 27
    assert (tmp_path / "figures" / "H" / "spread-M.png").is_file()


def test_a_degenerate_hyperbola_is_shown_as_the_inverse_law(tmp_path: Path) -> None:
    passport = _passport()
    passport["H"]["fit"] = {**_fit(method="glmm_heterogeneous"), "spread": _spread()}
    laws = _spread_laws()
    laws["M"] = {"form": "inverse", "params": {"c": 9.03, "A": 10.3}, "chi2": 9.0, "n_points": 9}
    passport["H"]["spread_laws"] = laws
    text = render_passport(passport, PLAN)
    assert "| number of rules | `c + A/m` | `c = 9.03`, `A = 10.30` | 9.0 over 9 groups |" in text
    _, written = write_documents(passport, tmp_path, ASSIGNMENT, PLAN)
    assert (tmp_path / "figures" / "H" / "spread-M.png").is_file()


def test_p_sync_has_the_law_and_the_figures_of_the_number_of_rules(tmp_path: Path) -> None:
    passport = _passport()
    text = render_passport(passport, PLAN)
    assert "The law of the number of rules is a line, `a + b·log₂ m`" in text
    # The form with the smallest QAIC is chosen.
    assert "chosen by the smallest `QAIC` among 4 forms" in text
    assert "`α = δ_VS + δ_P + δ_M =" in text
    write_passport_figures(passport, tmp_path, ASSIGNMENT, PLAN)
    assert (tmp_path / "P_sync" / "M-before-correction.png").is_file()
    assert (tmp_path / "P_sync" / "M-after-correction.png").is_file()


def test_p_sync_shows_its_instruction_capacity() -> None:
    passport = _passport()
    passport["P_sync"]["capacities"] = [
        {
            "key": "instruction_capacity",
            "level": 0.5,
            "method": "law",
            "value": None,
            "status": "no_slope",
        },
        {
            "key": "instruction_capacity",
            "level": 0.5,
            "method": "bins",
            "value": None,
            "status": "beyond_support",
            "lower_bound": 128,
            "reserve_left": 35.0,
        },
        *passport["P_sync"]["capacities"],
    ]
    text = render_passport(passport, PLAN)
    sync = text[text.index("### P_sync") : text.index("### H")]
    assert (
        "| instruction capacity, rules | 0.5 | none: the law gives no root | "
        "at least 128, reserve left 35.00 |"
    ) in sync
    assert "The instruction capacity is the number of rules" in sync
