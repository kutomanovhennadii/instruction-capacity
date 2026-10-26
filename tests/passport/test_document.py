"""The passport document renders from a passport dictionary in the order of the sections."""

from iec.passport.document.renderer import render_passport

_PASSPORT = {
    "model": {"id": "mock-model", "experiment_id": "demo"},
    "fit": {
        "vspmr_glmm_v1": {
            "sigma2": 0.5,
            "sigma2_se": 0.1,
            "sigma2_method": "inside the fit",
            "shift_in_fit": True,
            "rho": 0.13,
            "cell_weighting": None,
        }
    },
    "passport": {
        "vspmr_glmm_v1": {
            "approximation": {
                "status": "not_measured",
                "reason": "fit has no covariance",
                "zero_point": "surface: the first answer word",
            }
        }
    },
    "resources": {
        "unique_point_keys": 4,
        "ok_runs": 3,
        "prompt_tokens_billed": 100,
        "completion_tokens_billed": 50,
        "trials": 30,
        "total_cost_usd": 0.0012,
    },
    "bin_cube": "bin cube: point counts by bins D × m × n",
    "alarms": [
        {
            "severity": "warning",
            "code": "INCOMPLETE_TOKEN_COUNTERS",
            "message": "some billed points carry no provider token counters",
            "evidence": {"records": 1},
            "affects_primary_metrics": False,
        }
    ],
}


def test_the_document_has_its_sections_in_order_and_the_formulas() -> None:
    text = render_passport(_PASSPORT, {})
    titles = [line for line in text.splitlines() if line.startswith("## ")]
    assert titles == [
        "## Totals",
        "## Capacities and loads",
        "## Model parameters",
        "## Fit diagnostics",
        "## Bin cube filling",
    ]
    assert text.startswith("# Measurement passport `demo`")
    assert text.count("$$") == 8
    assert "- money: **$0.0012** - cost of the published points" in text
    assert "**provider counters are incomplete**: 1 billed points" in text
    assert "Reason: fit has no covariance." in text
    assert "between-point variance σ̂² (inside the fit): " in text
    assert "nodes with replicates" not in text
    assert "(`vspmr_glmm_v1`)" in text
    assert [line for line in text.splitlines() if line.startswith("### ")] == [
        "### `vspmr_glmm_v1`"
    ] * 3


def test_every_model_of_the_passport_has_its_subsection() -> None:
    """Two models: a model section has one heading and a subsection per model, in order."""
    passport = {
        **_PASSPORT,
        "fit": {
            **_PASSPORT["fit"],
            "second_model": {
                "sigma2": 1.3,
                "sigma2_se": 0.16,
                "sigma2_method": "outside the fit",
                "shift_in_fit": False,
                "rho": 0.28,
                "cell_weighting": None,
            },
        },
        "passport": {
            **_PASSPORT["passport"],
            "second_model": _PASSPORT["passport"]["vspmr_glmm_v1"],
        },
    }
    text = render_passport(passport, {})
    lines = text.splitlines()
    assert [line for line in lines if line.startswith("## ")] == [
        "## Totals",
        "## Capacities and loads",
        "## Model parameters",
        "## Fit diagnostics",
        "## Bin cube filling",
    ]
    assert [line for line in lines if line.startswith("### ")] == [
        "### `vspmr_glmm_v1`",
        "### `second_model`",
    ] * 3
    assert text.count("$$") == 16
    assert "between-point variance σ̂² (outside the fit): " in text
