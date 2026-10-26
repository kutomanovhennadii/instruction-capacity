"""Titles and labels of the passport document: sections and public quantities.

A number and its name are kept apart: a quantity carries its key and substitutions, the text
comes from here. The keys are those of ``passport_data.json`` (``approximation.public``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Label:
    """Name, unit and note template of a quantity; substitutions ``{q}`` and its ``args``."""

    title: str
    unit: str
    note: str


SECTION_TITLES: dict[str, str] = {
    "document": "Measurement passport",
    "header": "Totals",
    "capacities": "Capacities and loads",
    "figures": "Figures",
    "model_params": "Model parameters",
    "diagnostics": "Fit diagnostics",
    "collapse": "Collapse of the answer",
    "bin_cube": "Bin cube filling",
}

METHODS: dict[str, str] = {"bins": "binned", "law": "by law"}

WORDS: dict[str, str] = {
    "quantity": "Quantity",
    "value": "Value ± 1.96·SE",
    "unit": "Unit",
    "status": "Status",
    "note": "Note",
    "parameter": "Parameter",
    "undefined": "undefined",
    "no_public": (
        "Public quantities are not computed: approximation laws are not measured for every "
        "block (status `{status}`)."
    ),
}

QUANTITIES: dict[str, Label] = {
    "iec_capacity": Label(
        "IEC",
        "rules",
        "how many rules the model holds above level {q}; the gap between the two calculations "
        "measures the approximation",
    ),
    "igc_capacity": Label(
        "IGC",
        "rules",
        "how many rules the model holds above level {q} when every word of the text is "
        "replaced (p = 1); the gap between the two calculations measures the approximation",
    ),
    "fatigue_price_per_doubling": Label(
        "price of a position doubling",
        "logit/doubling",
        "logit lost per doubling of the word position on the fatigue slope",
    ),
    "fatigue_capacity_words": Label(
        "FEC",
        "words",
        "word position where accumulated fatigue exhausts the level-{q} reserve; texts "
        "measured up to {support} words",
    ),
    "volume_price_per_doubling": Label(
        "price of a volume doubling",
        "logit/doubling",
        "logit lost when the text length doubles",
    ),
    "loss_at_full_density": Label(
        "loss at full density",
        "logit",
        "logit lost when every word of the text is replaced (p = 1)",
    ),
    "rule_depth_degradation_factor": Label(
        "late-rule degradation factor",
        "times/doubling",
        "how many times the degradation grows when the rule number doubles",
    ),
    "rule_depth_scale_A_R": Label(
        "late-rule degradation scale",
        "logit",
        "coefficient of the power part of the rule-number law",
    ),
    "rule_set_discount_per_doubling": Label(
        "rule-set discount per doubling",
        "logit/doubling",
        "level change when the number of rules in the table doubles",
    ),
}


def quantity_label(key: str) -> Label:
    """Return the label of a quantity; an unknown key - the key itself without unit and note."""
    return QUANTITIES.get(key, Label(key, "", ""))


class _Defaulting(dict[str, Any]):
    """Substitutions of a template: a missing one prints as a dash instead of failing."""

    def __missing__(self, key: str) -> str:
        return "—"


def format_note(template: str, level: float | None, args: dict[str, Any]) -> str:
    """Substitute the level ``q`` and the arguments of a quantity into a note template."""
    values = _Defaulting(args)
    values["q"] = "—" if level is None else f"{level:g}"
    return template.format_map(values)
