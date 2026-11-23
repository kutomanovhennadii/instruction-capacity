"""The text of the standard stress test in the passport: its definition, the numbers at its point, the structure of its errors.

The definition and the reading of the numbers follow the accepted text of the test, ``docs/08-stress-test.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from iec.passport.document.number_text import number

#: The kinds of errors in the order of the text and their names.
KINDS = (
    ("synchronization", "Synchronization"),
    ("replacements", "Replacements"),
    ("background", "Background"),
)
#: The text of a test without a point, by its status.
STATUS_TEXTS = {
    "below_one_rule": "the combined error of the last word reaches one half already at one rule",
    "beyond_range": "the combined error of the last word does not reach one half up to 2²⁰ rules",
    "no_laws": "a quantity has no laws",
}


def _value(row: Mapping[str, Any], digits: int = 1) -> str:
    """Write a number of the test with its half-width; a number the passport holds as ``null`` is not defined."""
    if row.get("value") is None:
        return "not defined"
    if row.get("half") is None:
        return number(float(row["value"]), digits)
    return f"{number(float(row['value']), digits)} ± {number(float(row['half']), digits)}"


def _definition(protocol: Mapping[str, Any]) -> list[str]:
    """Return the definition of the test and the reading of its numbers, by the accepted text of the test.

    Args:
        protocol: The protocol of the test: the density and the number of words per rule.
    """
    density = float(protocol["density"])
    words = int(protocol["words_per_rule"])
    return [
        "The standard stress test describes how a model behaves as the complexity of a task with fixed "
        "proportions of the load increases step by step. The complexity increases until the probability of an "
        "error on the last word of the text reaches 50%. The density of replacements in the test is "
        f"`p = {number(density, 1)}`, and the text has one rule per {words} words: with `m` rules the text has "
        f"`T = {words}·m` words and `D = {1 / density:g}m` distinct words, each of them repeated "
        f"{words * density:g} times. The length of the text and the number of rules increase together, keeping "
        "these proportions.",
        "",
        "The point of 50% is set by the combined error of the last word of the median request, "
        "`E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the "
        "text, and `H` is the conditional probability that the replacement is made correctly on a kept place. "
        "False replacements of background words do not enter the definition of the point, but they are "
        "measured at the point found.",
        "",
        "The influence of each load is set by its price in the logit of the probability of the correct outcome "
        "of the last word. The influence keeps its sign: a negative value means that the load lowers the "
        "probability of an error of that kind. The three shares of the influence sum to 100; when the total "
        "price of the loads is zero or less, the influence is not defined. The mean share of errors over the "
        "text is computed over all its words for the median request. Every number has a 95% confidence "
        "interval by the delta method; the parameters of the three quantities are taken independent, since "
        "the passport has no covariances between its sections.",
        "",
    ]


def stress_test_lines(section: Mapping[str, Any]) -> list[str]:
    """Return the section of the standard stress test.

    Args:
        section: The section ``stress_test`` of the passport.
    """
    lines = ["## The standard stress test", "", *_definition(section["protocol"])]
    if section["status"] not in ("measured", "extrapolated"):
        return [*lines, f"There is no point: {STATUS_TEXTS[section['status']]}.", ""]
    mark = ", a continuation of the laws" if section["status"] == "extrapolated" else ""
    structure = section["structure"]
    lines += [
        f"1. The point of 50%: T = {_value(section['words'], 0)} words, {_value(section['rules'])} rules{mark}.",
        f"2. The combined error at the point: {number(float(section['combined_error_at_point']), 0)}%.",
        "3. The structure of the error at the point, per 100 errors: synchronization "
        f"{_value(structure['synchronization'])}, replacements {_value(structure['replacements'])}; background over "
        f"them, per the same 100: {_value(structure['background'])}.",
    ]
    for position, (key, name) in enumerate(KINDS, start=4):
        kind = section[key]
        influence = kind["influence"]
        lines.append(
            f"{position}. {name}: errors over the text for the median request {_value(kind['text_errors'])}%; "
            f"influence per 100 errors of {name.lower()} at the point: length {_value(influence['length'])}, "
            f"density {_value(influence['density'])}, rules {_value(influence['rules'])}."
        )
    lines += [
        f"7. The combined error over the text for the median request: {_value(section['combined_text_errors'])}%.",
        "",
    ]
    return lines
