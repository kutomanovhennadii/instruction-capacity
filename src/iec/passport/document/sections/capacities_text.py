"""The capacities of the three quantities in the passport: the part that a user reads first.

The part stands right after the header of the document, before the standard stress test and the
sections of the quantities: the zero point of a capacity and the level 0.5 at which the capacities
are taken, then for every quantity its table of the capacities by the laws and next to them by the
bins, with what its capacities are. The laws that the capacities come from stand in the sections of
the quantities below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from iec.passport.document.sections.background_text import TITLE as BACKGROUND_TITLE
from iec.passport.document.sections.rule_word_text import TITLE as RULE_WORD_TITLE
from iec.passport.document.sections.sync_text import TITLE as SYNC_TITLE
from iec.passport.document.text_cells import capacity_rows

#: The quantities in the order of the document: the section, its title, what its capacities are.
QUANTITIES = (
    (
        "P_sync",
        SYNC_TITLE,
        "The instruction capacity is the number of rules at which the law of the number of rules "
        "brings the logit to zero, and next to it the walk along the bins of that axis; by the "
        "law it is none when the fall of that law over its bins does not differ from zero. The fatigue "
        "capacity is the word at which the full law of the grid at the density going to zero and one "
        "rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.",
    ),
    ("H", RULE_WORD_TITLE, ""),
    (
        "F",
        BACKGROUND_TITLE,
        "The instruction capacity is the number of rules at the zero point. The fatigue capacity is the "
        "number of the last word of a text of the same length at the zero point.",
    ),
)


def capacities_lines(passport: Mapping[str, Any]) -> list[str]:
    """Return the part of the capacities of the three quantities.

    Args:
        passport: The passport dictionary with the sections ``P_sync``, ``H`` and ``F``.
    """
    lines = [
        "## The capacities",
        "",
        "The zero point of a capacity is a text of 20 words, the start of the answer, the density at the "
        "maximum of its law and one rule; every other axis of the capacity is set by its definition.",
        "",
        "The capacities are taken at the level 0.5. At it the share of successes over all requests and the "
        "success of the median request reach one half at the same logit, zero, whatever the variance of the "
        "request shift: a capacity is the load at which the laws bring the logit of the median request to "
        "zero, and it does not depend on that variance.",
        "",
    ]
    for key, title, meaning in QUANTITIES:
        section = passport[key]
        lines += [f"### {title}", ""]
        if section["status"] != "measured":
            lines += ["There are no capacities: the quantity has no laws.", ""]
            continue
        lines += [*capacity_rows(section["capacities"]), ""]
        if meaning:
            lines += [meaning, ""]
    return lines
