"""Conditions of one measurement point: the task axes and the three seeds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PointConditions:
    """The six numbers that fully determine the task of one point.

    ``D`` is the number of distinct words of the source text, ``n`` how many times each of
    them occurs, ``m`` the number of substitution rules. ``seed_rules`` draws the rule list,
    ``seed_text`` shuffles the source text, ``seed_order`` orders the rule lines: ``0`` keeps
    the drawn order, any other value permutes the lines with that seed. Together with the
    vocabulary these numbers determine the rule list, the source text and the prompt byte
    for byte, and they identify the point inside a measurement.

    A passive structure: consistency (``m <= D``, a large enough vocabulary) is checked by
    the generators, which hold the vocabulary.
    """

    D: int  # noqa: N815 - protocol axis name
    n: int
    m: int
    seed_rules: int
    seed_text: int
    seed_order: int = 0

    @property
    def v(self) -> int:
        """Return the source text length in words, ``v = D * n``."""
        return self.D * self.n

    @property
    def identity(self) -> tuple[int, int, int, int, int, int]:
        """Return the identity of the point inside a measurement."""
        return (self.D, self.n, self.m, self.seed_rules, self.seed_text, self.seed_order)

    def as_dict(self) -> dict[str, int]:
        """Return the serializable snapshot of the conditions."""
        return {
            "D": self.D,
            "n": self.n,
            "m": self.m,
            "seed_rules": self.seed_rules,
            "seed_text": self.seed_text,
            "seed_order": self.seed_order,
        }
