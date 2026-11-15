"""The rule table ``R_m``: pairs "word to replace -> replacement" in a meaningful order."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rule:
    """One replacement rule: ``source`` occurs in the text, ``target`` in the answer."""

    source: str
    target: str

    def as_dict(self) -> dict[str, str]:
        """Return a serializable snapshot of the rule."""
        return {"source": self.source, "target": self.target}


@dataclass(frozen=True, slots=True)
class RuleTable:
    """The replacement table shown to the model.

    The order of the ``rules`` tuple is the order shown to the model; the position of a
    rule in the list is a measured axis of the experiment, so the order is significant and
    is kept as is. The construction invariants (all ``2m`` words are distinct, the source
    and target groups do not intersect, the words come from vocabulary ``vocab_id``) are
    ensured by the rule generator - the contract only carries the result.
    """

    rules: tuple[Rule, ...]
    vocab_id: str
    seed_rules: int

    @property
    def m(self) -> int:
        """Return the number of rules in the table."""
        return len(self.rules)

    @property
    def sources(self) -> tuple[str, ...]:
        """Return the words to replace in table order."""
        return tuple(rule.source for rule in self.rules)

    @property
    def targets(self) -> tuple[str, ...]:
        """Return the replacement words in table order."""
        return tuple(rule.target for rule in self.rules)

    def as_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the table, keeping the order of the rules."""
        return {
            "rules": [rule.as_dict() for rule in self.rules],
            "vocab_id": self.vocab_id,
            "seed_rules": self.seed_rules,
            "m": self.m,
        }
