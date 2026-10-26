"""Rule list generator: random pairs of vocabulary words from ``seed_rules``."""

from __future__ import annotations

import random

from iec.contracts import PointConditions, Rule, RuleTable, Vocab
from iec.infrastructure.exceptions import ConfigError, VocabError
from iec.infrastructure.telemetry import TelemetryContext, telemetry_step

_MODULE_NAME = "rule_generator"


def _telemetry_context(self: RuleGenerator, params: PointConditions) -> TelemetryContext:
    """Extract the point identity for the telemetry event of rule generation."""
    return TelemetryContext(m=params.m, seed_rules=params.seed_rules, seed_order=params.seed_order)


class RuleGenerator:
    """Generator of the rule list of one task.

    ``2m`` distinct words are drawn from the vocabulary in one draw of
    ``random.Random(seed_rules)`` over the sorted word list (sorting is required: a set
    yields words in an unpredictable order, which would break reproducibility). The first
    half of the draw are the words to replace, the second half their replacements, so the
    two groups never intersect.

    ``seed_order`` sets the order in which the same rules are shown to the model: ``0``
    keeps the drawn order, any other value permutes the lines with
    ``random.Random(seed_order)``. The pairs depend only on ``seed_rules``, so a different
    order of the same pairs separates the cost of a rule position from the difficulty of
    a particular word pair. The same ``(seed_rules, seed_order)`` gives the same list to
    the letter.
    """

    def __init__(self, vocab: Vocab) -> None:
        """Check the vocabulary and prepare the generator.

        Args:
            vocab: Vocabulary from which the words of both groups are drawn.

        Raises:
            VocabError: the vocabulary is empty - there is nothing to draw from.
        """
        if not vocab.words:
            raise VocabError(
                f"rule generation requires a non-empty vocab, got '{vocab.vocab_id}' with 0 words",
                error_code="RULES_EMPTY_VOCAB",
            )
        self._vocab = vocab
        self._sorted_words = sorted(vocab.words)

    @telemetry_step("rules_generate", module_name=_MODULE_NAME, context_extractor=_telemetry_context)
    def generate(self, params: PointConditions) -> RuleTable:
        """Build a list of ``params.m`` rules from ``params.seed_rules``.

        Args:
            params: Point conditions; the generator reads ``m``, ``seed_rules`` and
                ``seed_order``.

        Returns:
            ``RuleTable`` with ``m`` rules in the order shown to the model; all ``2m``
            words are distinct and the two groups do not intersect.

        Raises:
            ConfigError: ``m`` is below 1 (``RULES_M_INVALID``), ``seed_order`` is negative
                (``RULES_SEED_ORDER_INVALID``) or the vocabulary cannot hold ``2m``
                distinct words (``RULES_VOCAB_TOO_SMALL``).
        """
        if params.m < 1:
            raise ConfigError(
                f"rule count m must be >= 1, got {params.m}",
                error_code="RULES_M_INVALID",
            )
        if params.seed_order < 0:
            raise ConfigError(
                f"order seed must be >= 0, got {params.seed_order}",
                error_code="RULES_SEED_ORDER_INVALID",
            )
        if params.m > self._vocab.max_pairs:
            raise ConfigError(
                f"vocab '{self._vocab.vocab_id}' fits at most {self._vocab.max_pairs} "
                f"disjoint pairs ({self._vocab.size} words), got m={params.m}",
                error_code="RULES_VOCAB_TOO_SMALL",
            )

        rng = random.Random(params.seed_rules)
        drawn = rng.sample(self._sorted_words, 2 * params.m)
        rules = tuple(
            Rule(source=source, target=target)
            for source, target in zip(drawn[: params.m], drawn[params.m :], strict=True)
        )
        return RuleTable(
            rules=self._permute(rules, params),
            vocab_id=self._vocab.vocab_id,
            seed_rules=params.seed_rules,
        )

    def _permute(self, rules: tuple[Rule, ...], params: PointConditions) -> tuple[Rule, ...]:
        """Permute the lines with ``seed_order``; ``0`` keeps the drawn order.

        With ``seed_order = 0`` no random generator is created at all. A permuted order
        equal to the drawn one is not an error: with ``m = 1`` every permutation is the
        identity, and with small ``m`` a coincidence is simply likely.
        """
        if params.seed_order == 0:
            return rules
        order = random.Random(params.seed_order).sample(range(params.m), params.m)
        return tuple(rules[index] for index in order)
