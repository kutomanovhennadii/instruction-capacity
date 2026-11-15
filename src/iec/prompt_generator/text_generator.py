"""Generator of the word stream ``X``: a balanced multiset shuffled from ``seed_text``."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Iterable, Sequence

from iec.contracts import PointConditions, RuleTable, SourceText, Vocab
from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry import (
    TASK_GENERATION,
    TelemetryHub,
    point_context,
    telemetry_step,
)
from iec.model_client.tokenizer import Tokenizer

_MODULE_NAME = "text_generator"

# The line width of the stream is a protocol constant: the instrument has one mode of operation.
WORDS_PER_LINE = 10


def replacement_positions(words: Sequence[str], sources: Iterable[str]) -> tuple[int, ...]:
    """Return the indices of the words that the rules replace, in reading order.

    A position is replaced when its word stands on the left side of a rule. The words to replace
    and the replacements never intersect, and replacements never occur in the source text, so the
    markup is unambiguous. The generator marks a new text by this rule, and the point executor
    restores the markup of a published record by the same function.

    Args:
        words: The words of the source text in reading order.
        sources: The left-hand words of the rules.

    Returns:
        The indices of the words to replace, ascending.
    """
    source_set = frozenset(sources)
    return tuple(index for index, word in enumerate(words) if word in source_set)


class TextGenerator:
    """Generator of the source text of one task.

    Exact frequency balancing is achieved by construction: a multiset is assembled in which
    each of the ``D`` lexicon words is repeated exactly ``n`` times, and it is shuffled as a
    whole by ``random.Random(seed_text)``. Shuffling changes the order but not the counts, so
    the frequencies of targets and background are strictly equal and need no checking
    afterwards.

    The lexicon is the ``m`` words to replace from the table plus ``D - m`` background words
    drawn from the sorted vocabulary; replacement words are excluded from the draw, so they
    do not occur in the stream at all - in a correct answer a replacement word appears only
    as the result of a rule. The tokenizer is needed solely to record the fact
    ``len_tokens``.

    ``telemetry`` is the hub that records every generation in the block ``task_generation``
    with the point key of the conditions.
    """

    telemetry: TelemetryHub

    def __init__(
        self, vocab: Vocab, tokenizer: Tokenizer, *, telemetry: TelemetryHub | None = None
    ) -> None:
        """Prepare the generator.

        Args:
            vocab: The vocabulary ``V`` - the source of the background words of the lexicon.
            tokenizer: The tokenizer for one measurement of ``len_tokens`` of the ready text.
            telemetry: Hub that records every generation in the block ``task_generation``;
                without it the generator gets a hub of its own that writes the log line only.
        """
        self._vocab = vocab
        self._tokenizer = tokenizer
        self.telemetry = telemetry if telemetry is not None else TelemetryHub.log_only()
        self._logger = get_logger(LogContext(module_name=_MODULE_NAME))

    @telemetry_step(
        "text_generate",
        block=TASK_GENERATION,
        module_name=_MODULE_NAME,
        context_extractor=point_context,
    )
    def generate(self, params: PointConditions, rule_table: RuleTable) -> SourceText:
        """Build the word stream from the seed ``params.seed_text``.

        Args:
            params: Point conditions; the generator reads ``D`` (distinct words), ``n``
                (occurrences of each) and ``seed_text``; the length ``v`` is derived as
                ``D*n``.
            rule_table: The replacement table; its ``sources`` enter the lexicon, its
                ``targets`` are excluded from the stream.

        Returns:
            ``SourceText``: v words (each of the D words exactly n times), the markup of the
            target positions in reading order, the measured ``len_tokens``.

        Raises:
            ConfigError: D is less than the number of rules or the vocabulary cannot hold
                the lexicon together with the replacement words, D + m > size
                (``TEXT_DISTINCT_INVALID``). The divisibility of the length by the number of
                distinct words need not be checked any more: ``v = D*n`` is a multiple of
                ``D`` by construction of the contract.
        """
        distinct, repeat, m = params.D, params.n, rule_table.m
        if distinct < m or distinct + m > self._vocab.size:
            raise ConfigError(
                f"D={distinct} must satisfy m <= D and D + m <= vocab size "
                f"(m={m}, vocab '{self._vocab.vocab_id}' has {self._vocab.size} words)",
                error_code="TEXT_DISTINCT_INVALID",
            )

        rng = random.Random(params.seed_text)
        lexicon = list(rule_table.sources) + self._draw_background(rule_table, distinct - m, rng)
        stream = [word for word in lexicon for _ in range(repeat)]
        rng.shuffle(stream)
        self._logger.info(
            f"lexicon {distinct} words ({m} sources), {repeat} occurrences each, "
            f"{params.v} words total"
        )

        draft = SourceText(
            words=tuple(stream),
            words_per_line=WORDS_PER_LINE,
            replacement_positions=replacement_positions(stream, rule_table.sources),
            len_tokens=0,
            tokenizer_name=self._tokenizer.name,
            seed_text=params.seed_text,
        )
        # Measured once on the ready text; the line assembly is reused from the contract, so
        # that exactly the form that goes to the model is measured.
        len_tokens = self._tokenizer.count_tokens(draft.text)
        self._logger.info(f"measured len_tokens={len_tokens}")
        return dataclasses.replace(draft, len_tokens=len_tokens)

    def _draw_background(self, rule_table: RuleTable, count: int, rng: random.Random) -> list[str]:
        """Draw background words from the vocabulary without the table words (both groups)."""
        if count == 0:
            return []
        excluded = frozenset(rule_table.sources) | frozenset(rule_table.targets)
        eligible = sorted(self._vocab.words - excluded)
        return rng.sample(eligible, count)
