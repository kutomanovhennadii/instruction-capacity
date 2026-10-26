"""Core of building the vocabulary ``V``: a double filter and an intersection over tokenizers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from iec.contracts import Vocab
from iec.infrastructure.exceptions import VocabError
from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry import telemetry_step
from iec.model_client.tokenizer import Tokenizer
from iec.vocab.vocab_config import VocabBuildConfig
from iec.vocab.word_source import load_noun_monosyllables

_MODULE_NAME = "vocab_builder"

# Source of the initial word set: lowercase ASCII, monosyllabic, nouns.
WordSource = Callable[[VocabBuildConfig], set[str]]

# Single-token status is checked on the word with a leading space: in the BPE vocabularies of
# GPT and Qwen the vocabulary unit is the word after a separator ("▁sword"), the bare
# form is often absent altogether. In the protocol prompt the word also follows a separator,
# so this form is the working one. Measured on 5871 monosyllabic nouns: 1886 words without
# the space against 3284 with it.
PROBE_PREFIX = " "


class VocabBuilder:
    """Builder of the vocabulary ``V`` from a corpus of short English words.

    Pipeline: word source (lowercase ASCII -> monosyllabic -> nouns) -> single-token filter
    per tokenizer -> intersection of the sets -> ``Vocab`` with a deterministic hash. The
    word source is moved outside the module (``word_source``), so the core is tested on an
    arbitrary set without access to NLTK.

    Single-token status is checked on the form with a leading space (:data:`PROBE_PREFIX`) -
    that is how the word looks in the protocol prompt and how it lies in the BPE vocabulary;
    in ``Vocab`` the word is stored without the space. The substitution is made by the
    builder, not by the tokenizer: the probe form is knowledge of the experiment protocol,
    not a provider detail.
    """

    def __init__(
        self,
        config: VocabBuildConfig,
        tokenizers: Sequence[Tokenizer],
        *,
        word_source: WordSource | None = None,
    ) -> None:
        """Check the dependencies and prepare the builder.

        Args:
            config: Build parameters (identifier, source, number of syllables and so on).
            tokenizers: Tokenizers over which the intersection of single-token sets is taken.
            word_source: Source of the initial word set; by default the NLTK corpus
                (CMU Pronouncing Dictionary + WordNet). Passed explicitly in tests.

        Raises:
            VocabError: the tokenizer list is empty - the intersection is undefined.
        """
        if not tokenizers:
            raise VocabError(
                "vocab build requires at least one tokenizer",
                error_code="VOCAB_NO_TOKENIZERS",
            )
        self._config = config
        self._tokenizers = tuple(tokenizers)
        self._word_source: WordSource = (
            word_source if word_source is not None else load_noun_monosyllables
        )
        self._logger = get_logger(LogContext(module_name=_MODULE_NAME))

    @telemetry_step("vocab_build", module_name=_MODULE_NAME)
    def build(self) -> Vocab:
        """Build the vocabulary as the intersection of single-token sets.

        Returns:
            ``Vocab`` with the filtered words, the computed hash and metadata.

        Raises:
            VocabError: the word source is empty or the intersection is empty.
            ProviderError: a tokenizer failure during the single-token check (not
                suppressed).
        """
        base_words = self._word_source(self._config)
        self._logger.info(f"word source produced {len(base_words)} candidate words")
        if not base_words:
            raise VocabError(
                "word source produced no candidate words",
                error_code="VOCAB_EMPTY_SOURCE",
            )

        per_tokenizer_sets = self._single_token_sets(base_words)
        intersection = self._intersect(per_tokenizer_sets)
        self._logger.info(f"intersection size {len(intersection)}")
        if not intersection:
            raise VocabError(
                "single-token intersection is empty across the tokenizer set",
                error_code="VOCAB_EMPTY_INTERSECTION",
            )

        return self._assemble(intersection)

    def _single_token_sets(self, words: set[str]) -> list[set[str]]:
        """Select single-token words per tokenizer separately.

        The form with a leading space is checked (see :data:`PROBE_PREFIX`), while the word
        enters the vocabulary without it. A tokenizer error is propagated (on the adapter
        boundary it is already wrapped in ``ProviderError``); there is no silent skipping of
        words on a failure.
        """
        result: list[set[str]] = []
        for tokenizer in self._tokenizers:
            single = {word for word in words if tokenizer.is_single_token(PROBE_PREFIX + word)}
            self._logger.info(
                f"tokenizer {tokenizer.name}: {len(single)}/{len(words)} single-token words"
            )
            result.append(single)
        return result

    @staticmethod
    def _intersect(sets: list[set[str]]) -> set[str]:
        """Return the intersection of the single-token sets of all tokenizers."""
        return set.intersection(*sets)

    def _assemble(self, words: set[str]) -> Vocab:
        """Assemble the ``Vocab`` contract with the hash and build metadata."""
        frozen = frozenset(words)
        return Vocab(
            vocab_id=self._config.vocab_id,
            words=frozen,
            hash=Vocab.compute_hash(frozen),
            tokenizer_names=tuple(tokenizer.name for tokenizer in self._tokenizers),
            source=f"{self._config.source}_monosyllabic_singular_nouns",
            build_params={
                "max_syllables": self._config.max_syllables,
                "ascii_lower_only": self._config.ascii_lower_only,
            },
        )
