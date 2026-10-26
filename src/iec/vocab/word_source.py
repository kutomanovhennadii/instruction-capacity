"""Source of vocabulary candidates on NLTK: the CMU dictionary, syllables, nouns, lemmas.

The module is the only boundary of the vocabulary builder with the NLTK corpora (CMU
Pronouncing Dictionary and WordNet). Missing corpora are downloaded once; a plain set of
strings goes out, so the ``vocab_builder`` core does not depend on NLTK and is tested
without the network.
"""

from __future__ import annotations

from collections.abc import Callable

from iec.infrastructure.exceptions import VocabError
from iec.infrastructure.logging import LogContext, get_logger
from iec.vocab.vocab_config import VocabBuildConfig

_MODULE_NAME = "word_source"
_REQUIRED_CORPORA = ("cmudict", "wordnet", "omw-1.4")

# Minimum length of a target word. One- and two-letter entries of the corpus are letter
# names ("a".."z"), abbreviations ("ct", "lb", "mt") and English function words ("be", "do",
# "in", "it", "no", "or"). The latter are dangerous: they occur in the text of the experiment
# instructions themselves, so the rule "replace in with stone" would measure a defect of the
# bench, not the instruction capacity. The threshold 3 cuts off all three groups with one
# condition, without a stop list.
MIN_WORD_LENGTH = 3

_logger = get_logger(LogContext(module_name=_MODULE_NAME))


def _ensure_corpora() -> None:
    """Download the required NLTK corpora if they are missing (idempotent).

    Side effects:
        May download NLTK data into the local cache on first access (needs the network).
    """
    import nltk

    for corpus in _REQUIRED_CORPORA:
        nltk.download(corpus, quiet=True)


def _count_syllables(pronunciation: list[str]) -> int:
    """Return the syllable count by the ARPAbet transcription (vowels carry a stress digit)."""
    return sum(1 for phoneme in pronunciation if phoneme[-1].isdigit())


def _is_ascii_lower_word(word: str) -> bool:
    """Return whether the word is lowercase ASCII letters only (no hyphens, apostrophes, digits)."""
    return word.isascii() and word.isalpha() and word.islower()


def _is_long_enough(word: str) -> bool:
    """Return whether the word reaches the minimum target length (:data:`MIN_WORD_LENGTH`)."""
    return len(word) >= MIN_WORD_LENGTH


def _is_singular(word: str, lemmatize: Callable[[str], str]) -> bool:
    """Return whether the word equals its noun lemma.

    The lemmatizer turns plural forms into the singular (``arms`` -> ``arm``), so a mismatch
    means the plural. Words that merely end in ``-s`` (``glass``, ``chess``, ``news``,
    ``lens``) stay their own lemma.
    """
    return lemmatize(word) == word


def load_noun_monosyllables(config: VocabBuildConfig) -> set[str]:
    """Collect the set of candidate words for the vocabulary ``V``.

    The source pipeline: lowercase ASCII letters -> not shorter than
    :data:`MIN_WORD_LENGTH` letters -> at most ``config.max_syllables`` syllables -> only
    words that have a noun sense (WordNet) -> only the singular. The word count is logged
    before and after each filter.

    Plural forms are dropped on purpose: the vocabulary supplies the protocol with
    substitution pairs, and a pair like ``arm``/``arms`` in one set blurs the replacement -
    the model may apply the rule to another form of the same word, and the measurement would
    catch a morphological confusion instead of the instruction capacity. The length threshold
    cuts off letter names and function words for the reason given at
    :data:`MIN_WORD_LENGTH`.

    Args:
        config: Build parameters (number of syllables, ASCII flag).

    Returns:
        The set of candidate words in canonical form (lowercase ASCII, without a space).

    Raises:
        VocabError: an NLTK corpus is unavailable or empty
            (``error_code="VOCAB_SOURCE_UNAVAILABLE"``).

    Side effects:
        On the first run downloads the NLTK corpora (see :func:`_ensure_corpora`).
    """
    try:
        _ensure_corpora()
        from nltk.corpus import cmudict, wordnet
        from nltk.stem import WordNetLemmatizer

        entries: dict[str, list[list[str]]] = cmudict.dict()
        lemmatizer = WordNetLemmatizer()
    except Exception as cause:
        raise VocabError.from_cause(
            "failed to load NLTK corpora (cmudict/wordnet)",
            cause,
            error_code="VOCAB_SOURCE_UNAVAILABLE",
        ) from cause

    total = len(entries)
    _logger.info(f"cmudict entries: {total}")
    if total == 0:
        raise VocabError(
            "cmudict corpus is empty",
            error_code="VOCAB_SOURCE_UNAVAILABLE",
        )

    if config.ascii_lower_only:
        ascii_words = {word for word in entries if _is_ascii_lower_word(word)}
    else:
        ascii_words = set(entries)
    _logger.info(f"after ascii-lower filter: {len(ascii_words)}")

    long_enough = {word for word in ascii_words if _is_long_enough(word)}
    _logger.info(f"after min-length filter: {len(long_enough)}")

    monosyllabic = {
        word
        for word in long_enough
        if any(_count_syllables(pron) <= config.max_syllables for pron in entries[word])
    }
    _logger.info(f"after monosyllabic filter: {len(monosyllabic)}")

    nouns = {word for word in monosyllabic if wordnet.synsets(word, pos=wordnet.NOUN)}
    _logger.info(f"after noun filter: {len(nouns)}")

    def to_noun_lemma(word: str) -> str:
        return str(lemmatizer.lemmatize(word, pos=wordnet.NOUN))

    singular = {word for word in nouns if _is_singular(word, to_noun_lemma)}
    _logger.info(f"after singular filter: {len(singular)}")

    return singular
