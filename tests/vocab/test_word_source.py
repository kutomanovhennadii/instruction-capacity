"""Mirror tests of the word source filters that need no NLTK corpora.

The filter functions take the lemmatizer as a parameter, so they are checked on a
substitute mapping without access to WordNet (no network).
"""

import pytest

from iec.vocab.word_source import (
    _count_syllables,
    _is_ascii_lower_word,
    _is_long_enough,
    _is_singular,
)

# Substitute lemmas: on the left a form from the corpus, on the right what WordNetLemmatizer
# would return.
LEMMAS = {
    "arms": "arm",
    "stones": "stone",
    "wolves": "wolf",
    "acts": "act",
    # Words that merely end in -s: the lemmatizer leaves them as they are.
    "glass": "glass",
    "chess": "chess",
    "news": "news",
    "lens": "lens",
    "arm": "arm",
    "stone": "stone",
}


def _lemmatize(word: str) -> str:
    """Return the substitute lemma of the word."""
    return LEMMAS[word]


@pytest.mark.parametrize("word", ["arm", "stone", "glass", "chess", "news", "lens"])
def test_singular_forms_are_kept(word: str) -> None:
    assert _is_singular(word, _lemmatize) is True


@pytest.mark.parametrize("word", ["arms", "stones", "wolves", "acts"])
def test_plural_forms_are_dropped(word: str) -> None:
    assert _is_singular(word, _lemmatize) is False


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("lake", True),
        ("Lake", False),
        ("re-do", False),
        ("don't", False),
        ("x2", False),
        ("café", False),
    ],
)
def test_ascii_lower_filter(word: str, expected: bool) -> None:
    assert _is_ascii_lower_word(word) is expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # Letter names and English function words are unfit as replacement targets:
        # they occur in the text of the experiment instructions themselves.
        ("a", False),
        ("i", False),
        ("in", False),
        ("do", False),
        ("or", False),
        ("ct", False),
        ("act", True),
        ("lake", True),
    ],
)
def test_min_length_filter_drops_letters_and_function_words(word: str, expected: bool) -> None:
    assert _is_long_enough(word) is expected


@pytest.mark.parametrize(
    ("pronunciation", "expected"),
    [
        (["L", "EY1", "K"], 1),  # lake
        (["AH0", "B", "AW1", "T"], 2),  # about
        (["B", "K"], 0),  # no vowels
    ],
)
def test_syllable_count_uses_stress_digits(pronunciation: list[str], expected: int) -> None:
    assert _count_syllables(pronunciation) == expected
