"""A tokenizer wrapper that counts the probes of a vocabulary build and the single-token answers.

The build probes every candidate word with every tokenizer of the set. The counts of the probes
and of the single-token answers are the control numbers of the version reference: the number of
candidates and the number of single-token words per tokenizer. The wrapper answers exactly as the
wrapped tokenizer does and writes a log line every ``PROGRESS_EVERY`` probes, so a slow tokenizer,
such as the count of a Gemini model over the network, shows its progress.
"""

from __future__ import annotations

from iec.infrastructure.logging import LogContext, get_logger
from iec.model_client.tokenizer import Tokenizer

_MODULE_NAME = "counting_tokenizer"

#: A progress line of a tokenizer is written after every this many probes.
PROGRESS_EVERY = 500


class CountingTokenizer:
    """A tokenizer that counts its probes and single-token answers; the answers stay the same.

    It satisfies the ``Tokenizer`` contract, so the vocabulary builder takes it in place of the
    tokenizer it wraps. ``probes`` is the number of probes so far, ``single`` the number of
    single-token answers among them.
    """

    def __init__(self, inner: Tokenizer) -> None:
        """Wrap a tokenizer.

        Args:
            inner: The tokenizer whose answers are counted.
        """
        self._inner = inner
        self.probes = 0
        self.single = 0
        self._logger = get_logger(LogContext(module_name=_MODULE_NAME))

    @property
    def name(self) -> str:
        """Return the versioned name of the wrapped tokenizer."""
        return self._inner.name

    def is_single_token(self, word: str) -> bool:
        """Answer as the wrapped tokenizer does and count the probe and a single-token answer.

        Args:
            word: The probed string, checked as is.

        Returns:
            The answer of the wrapped tokenizer.

        Side effects:
            Writes a progress line after every ``PROGRESS_EVERY`` probes.
        """
        result = self._inner.is_single_token(word)
        self.probes += 1
        self.single += int(result)
        if self.probes % PROGRESS_EVERY == 0:
            self._logger.info(f"tokenizer {self.name}: {self.probes} probes, {self.single} single")
        return result

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens of the text by the wrapped tokenizer."""
        return self._inner.count_tokens(text)
