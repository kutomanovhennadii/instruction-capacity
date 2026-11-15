"""The one rule of loading a tokenizer library: its expected failures become ``ProviderError``.

A tokenizer adapter loads the tokenizer of its library - an encoding of tiktoken, a Hugging Face
tokenizer, a ``tokenizer.json`` file - and every library fails in its own exceptions. The adapter
describes them in a ``LibraryLoading``: the exceptions its library raises on a missing model or an
unavailable network, the error code and the text of the failure. This class turns exactly those
exceptions into ``ProviderError`` and keeps the cause (specification §7.2); any other exception is
a defect and is not wrapped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from iec.infrastructure.exceptions import ProviderError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LibraryLoading:
    """How an adapter loads the tokenizer of its library: the failures it expects and their code.

    ``errors`` are the exceptions the library raises on a missing model or an unavailable network;
    ``error_code`` is the code of such a failure; ``message`` is its text with the place
    ``{name}`` for the name of the model.
    """

    errors: tuple[type[Exception], ...]
    error_code: str
    message: str

    def load(self, loader: Callable[[], T], name: str) -> T:
        """Call the loader of the library and turn the expected failures into ``ProviderError``.

        Args:
            loader: Function that loads the tokenizer object of the library.
            name: Name of the model, for the text of the failure.

        Returns:
            The tokenizer object of the library.

        Raises:
            ProviderError: ``error_code`` - the loader raised one of ``errors``; the cause is kept.
        """
        try:
            return loader()
        except self.errors as cause:
            raise ProviderError.from_cause(
                self.message.format(name=name), cause, error_code=self.error_code
            ) from cause
