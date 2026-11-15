"""Mirror tests of the one rule of loading a tokenizer library."""

from collections.abc import Callable

import pytest

from iec.infrastructure.exceptions import ProviderError
from iec.model_client.tokenizer_adapters.library_loading import LibraryLoading

_LOADING = LibraryLoading(
    errors=(OSError,), error_code="TOY_LOAD_FAILED", message="failed to load toy '{name}'"
)


def _failing(error: Exception) -> Callable[[], object]:
    """Return a loader that raises the given error."""

    def load() -> object:
        """Raise the error instead of loading."""
        raise error

    return load


def test_the_loaded_object_is_returned() -> None:
    assert _LOADING.load(lambda: "tokenizer", "toy-model") == "tokenizer"


def test_an_expected_error_becomes_a_provider_error_with_its_cause() -> None:
    cause = OSError("no such repository")

    with pytest.raises(ProviderError) as caught:
        _LOADING.load(_failing(cause), "toy-model")

    assert caught.value.error_code == "TOY_LOAD_FAILED"
    assert str(caught.value) == "failed to load toy 'toy-model'"
    assert caught.value.__cause__ is cause


def test_an_unexpected_error_is_a_defect_and_passes_unchanged() -> None:
    cause = RuntimeError("bug in the loader")

    with pytest.raises(RuntimeError) as caught:
        _LOADING.load(_failing(cause), "toy-model")

    assert caught.value is cause
