"""Mirror tests of the tokenizer builder and the registry of tokenizer factories."""

import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.model_client import build_default_tokenizer_builder
from iec.model_client.tokenizer_adapters import gemini_count_tokenizer
from iec.model_client.tokenizer_adapters import hf_tokenizer
from iec.model_client.tokenizer_adapters import hub_tokenizer
from iec.model_client.tokenizer_adapters import tiktoken_tokenizer


def test_default_builder_knows_the_four_providers_by_the_names_of_the_adapters() -> None:
    assert build_default_tokenizer_builder().known_providers() == frozenset(
        {
            tiktoken_tokenizer.TIKTOKEN_PROVIDER,
            hf_tokenizer.HF_PROVIDER,
            hub_tokenizer.HUB_PROVIDER,
            gemini_count_tokenizer.GEMINI_COUNT_PROVIDER,
        }
    )


def test_unknown_provider_is_a_configuration_error() -> None:
    with pytest.raises(ConfigError) as caught:
        build_default_tokenizer_builder().build("missing", "model")

    assert caught.value.error_code == "UNKNOWN_TOKENIZER_PROVIDER"
    assert "missing" in str(caught.value)
    assert "tiktoken" in str(caught.value)


def test_tiktoken_is_built_through_the_registry() -> None:
    tokenizer = build_default_tokenizer_builder().build("tiktoken", "gpt-4o")

    assert isinstance(tokenizer, tiktoken_tokenizer.TiktokenTokenizer)
    assert tokenizer.name == "tiktoken/gpt-4o"


def test_gemini_count_without_a_key_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gemini_count_tokenizer.GEMINI_API_KEY_ENV, raising=False)

    with pytest.raises(ConfigError) as caught:
        build_default_tokenizer_builder().build("gemini_count", "gemini-3.5-flash-lite")

    assert caught.value.error_code == "GEMINI_API_KEY_MISSING"


def test_gemini_count_with_a_key_is_built_through_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(gemini_count_tokenizer.GEMINI_API_KEY_ENV, "test-key")

    tokenizer = build_default_tokenizer_builder().build("gemini_count", "gemini-3.5-flash-lite")

    assert isinstance(tokenizer, gemini_count_tokenizer.GeminiCountTokenizer)
    try:
        assert tokenizer.name == "gemini_count/gemini-3.5-flash-lite"
    finally:
        tokenizer.client.close()


def test_hub_tokenizer_is_built_through_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _File:
        @staticmethod
        def from_pretrained(repo_id: str) -> str:
            return f"file of {repo_id}"

    monkeypatch.setattr(hub_tokenizer, "TokenizerFile", _File)

    tokenizer = build_default_tokenizer_builder().build("hfjson", "org/repo")

    assert isinstance(tokenizer, hub_tokenizer.HubTokenizer)
    assert tokenizer.name == "hfjson/org/repo"


def test_hf_tokenizer_is_built_through_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Auto:
        @staticmethod
        def from_pretrained(model_name: str) -> str:
            return f"tokenizer of {model_name}"

    monkeypatch.setattr(hf_tokenizer, "AutoTokenizer", _Auto)

    tokenizer = build_default_tokenizer_builder().build("hf", "org/model")

    assert isinstance(tokenizer, hf_tokenizer.HFTokenizer)
    assert tokenizer.name == "hf/org/model"
