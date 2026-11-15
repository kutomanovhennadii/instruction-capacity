"""The tokenizer read from a tokenizer.json file."""

import pytest
from tokenizers import Tokenizer as TokenizerFile
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from iec.infrastructure.exceptions import ProviderError
from iec.model_client.tokenizer_adapters import hub_tokenizer
from iec.model_client.tokenizer_adapters.hub_tokenizer import HubTokenizer, build_hub_tokenizer


def test_a_hub_tokenizer_counts_without_special_tokens() -> None:
    raw = TokenizerFile(WordLevel({"[UNK]": 0, "lake": 1, "stone": 2}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = HubTokenizer(name="hfjson/test", _tokenizer=raw)
    assert tokenizer.is_single_token(" lake") is True
    assert tokenizer.is_single_token(" lake stone") is False
    assert tokenizer.count_tokens("lake stone lake") == 3


def test_a_hub_tokenizer_that_cannot_load_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Missing:
        @staticmethod
        def from_pretrained(repo_id: str) -> None:
            raise OSError(f"{repo_id} not found")

    monkeypatch.setattr(hub_tokenizer, "TokenizerFile", _Missing)
    with pytest.raises(ProviderError) as failure:
        build_hub_tokenizer("nobody/nothing")
    assert failure.value.error_code == "HUB_TOKENIZER_LOAD_FAILED"
