"""The verify command on the published vocabulary: it matches its reference version."""

import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_vocab.py"
_PUBLISHED_HASH = "sha256:5a91097540bec6584f7ffc94bffb4e5a6a2a9087f28828fa44d7214ccfa75906"


def _main() -> Callable[[Sequence[str] | None], int]:
    spec = importlib.util.spec_from_file_location("verify_vocab", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_published_vocabulary_matches_its_reference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _main()(["--skip-provenance"])

    out = capsys.readouterr().out
    assert code == 0
    assert _PUBLISHED_HASH in out
    assert "RESULT: PASSED" in out
