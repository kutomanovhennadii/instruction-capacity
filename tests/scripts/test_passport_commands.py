"""The passport commands fail clearly on a directory that is not a measurement."""

import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _main(name: str) -> Callable[[Sequence[str] | None], int]:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_build_passport_reports_a_missing_measurement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main("build_passport")([str(tmp_path)]) == 1
    assert "MEASUREMENT_FILES_MISSING" in capsys.readouterr().err


def test_verify_reports_a_missing_measurement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main("verify")([str(tmp_path)]) == 1
    assert "MEASUREMENT_FILES_MISSING" in capsys.readouterr().err
