import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_point.py"


def _main():
    spec = importlib.util.spec_from_file_location("measure_point", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_measure_point_appends_one_record_with_the_mock_model(tmp_path: Path) -> None:
    out = tmp_path / "points.jsonl"
    code = _main()(
        [
            "--provider", "mock", "--model", "mock-model", "--max-tokens", "4096",
            "--D", "16", "--n", "5", "--m", "8",
            "--seed-rules", "11", "--seed-text", "12", "--out", str(out),
        ]
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert code == 0
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "ok"
