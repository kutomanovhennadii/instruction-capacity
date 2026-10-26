"""The public part is English only: no Cyrillic characters in its text files."""

import re
from pathlib import Path

_PUBLIC_ROOT = Path(__file__).resolve().parents[1]
_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt"}
_SKIPPED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_CYRILLIC = re.compile("[\\u0400-\\u04ff]")


def test_public_text_files_contain_no_cyrillic() -> None:
    offenders = []
    for path in sorted(_PUBLIC_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in _SUFFIXES:
            continue
        if _SKIPPED_DIRS.intersection(path.parts):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _CYRILLIC.search(line):
                offenders.append(f"{path.relative_to(_PUBLIC_ROOT)}:{number}")
    assert not offenders, "Cyrillic text in the public part:\n" + "\n".join(offenders[:50])
