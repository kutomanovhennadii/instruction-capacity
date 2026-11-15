"""Mirror tests of the reliable writing of a text file."""

from pathlib import Path

import pytest

from iec.infrastructure.files import write_text_atomically


def test_the_text_is_written_and_missing_folders_are_created(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "folder" / "file.json"

    write_text_atomically(target, "lake\nstone\n")

    assert target.read_text(encoding="utf-8") == "lake\nstone\n"


def test_an_existing_file_is_replaced_and_no_temporary_file_stays(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    target.write_text("old", encoding="utf-8")

    write_text_atomically(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file.json"]


def test_a_failed_write_keeps_the_old_content(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    target.write_text("old", encoding="utf-8")

    # A lone surrogate cannot be encoded in UTF-8: the write fails before the rename.
    with pytest.raises(UnicodeEncodeError):
        write_text_atomically(target, "lake \ud800")

    assert target.read_text(encoding="utf-8") == "old"
