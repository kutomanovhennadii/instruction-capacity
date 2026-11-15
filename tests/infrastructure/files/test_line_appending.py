"""Mirror tests of the reliable appending of a line to a journal."""

from pathlib import Path

import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.infrastructure.files import append_line


def test_lines_are_appended_in_order_and_the_file_is_created(tmp_path: Path) -> None:
    journal = tmp_path / "points.jsonl"

    append_line(journal, "lake")
    append_line(journal, "stone")

    assert journal.read_text(encoding="utf-8").split("\n") == ["lake", "stone", ""]


def test_a_cut_line_is_ended_before_the_next_line(tmp_path: Path) -> None:
    journal = tmp_path / "points.jsonl"
    journal.write_bytes(b'lake\n{"cut')

    append_line(journal, "stone")

    assert journal.read_text(encoding="utf-8").split("\n") == ["lake", '{"cut', "stone", ""]


def test_a_line_with_a_line_end_is_refused(tmp_path: Path) -> None:
    journal = tmp_path / "points.jsonl"

    with pytest.raises(ContractViolationError) as caught:
        append_line(journal, "lake\nstone")

    assert caught.value.error_code == "JOURNAL_LINE_INVALID"
    assert not journal.exists()
