"""Reliable appending of a line to a journal: the whole line at once, synced to the disk.

A journal of the bench, such as a file of the records of points, is only appended to and is never
rewritten (specification §7.7, item 7). Two rules make an append reliable. The whole line is handed
to the file in one call, and the file is synced to the disk before the function returns, so a line
reported written survives a broken-off process. A process broken off in the middle of a write may
still leave its line cut short, without its line end: the next append then ends the cut line first,
so its own line never sticks to the cut one. A broken-off write damages only its own line; the cut
line stays in the journal, and the reader of the journal names it.
"""

from __future__ import annotations

import os
from pathlib import Path

from iec.infrastructure.exceptions import ContractViolationError

#: The line end of a journal; text mode writes it as the line end of the platform.
LINE_END = "\n"
#: Error code of a line that would not stay one line of the journal.
JOURNAL_LINE_INVALID = "JOURNAL_LINE_INVALID"


def append_line(path: Path, line: str) -> None:
    """Append one line to a journal reliably: whole, after a complete line, synced to the disk.

    The line is written in UTF-8 and in text mode, as the journals were written before the
    appending became reliable, so the line ends follow the platform.

    Args:
        path: The journal; it is created when missing, and its folder must exist.
        line: The line without its line end.

    Raises:
        ContractViolationError: ``JOURNAL_LINE_INVALID`` - the line contains a line end ``\\n``
            or ``\\r`` and would become more than one line of the journal.
        OSError: the journal cannot be read or written.

    Side effects:
        Appends the line and its line end to ``path``. When the journal ends with a line cut
        short by a broken-off write, a line end goes first and ends that line. The file is synced
        to the disk before the function returns.
    """
    if LINE_END in line or "\r" in line:
        raise ContractViolationError(
            f"a journal line must not contain a line end: {line[:80]!r}",
            error_code=JOURNAL_LINE_INVALID,
        )
    prefix = LINE_END if _ends_with_cut_line(path) else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + line + LINE_END)
        handle.flush()
        os.fsync(handle.fileno())


def _ends_with_cut_line(path: Path) -> bool:
    """Tell whether the journal ends with a line cut short: bytes after its last line end.

    Args:
        path: The journal; a missing or empty journal has no cut line.

    Returns:
        ``True`` when the journal is not empty and its last byte is not ``\\n``.
    """
    if not path.exists():
        return False
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return False
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"
