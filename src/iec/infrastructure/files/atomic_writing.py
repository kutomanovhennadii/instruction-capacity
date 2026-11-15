"""Reliable writing of a text file: a temporary file next to it and a rename onto its place.

A process broken off in the middle of a write must leave the old file or the new one, never a
half-written one (specification §7.7, item 7): a damaged vocabulary, passport or snapshot would
stop the next run. The text goes into a temporary file in the folder of the target, and
``os.replace`` puts it onto the place of the target in one step: a rename within one folder
replaces the target atomically on every supported system.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Ending of the name of the temporary file that is then renamed onto the target.
TEMPORARY_SUFFIX = ".tmp"


def write_text_atomically(path: Path, text: str) -> None:
    """Write a text file reliably: a temporary file next to it, then a rename onto its place.

    The text is written in UTF-8 and in text mode, as ``Path.write_text`` writes it, so the line
    ends follow the platform exactly as they did before the writing became reliable.

    Args:
        path: The target file; its folder is created when missing.
        text: The whole content of the file.

    Raises:
        OSError: the folder or the temporary file cannot be written, or the rename fails.
        UnicodeEncodeError: the text cannot be encoded in UTF-8.

    Side effects:
        Creates or replaces ``path``; after a successful write no temporary file stays. A write
        that fails leaves ``path`` as it was; the temporary file it may leave behind is replaced
        by the next write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + TEMPORARY_SUFFIX)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
