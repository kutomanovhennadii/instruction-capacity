# files — reliable writing of the files of the bench

A file of the bench is written so that a process broken off in the middle of a write damages nothing written before. A whole
file — a vocabulary, a passport, a document — is replaced at once: the old file or the new one stays, never a half-written
one. A journal — the records of the points of a measurement — is only appended to: a line reported written is on the disk,
and a line cut short by a broken-off write never damages its neighbours.

## What is here

| Module | Role |
|---|---|
| `atomic_writing.py` | `write_text_atomically`: the text goes into a temporary file next to the target, then a rename puts it onto its place |
| `line_appending.py` | `append_line`: the whole line in one call, after a complete line, synced to the disk |

## Key signatures

```python
write_text_atomically(path: Path, text: str) -> None
append_line(path: Path, line: str) -> None
```

## Boundaries

The folder chooses no place of writing: the caller gives the path. A journal is never rewritten: a line cut short by a
broken-off write is ended by the next append and stays in the journal, and its reader names it. The text goes in UTF-8 and
in text mode, so line ends follow the platform as `Path.write_text` writes them.
