"""Sink that appends every event to a JSONL file: one JSON line per event.

The file is opened for every event and closed at once: the record is on disk when the call
returns, the file can be read or moved while a run works, and no handle stays open. Threads of a
pool write under one lock, so lines never interleave. The file only grows (specification §7.7: a
journal is appended to).
"""

from __future__ import annotations

import threading
from pathlib import Path

from iec.infrastructure.telemetry.sinks.sink_contract import OneTimeWarning
from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent

_MODULE_NAME = "jsonl_file_sink"


class JsonlFileSink:
    """Appends events to one JSONL file; a disk error drops events with one warning."""

    def __init__(self, path: Path) -> None:
        """Remember the file and create its directory.

        Args:
            path: Path of the JSONL file; a relative path counts from the working directory.

        Side effects:
            Creates the missing directories of the file.
        """
        self._path = path
        self._lock = threading.Lock()
        self._warning = OneTimeWarning(_MODULE_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Path of the file the events are appended to."""
        return self._path

    def __call__(self, event: StepTelemetryEvent, /) -> None:
        """Append the event as one JSON line.

        Side effects:
            Appends a line to the file; on a write error logs one warning and drops the event.
        """
        line = event.as_json_line() + "\n"
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except OSError as error:
            self._warning.issue(
                f"telemetry sink cannot write to {self._path}: {error}; events are dropped"
            )
