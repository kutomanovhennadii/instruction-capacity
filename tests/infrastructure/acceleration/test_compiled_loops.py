"""Mirror tests of the compiled loops."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from iec.infrastructure.acceleration import COMPILED, compiled, plain

SRC = Path(__file__).resolve().parents[3] / "src"


def _total(values: np.ndarray) -> int:
    result = 0
    for value in values:
        result += value
    return result


def test_a_compiled_loop_gives_the_result_of_the_plain_function() -> None:
    values = np.arange(10, dtype=np.int64)
    loop = compiled(_total)
    assert loop(values) == _total(values) == 45
    assert plain(loop) is _total


def test_the_switch_turns_the_compilation_off() -> None:
    environment = {**os.environ, "IEC_COMPILED_LOOPS": "off", "PYTHONPATH": str(SRC)}
    printed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from iec.infrastructure.acceleration import COMPILED; print(COMPILED)",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    ).stdout.strip()
    assert printed == "False"


def test_the_loops_are_compiled_where_numba_is_installed() -> None:
    try:
        import numba  # noqa: F401
    except ImportError:
        assert not COMPILED
    else:
        assert COMPILED == (os.environ.get("IEC_COMPILED_LOOPS", "on") != "off")
