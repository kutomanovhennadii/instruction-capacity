"""Compiled loops: numba compiles them when it is installed, otherwise they run as plain Python.

The hot loops of the bench that numpy cannot vectorize - the checks of candidate lines of the anchors,
the longest increasing chain, the bit-parallel subsequences of many pairs of lines - are written as
Python functions of numbers and numpy arrays and decorated with :func:`compiled`. When numba is
installed, such a function is compiled into machine code at its first call, and the code is kept in
``__pycache__`` for the next processes. When numba is not installed, or the environment variable
``IEC_COMPILED_LOOPS`` is ``off``, the function stays plain Python: the result is the same to the last
number, only slower, and one warning advises to install numba. Production runs compiled; the plain
path serves a reader who verifies a measurement without numba: the speed of production is the main
task, the verification by a reader is secondary.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

from iec.infrastructure.logging import LogContext, get_logger

try:
    import numba
except ImportError:  # the environment of a reader without numba
    numba = None  # type: ignore[assignment]

_MODULE_NAME = "compiled_loops"
_Loop = TypeVar("_Loop", bound=Callable[..., Any])

#: Whether the loops are compiled: numba is installed, and ``IEC_COMPILED_LOOPS`` is not ``off``.
COMPILED: bool = numba is not None and os.environ.get("IEC_COMPILED_LOOPS", "on") != "off"

if not COMPILED:
    get_logger(LogContext(module_name=_MODULE_NAME)).warning(
        "the loops of the validator run without compilation and are slower; "
        "install numba for speed: pip install numba"
    )


def compiled(function: _Loop) -> _Loop:
    """Compile a loop by numba, or return it unchanged when the loops are not compiled.

    Args:
        function: A function of numbers and numpy arrays only, without Python objects inside.

    Returns:
        The compiled function, called as the original one, or the original function itself.
    """
    if not COMPILED:
        return function
    return numba.njit(cache=True)(function)  # type: ignore[return-value]


def plain(function: Callable[..., Any]) -> Callable[..., Any]:
    """Return the plain Python function behind a compiled loop, for the test of equal results.

    Args:
        function: A function decorated with :func:`compiled`.

    Returns:
        The original Python function; the function itself when it was not compiled.
    """
    return getattr(function, "py_func", function)
