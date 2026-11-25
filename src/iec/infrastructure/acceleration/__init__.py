"""Compiled loops of the bench: numba compiles them when it is installed, plain Python otherwise."""

from iec.infrastructure.acceleration.compiled_loops import COMPILED, compiled, plain

__all__ = ["COMPILED", "compiled", "plain"]
