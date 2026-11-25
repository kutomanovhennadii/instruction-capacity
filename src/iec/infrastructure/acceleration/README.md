# acceleration — compiled loops

The hot loops of the bench that numpy cannot vectorize are written as Python functions of numbers and numpy arrays and
decorated with `compiled`. When numba is installed, such a function is compiled into machine code at its first call, and the
code is kept in `__pycache__` for the next processes. When numba is not installed, or the environment variable
`IEC_COMPILED_LOOPS` is `off`, the function stays plain Python: the result is the same to the last number, only slower, and
one warning advises to install numba.

## What is here

| Module | Role |
|---|---|
| `compiled_loops.py` | `compiled`: the decorator that compiles a loop by numba or leaves it plain; `plain`: the plain Python function behind a compiled one; `COMPILED`: whether the loops are compiled |

## Key signatures

```python
compiled(function) -> function
plain(function) -> function
COMPILED: bool
```

## Boundaries

A compiled function holds no Python objects: only numpy arrays of `int64`, `uint64` and `bool`, numbers and calls of other
compiled functions; dictionaries, strings and lists are prepared before the call. numba is an optional dependency of the
package (`pip install "iec[numba]"`), and the exact versions of `requirements.txt` include it. The folder knows nothing of
the loops it compiles: they live in the modules that use them, the anchors and the gaps of the alignment.
