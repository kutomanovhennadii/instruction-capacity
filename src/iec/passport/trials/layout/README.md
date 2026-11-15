# layout — the bins of the passport axes and the run cube

The axes of the passport are cut into bins by doubling the upper bound: `{1}, {2}, {3-4}, {5-8}, ...` from one, or
`{1-20}, {21-40}, {41-80}, ...` from the 20 words of the start step. Early values are told apart one by one, late values are
pooled. The run cube V × P × M - text length, density level and number of rules - gives the passport its bins and weighs the
points so that every cell weighs the same.

## What is here

| Module | Role |
|---|---|
| `bin_layout.py` | `BinLayout`, the doubling rule `doubling_bounds()` and `locate_bin()`, the start step and `REFERENCE_TOTAL_WORDS = 20` |
| `vpm_cube.py` | `VpmCube`: the cell of a node in the run cube V × P × M; `build_layout()`, `cell_balanced_weights()`, `point_key()`, `density_level()` |
| `axis_bins.py` | the bins as the fits use them: `doub_bin()`, the length × word-number grid `vs_grid_tops()` and `vs_grid_cells()`, density bins from the sparse end |

## Key signatures

```python
build_layout(points: Sequence[PointRecord]) -> BinLayout
cell_balanced_weights(ok_points: Sequence[PointRecord], cube: VpmCube) -> tuple[dict[str, float], dict[str, Any]] | None
vs_grid_cells(layout: BinLayout) -> tuple[tuple[int, int], ...]
```

## Boundaries

The folder cuts bins and weighs points; it counts no trials. The grid of a layout is a few dozen cells and is computed at
every call: the folder keeps no cache of the process.
