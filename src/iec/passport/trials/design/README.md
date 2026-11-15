# design — the folded design matrices of the quantities

A row of a folded matrix is a point and a cell of the length × word-number grid, for H also a bin of the rule number; the
columns are the cells `vs_V_S` and the bins `p_bin_k`, `m_bin_k` and `r_bin_k` without the reference bins. The conditions of
a request are common to all trials of a row, so the folding is exact.

## What is here

| Module | Role |
|---|---|
| `folded_design.py` | `build_design()`: the folding, one for all quantities; `VariableDesign`, `GridCell`, `AxisTable`, `AxisBin` |
| `quantity_designs.py` | the strategy of a quantity: its trials of a point and its axes; the registry `QUANTITY_DESIGNS` and `VARIABLE_AXES` |
| `design_axes.py` | the strategy of an axis: the bin, the bounds, the reference bin, the abscissa and the column prefix; `COLUMN_PREFIX` |

## Key signatures

```python
build_design(points: Sequence[PointTrials], variable: str, layout: BinLayout,
             weights: Mapping[str, float]) -> VariableDesign
QUANTITY_DESIGNS["H"].cells(point: PointTrials) -> list[DesignCell]
```

## The strategies

| Quantity | Trials of a point | Axes besides the grid |
|---|---|---|
| `P_sync` | the places | density `P`, number of rules `M` |
| `H` | the rule words on their places | `P`, `M` and the rule number `R` |
| `F` | the background words on their places | `P`, `M` |

Density and the number of rules are axes of a request, constant inside a point; the rule number is an axis of a cell. A new
quantity is a new class in `quantity_designs.py` and a line of the registry; the folding does not change.

## Boundaries

The folder builds matrices; it fits nothing. An unknown quantity is a `ContractViolationError(VARIABLE_UNKNOWN)`, a point with
trials and without a weight is a `ContractViolationError(POINT_WEIGHT_MISSING)`.
