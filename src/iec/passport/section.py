"""The VS-PMR section of the passport: centred bins with their errors, excluded bins.

Four blocks - the length × position grid, the density ``p = m/D`` (from the sparse end to
``p = 1``; a bin carries its ``p`` interval), the number of rules, the rule number. Each block
is **centred**: the mean of its measured bins (a reference bin is an implicit zero) is zero,
so the picture does not depend on the choice of reference bins. SE is by the full covariance
of the fit: a centred value is the difference of a coefficient and the block mean, and both
carry their covariance.

Bins removed by the exclusion rule (all replacements right or all wrong) stay in their block
with status ``excluded`` and no value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from iec.passport.bin_layout import BinLayout
from iec.passport.passport_model import PassportModel
from iec.passport.vspmr_model import (
    vs_grid_cells,
    vspmr_density_reference,
    vspmr_density_p,
)
from iec.prompt_generator.text_generator import WORDS_PER_LINE

#: Status of a bin: estimated by the fit.
STATUS_MEASURED = "measured"
#: Status of a bin: removed by the exclusion rule - there is no estimate.
STATUS_EXCLUDED = "excluded"

_CENTRING_NOTE = (
    "every block is centred: the mean of the measured bins of the block (a reference bin is "
    "an implicit zero) is zero; SE by the full covariance of the fit"
)


def expected_sign(name: str) -> str | None:
    """Return the declared sign hypothesis of a coefficient: ``"+"``, ``"−"`` or ``None``.

    The level ``alpha`` is positive; the table-size bins ``m_bin_*`` (and the load terms
    ``g_v``, ``out_bin_*`` of earlier models) are negative - an extra rule makes the task
    heavier. Free profiles have no hypothesis: their direction is what is measured.
    """
    if name == "alpha":
        return "+"
    if name == "g_v" or name.startswith(("m_bin_", "out_bin_")):
        return "−"
    return None


@dataclass(frozen=True, slots=True)
class PassportSection:
    """The section of one model in the passport, grouped by its reader.

    ``capacities`` - capacities by quality level (empty for VS-PMR: its capacities are the
    public quantities of the approximation block); ``bernoulli`` - the fitted bins;
    ``experiment`` - what tells whether the numbers can be trusted.
    """

    model_id: str
    capacities: dict[str, Any]
    bernoulli: dict[str, Any]
    experiment: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return the section as a dictionary of ``passport_data.json``."""
        return {
            "model_id": self.model_id,
            "capacities": self.capacities,
            "bernoulli": self.bernoulli,
            "experiment": self.experiment,
        }


def vs_grid_passport_section(
    profile: PassportModel,
    theta: Mapping[str, float],
    covariance: Sequence[Sequence[float]],
    layout: BinLayout,
    *,
    model_id: str,
    fit_status: Mapping[str, Any],
    covariance_provenance: str,
) -> PassportSection:
    """Build the VS-PMR section: four centred blocks, the excluded bins, the fit status."""
    names = list(theta)
    index = {name: position for position, name in enumerate(names)}
    values = np.array([float(theta[name]) for name in names], dtype=float)
    matrix = np.array([list(row) for row in covariance], dtype=float)
    if matrix.shape != (len(names), len(names)):
        matrix = np.zeros((len(names), len(names)))
    symmetric = (matrix + matrix.T) / 2.0
    excluded = [name for name in profile.parameter_names(layout) if name not in index]

    def block(entries: Sequence[tuple[dict[str, Any], str | None]]) -> list[dict[str, Any]]:
        """Centre a block; ``None`` instead of a name is a reference bin, an implicit zero."""
        measured = [name for _, name in entries if name is None or name in index]
        mean = np.zeros(len(names))
        for name in measured:
            if name is not None:
                mean[index[name]] += 1.0 / len(measured)
        rows: list[dict[str, Any]] = []
        for fields, name in entries:
            if name is not None and name not in index:
                rows.append({**fields, "value": None, "se": None, "status": STATUS_EXCLUDED})
                continue
            gradient = -mean.copy()
            if name is not None:
                gradient[index[name]] += 1.0
            variance = float(gradient @ symmetric @ gradient)
            rows.append(
                {
                    **fields,
                    "value": float(gradient @ values),
                    "se": math.sqrt(max(variance, 0.0)),
                    "status": STATUS_MEASURED,
                }
            )
        return rows

    grid = block(
        [
            (
                {
                    "v_bin": volume,
                    "s_bin": position,
                    "words": list(layout.volume_bounds[volume - 1]),
                    "positions": _positions(position),
                },
                f"vs_{volume}_{position}",
            )
            for volume, position in vs_grid_cells(layout)
        ]
    )
    # Density - from the sparse end to p = 1: the bin number grows with p = m/D.
    reference = vspmr_density_reference(layout)
    density = block(
        [
            (
                {"bin": number, "p": list(vspmr_density_p(layout, number))},
                None if number == reference else f"p_bin_{number}",
            )
            for number in range(1, layout.P + 1)
        ]
    )

    def rows_axis(prefix: str) -> list[dict[str, Any]]:
        return block(
            [
                (
                    {"bin": number, "bounds": list(bounds)},
                    None if number == 1 else f"{prefix}{number}",
                )
                for number, bounds in enumerate(layout.row_bounds, start=1)
            ]
        )

    return PassportSection(
        model_id=model_id,
        capacities={},
        bernoulli={
            "centring": _CENTRING_NOTE,
            "vs_grid": grid,
            "density_profile": density,
            "table_size_profile": rows_axis("m_bin_"),
            "rule_position_profile": rows_axis("r_bin_"),
            "excluded": excluded,
        },
        experiment={
            "fit_status": dict(fit_status),
            "covariance_provenance": covariance_provenance,
            "parameters": len(names),
            "excluded_parameters": len(excluded),
        },
    )


def _positions(position_bin: int) -> list[int]:
    """Return the answer words of a position bin: bin 1 is the first line, bin ``k`` - lines
    ``2^(k−2)+1 … 2^(k−1)``."""
    if position_bin == 1:
        return [1, WORDS_PER_LINE]
    first = 2 ** (position_bin - 2) * WORDS_PER_LINE + 1
    return [first, 2 ** (position_bin - 1) * WORDS_PER_LINE]
