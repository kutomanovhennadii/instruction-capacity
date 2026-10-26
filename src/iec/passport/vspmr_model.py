"""The public passport model VS-PMR ``vspmr_glmm_v1``: its parameters, the design row of a
trial, the rule price, its fit.

The logit of a correct replacement is ``η = VS(v, s) + P(p) + M(m) + R(r) + b``. ``b`` is the
shift of a point, shared by all its trials and integrated out of the likelihood by the fit
(:mod:`iec.passport.glmm_fit`); the design row of a trial does not carry it. Text length and
word position are tied - doubling the text costs much at the start of the answer and almost
nothing deep in it - so they form one grid of free cells rather than a sum of two axes.
Density ``p = m/D``, the number of rules and the rule number are free bins with an implicit
zero in their reference bin. Density bins are numbered from the sparse end (1) to ``p = 1``
(the last).

**There is no common level ``alpha``.** The grid cells carry the level, all of them free, so
the exclusion rule may remove any cell and the model stays defined: short texts are almost
error-free, and the cells of the shortest texts are almost always degenerate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.bin_layout import BinLayout, bin_width
from iec.passport.model_fitting import FitResult, fit_vspmr_glmm
from iec.passport.passport_model import Trial
from iec.passport.vpm_cube import density_level, density_p
from iec.prompt_generator.text_generator import WORDS_PER_LINE

if TYPE_CHECKING:
    from iec.passport.design import Design
    from iec.passport.measured_points import MeasuredPoint

#: Name of the public passport model: VS-PMR with the shift of a point inside the likelihood,
#: ``η = VS(v, s) + P(p) + M(m) + R(r) + b``, ``b ~ N(0, σ²)`` (:mod:`iec.passport.glmm_fit`).
VSPMR_GLMM_V1 = "vspmr_glmm_v1"

_INVALID_CODE = "PREDICTOR_PROFILE_INVALID"


def doub_bin(value: int) -> int:
    """Return the doubling bin of a value counted from one: ``{1}, {2}, {3,4}, {5..8}, ...``.

    The top of bin ``k`` is ``2^(k−1)``. The word-position bin of the grid is the doubling bin
    of the answer line, ``doub(⌈s/WORDS_PER_LINE⌉)``.
    """
    value = max(1, int(value))
    return 1 if value <= 1 else int(math.floor(math.log2(value - 1))) + 2


def _require_positive(**values: int) -> None:
    """Require the conditions of a trial to be positive."""
    for name, value in values.items():
        if value < 1:
            raise ContractViolationError(
                f"trial feature input '{name}' must be >= 1, got {value}",
                error_code=_INVALID_CODE,
            )


def delta_axis(
    prefix: str,
    bin_index: int,
    theta: Mapping[str, float],
    bounds: tuple[tuple[int, int], ...],
) -> float:
    """Return the increment of a free axis per ONE step of its bin.

        Δ_k = (c_k − c_{k−1}) / |bin_k|

    A free profile enters a cumulative price by its increment, not its level: the level is a
    step, flat inside a bin. Dividing the jump by the bin width gives the increment per step;
    the increments of a bin sum to the jump. The first bin has no increment, and the
    coefficient before bin 2 is the implicit zero.
    """
    if bin_index <= 1:
        return 0.0
    current = float(theta[f"{prefix}{bin_index}"])
    previous = 0.0 if bin_index == 2 else float(theta[f"{prefix}{bin_index - 1}"])
    return (current - previous) / bin_width(bounds, bin_index)


def delta_axis_gradient(
    prefix: str,
    bin_index: int,
    bounds: tuple[tuple[int, int], ...],
    names: Sequence[str],
) -> list[float]:
    """Return ``∇Δ_k`` in the order of ``names``: two neighbouring bins with ``±1/|bin_k|``."""
    gradient = [0.0] * len(names)
    if bin_index <= 1:
        return gradient
    width = bin_width(bounds, bin_index)
    gradient[names.index(f"{prefix}{bin_index}")] = 1.0 / width
    if bin_index > 2:
        gradient[names.index(f"{prefix}{bin_index - 1}")] = -1.0 / width
    return gradient


@lru_cache(maxsize=64)
def vs_grid_tops(layout: BinLayout) -> tuple[int, ...]:
    """Return, per length bin, the last word-position bin of the answer.

    An answer word does not stand past the end of the text, so the position bin goes up to
    the bin of the last line of the longest text of the length bin:
    ``doub(⌈top of the bin / WORDS_PER_LINE⌉)``.
    """
    return tuple(doub_bin(-(-upper // WORDS_PER_LINE)) for _, upper in layout.volume_bounds)


@lru_cache(maxsize=64)
def vs_grid_cells(layout: BinLayout) -> tuple[tuple[int, int], ...]:
    """Return the cells of the length × position grid in order: (length bin, position bin)."""
    return tuple(
        (volume, position)
        for volume, top in enumerate(vs_grid_tops(layout), start=1)
        for position in range(1, top + 1)
    )


@lru_cache(maxsize=64)
def _vs_grid_index(layout: BinLayout) -> dict[tuple[int, int], int]:
    """Return the column of every grid cell - an index for building a row in O(1)."""
    return {cell: position for position, cell in enumerate(vs_grid_cells(layout))}


def vspmr_density_bin(layout: BinLayout, D: int, m: int) -> int:  # noqa: N803
    """Return the VS-PMR density bin: from the sparse end (1) to ``p = 1`` (the last).

    The number is the run-cube density level (``0`` at ``p = 1``) shifted so that the
    sparsest level of the layout is ``1``: ``level + P``. A node sparser than the layout goes
    to the first bin - the same edge pooling as on the other axes.
    """
    return max(1, density_level(D, m) + layout.P)


def vspmr_density_reference(layout: BinLayout) -> int:
    """Return the VS-PMR number of the reference density bin.

    The layout keeps the reference bin ``k`` counted from ``p = 1``; its level is ``−(k−1)``,
    its VS-PMR number ``−(k−1) + P``.
    """
    return layout.P - layout.density_reference + 1


def vspmr_density_p(layout: BinLayout, number: int) -> tuple[float, float]:
    """Return the density of a VS-PMR bin ``[p_lo, p_hi]``; the last bin is ``[1, 1]``."""
    return density_p(number - layout.P)


def _load_axes_price_gradient(
    rule_index: int, names: Sequence[str], layout: BinLayout
) -> tuple[float, ...]:
    """Return ``∇a(r)`` of the rule price ``a(r) = −ΔM(r) − R(r)`` in the order of ``names``."""
    gradient = [0.0] * len(names)
    if rule_index == 1:
        return tuple(gradient)
    bin_index = layout.row_bin(rule_index)
    gradient[list(names).index(f"r_bin_{bin_index}")] = -1.0
    for slot, value in enumerate(
        delta_axis_gradient("m_bin_", bin_index, layout.row_bounds, names)
    ):
        gradient[slot] -= value
    return tuple(gradient)


def rule_price(rule_index: int, theta: Mapping[str, float], layout: BinLayout) -> float:
    """Return the price of a rule: the increment of the table-size axis plus its position.

    A rule loads the model twice: the table grows by a line (axis ``M``) and the line is
    executed (axis ``R``)::

        a(r) = −ΔM(r) − R(r),   ΔM(r) = (M_k − M_{k−1}) / |bin_k|

    The first rule has no price: the first bins of both axes are ``{1}`` with implicit
    zero coefficients.
    """
    if rule_index == 1:
        return 0.0
    bin_index = layout.row_bin(rule_index)
    return -delta_axis("m_bin_", bin_index, theta, layout.row_bounds) - float(
        theta[f"r_bin_{bin_index}"]
    )


@dataclass(frozen=True, slots=True)
class VspmrGlmmModel:
    """The public passport model ``vspmr_glmm_v1``: the names of the coefficients, the design
    row of a trial and the fit with the shift of a point inside the likelihood; it implements
    :class:`~iec.passport.passport_model.PassportModel`."""

    profile_id: str = VSPMR_GLMM_V1

    def parameter_names(self, layout: BinLayout) -> tuple[str, ...]:
        """Return the names: grid cells, then the P, M, R bins without the reference bins."""
        reference = vspmr_density_reference(layout)
        return (
            *(f"vs_{volume}_{position}" for volume, position in vs_grid_cells(layout)),
            *(f"p_bin_{number}" for number in range(1, layout.P + 1) if number != reference),
            *(f"m_bin_{index}" for index in range(2, layout.R + 1)),
            *(f"r_bin_{index}" for index in range(2, layout.R + 1)),
        )

    def trial_vector(
        self,
        trial: Trial,
        layout: BinLayout,
        reference_D: int,  # noqa: N803 - protocol axis name
        reference_n: int,
    ) -> tuple[float, ...]:
        """Return the design row: one active grid cell and one column of ``P``, ``M``, ``R``.

        A position past the last line of the length bin goes to the last cell of the bin -
        the same edge pooling as on the other axes.
        """
        del reference_D, reference_n  # the zeros of the blocks are the reference bins
        _require_positive(D=trial.D, n=trial.n, m=trial.m, rule_index=trial.rule_index)
        volume = layout.volume_bin(trial.D * trial.n)
        position = min(
            doub_bin(math.ceil(trial.word_position / WORDS_PER_LINE)),
            vs_grid_tops(layout)[volume - 1],
        )
        cells = len(vs_grid_cells(layout))
        grid = [0.0] * cells
        grid[_vs_grid_index(layout)[(volume, position)]] = 1.0
        k_p = vspmr_density_bin(layout, trial.D, trial.m)
        reference = vspmr_density_reference(layout)
        k_m = layout.row_bin(trial.m)
        k_r = layout.row_bin(trial.rule_index)
        return (
            *grid,
            *(
                1.0 if number == k_p else 0.0
                for number in range(1, layout.P + 1)
                if number != reference
            ),
            *(1.0 if index == k_m else 0.0 for index in range(2, layout.R + 1)),
            *(1.0 if index == k_r else 0.0 for index in range(2, layout.R + 1)),
        )

    def fit(
        self,
        design: Design,
        points: Sequence[MeasuredPoint],
        run_weights: Mapping[str, float] | None,
    ) -> FitResult:
        """Fit the design with the shift of a point inside the likelihood (GLMM).

        ``points`` are not read: the variance of the shift comes from the fit itself.

        Raises:
            PassportError: ``PASSPORT_INPUT_MISSING`` - the design has no trials;
                ``FIT_NOT_CONVERGED`` - a fit did not converge.
            ContractViolationError: ``FIT_RUN_WEIGHTS_MISSING``, ``MODEL_FITTING_NON_FINITE``.
        """
        del points  # the variance of the shift is estimated inside the fit
        return fit_vspmr_glmm(design, run_weights)
