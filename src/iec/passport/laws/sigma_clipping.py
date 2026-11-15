"""The outlier clipping of a law: the algorithm of astropy.

The passport takes the procedure of ``astropy.modeling.fitting.FittingWithOutlierRemoval`` with
the outlier function ``astropy.stats.sigma_clip`` and its defaults. The form is fitted through all
the bins of a block with the weights of its section. The residuals ``β − f(x)`` of the bins not yet
clipped give the centre - their median - and the scale - their standard deviation; a bin farther
than three scales from the centre is clipped, and the clipping of the residuals repeats, up to five
times, until no new bin goes out (``sigma_clip``). The form is then fitted again through the bins
left, the residuals of all the bins come from the new fit, and the procedure repeats until the
number of the clipped bins stops changing (``FittingWithOutlierRemoval``). A clipped bin stays
clipped, as in astropy. The scale is one for all the bins: a poorly determined bin gets no wider
tolerance. The clipping stops before a fit would keep no more bins than its parameters. A scale at
the rounding of the arithmetic - ``CLIP_FLOOR`` of the size of the values - means that the law goes
through its bins exactly: its residuals are rounding noise, not data, and nothing is clipped.

astropy is not a dependency of the passport: the procedure is these lines, and a module of the same
interface may call astropy instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from iec.passport.laws.forms.law_fit import LawFit, fit_law
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.point_weights import Weigh

#: The clipping threshold in scales of the residuals (``sigma`` of ``sigma_clip``).
CLIP_SIGMA = 3.0
#: The repetitions of the clipping of the residuals of one fit (``maxiters`` of ``sigma_clip``).
CLIP_REPEATS = 5
#: The most fits of one law; the procedure stops earlier when the number of the clipped bins does
#: not change.
CLIP_ROUNDS = 10
#: The scale of the residuals, relative to the size of the values, below which they are rounding
#: noise and nothing is clipped.
CLIP_FLOOR = 1e-9


@dataclass(frozen=True, slots=True, eq=False)
class ClippedFit:
    """The fit of a form through the bins left by the clipping.

    ``fit`` - the fit, its sensitivity over all the bins of the block with zero columns for the
    clipped ones; ``kept`` - the bins left, a mask over the bins of the block; ``rounds`` - the
    fits made.
    """

    fit: LawFit
    kept: np.ndarray
    rounds: int


def clip_residuals(
    residuals: np.ndarray,
    clipped: np.ndarray,
    sigma: float = CLIP_SIGMA,
    repeats: int = CLIP_REPEATS,
    floor: float = 0.0,
) -> np.ndarray:
    """Return the clipped bins after ``sigma_clip`` of the residuals: the old and the new ones.

    Args:
        residuals: The residuals of all the bins.
        clipped: The bins clipped before, a mask.
        sigma: The threshold in standard deviations around the median.
        repeats: The most repetitions.
        floor: The scale at or below which the residuals are rounding noise: nothing is clipped.
    """
    mask = np.asarray(clipped, dtype=bool).copy()
    for _ in range(repeats):
        alive = residuals[~mask]
        if alive.size == 0:
            break
        center = float(np.median(alive))
        scale = float(np.std(alive))
        if scale <= floor:
            break
        new = mask | (np.abs(residuals - center) > sigma * scale)
        if np.array_equal(new, mask):
            break
        mask = new
    return mask


def clipped_fit(
    form: LawForm,
    x: np.ndarray,
    values: np.ndarray,
    weigh: Weigh,
    *,
    starts: Sequence[np.ndarray] | None = None,
    rounds: int = CLIP_ROUNDS,
) -> ClippedFit | None:
    """Fit a form through the bins of a block, clipping the outliers.

    Args:
        form: The form of the law.
        x: The abscissae of the bins.
        values: Their coefficients.
        weigh: The weighting of the bins left.
        starts: The starts of the form; ``None`` - its own.
        rounds: The most fits.

    Returns:
        ``ClippedFit``, or ``None`` when the form does not converge through all the bins.
    """
    size = values.size
    fit = fit_law(form, x, values, weigh(np.arange(size)), starts=starts)
    if not usable(fit):
        return None
    clipped = np.zeros(size, dtype=bool)
    floor = CLIP_FLOOR * max(1.0, float(np.max(np.abs(values))))
    made = 1
    while made < rounds:
        new = clip_residuals(values - form.value(fit.params, x), clipped, floor=floor)
        if int(new.sum()) == int(clipped.sum()):
            break
        kept = np.flatnonzero(~new)
        if kept.size <= len(form.parameters):
            break
        found = fit_law(form, x[kept], values[kept], weigh(kept), starts=starts)
        if not usable(found):
            break
        fit, clipped, made = _spread(found, kept, size), new, made + 1
    return ClippedFit(fit, ~clipped, made)


def usable(fit: LawFit) -> bool:
    """Return whether a fit converged with finite estimates."""
    return fit.converged and bool(np.all(np.isfinite(fit.params)))


def _spread(fit: LawFit, kept: np.ndarray, size: int) -> LawFit:
    """Return the fit with its sensitivity over all the bins: zero columns for the clipped ones."""
    sensitivity = np.zeros((fit.sensitivity.shape[0], size))
    sensitivity[:, kept] = fit.sensitivity
    return replace(fit, sensitivity=sensitivity)
