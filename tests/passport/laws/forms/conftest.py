"""Shared fixtures of the tests of the forms of laws: the numeric derivatives of a law."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from iec.passport.laws.forms.law_form import LawForm

#: The relative step of the central differences.
STEP = 1e-6


@pytest.fixture
def numeric_jacobian() -> Callable[[LawForm, np.ndarray, np.ndarray], np.ndarray]:
    """Return the derivatives of a law by its parameters, taken by central differences."""

    def derivatives(form: LawForm, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        columns = []
        for column in range(len(params)):
            step = STEP * max(1.0, abs(float(params[column])))
            shift = np.zeros(len(params))
            shift[column] = step
            rise = form.value(params + shift, x) - form.value(params - shift, x)
            columns.append(rise / (2.0 * step))
        return np.column_stack(columns)

    return derivatives
