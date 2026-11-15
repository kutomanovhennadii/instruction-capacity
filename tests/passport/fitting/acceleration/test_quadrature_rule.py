"""The quadrature rule: the Gauss-Hermite nodes and weights, their adaptive form, its key."""

from __future__ import annotations

import numpy as np
import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.fitting.acceleration.quadrature_rule import (
    QUADRATURE_RULE_INVALID,
    build_quadrature_rule,
)


def test_the_rule_holds_the_gauss_hermite_nodes_and_their_adaptive_weights() -> None:
    rule = build_quadrature_rule(15)
    nodes, weights = np.polynomial.hermite.hermgauss(15)

    assert rule.size == 15
    assert np.array_equal(rule.nodes, nodes)
    assert np.array_equal(rule.weights, weights)
    assert np.array_equal(rule.log_weights, np.log(weights) + nodes * nodes)


def test_the_rule_integrates_a_polynomial_exactly() -> None:
    rule = build_quadrature_rule(5)
    # The integral of e^(-z^2) z^4 is 3 sqrt(pi) / 4; five nodes are exact up to degree 9.
    assert float(rule.weights @ rule.nodes**4) == pytest.approx(0.75 * np.sqrt(np.pi), rel=1e-12)


def test_a_rule_without_nodes_is_a_contract_violation() -> None:
    with pytest.raises(ContractViolationError) as caught:
        build_quadrature_rule(0)
    assert caught.value.error_code == QUADRATURE_RULE_INVALID
