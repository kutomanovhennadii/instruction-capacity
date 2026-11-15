"""The Gauss-Hermite rule of an integral over the shift of a request, built once.

The marginal likelihood of a request integrates its shift out (``glmm_fit``), and the threshold
of a level averages a success over the shifts. Both use the Gauss-Hermite rule
"∫ e^(−z²)·f(z) dz ≈ Σ_k w_k·f(z_k)". Its nodes and weights are a structure for speed
(specification §2.11): a rule is built once by ``build_quadrature_rule`` and handed to every fit
ready. Its key is ``size``, the number of nodes: another number of nodes needs a new rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError

#: Error code of a rule asked with fewer than one node.
QUADRATURE_RULE_INVALID = "QUADRATURE_RULE_INVALID"


@dataclass(frozen=True, slots=True, eq=False)
class QuadratureRule:
    """The nodes and weights of a Gauss-Hermite rule.

    ``nodes`` - the nodes ``z_k``; ``weights`` - their weights ``w_k``; ``log_weights`` -
    ``ln w_k + z_k²``, the weights of the adaptive form: the change of variables
    ``b = mode + √2·scale·z`` turns the integral over the shift into the Gauss-Hermite form, and
    the factor ``e^(z²)`` restores the integrand.
    """

    nodes: np.ndarray
    weights: np.ndarray
    log_weights: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of nodes, the key of the rule."""
        return len(self.nodes)


def build_quadrature_rule(size: int) -> QuadratureRule:
    """Build the Gauss-Hermite rule of ``size`` nodes.

    Args:
        size: The number of nodes.

    Returns:
        ``QuadratureRule`` with the nodes, their weights and the weights of the adaptive form.

    Raises:
        ContractViolationError: ``QUADRATURE_RULE_INVALID`` - ``size`` is below one.
    """
    if size < 1:
        raise ContractViolationError(
            f"a quadrature rule needs at least one node, got {size}",
            error_code=QUADRATURE_RULE_INVALID,
        )
    nodes, weights = np.polynomial.hermite.hermgauss(size)
    return QuadratureRule(
        nodes=nodes, weights=weights, log_weights=np.log(weights) + nodes * nodes
    )
