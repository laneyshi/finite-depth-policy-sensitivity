"""Bounds in root.tex, commit 87ca041; all inputs use absolute spectral norms.

Constants must bound the actual environment/policy, not be fitted to error curves.
This module does not add damping or replace an inverse with a pseudoinverse.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Constants:
    reward: float
    policy_gradient: float
    policy_hessian: float
    transition_gradient_tv: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in vars(self).values()):
            raise ValueError('Uniform constants must be finite and nonnegative.')


def derivative_bounds(horizon, gamma, constants):
    """Return eta_H and eta_M for depths 1,...,horizon, including exact zero."""
    if not isinstance(horizon, int) or horizon < 1 or not 0 < gamma <= 1:
        raise ValueError('Require integer horizon >= 1 and 0 < gamma <= 1.')
    c = constants
    eh, em = np.zeros(horizon), np.zeros(horizon)
    for depth in range(1, horizon + 1):
        q = np.arange(horizon - depth, dtype=float)
        weights = gamma ** (depth + q) * (q + 1)
        eh[depth - 1] = c.reward * np.sum(weights * (
            c.policy_hessian + (q + 2 * depth) * c.policy_gradient ** 2))
        em[depth - 1] = (c.reward * c.policy_gradient * c.transition_gradient_tv
                         * np.sum(weights * (q + 2 * depth + 1)))
    return eh, em


def discounted_envelopes(horizon, gamma, constants):
    """The geometric envelopes apply only for gamma < 1."""
    if not 0 < gamma < 1:
        raise ValueError('Discounted envelopes require 0 < gamma < 1.')
    derivative_bounds(horizon, gamma, constants)
    c = constants
    f = c.reward * gamma ** np.arange(1, horizon + 1) / (1 - gamma) ** 2
    return (f * (c.policy_hessian + 2 * (horizon - 1) * c.policy_gradient ** 2),
            f * (2 * horizon - 1) * c.policy_gradient * c.transition_gradient_tv)


def sensitivity_bound(eta_h, eta_m, m, norm_s):
    """NaN marks depths outside the corollary's sufficient condition."""
    if not np.isfinite(m) or m <= 0 or not np.isfinite(norm_s) or norm_s < 0:
        raise ValueError('Require m > 0 and a finite nonnegative sensitivity norm.')
    eh, em = np.broadcast_arrays(np.asarray(eta_h, float), np.asarray(eta_m, float))
    if np.any(~np.isfinite(eh)) or np.any(~np.isfinite(em)) or np.any(eh < 0) or np.any(em < 0):
        raise ValueError('Derivative bounds must be finite and nonnegative.')
    result = np.full(eh.shape, np.nan)
    valid = eh < m
    np.divide(em + eh * norm_s, m - eh, out=result, where=valid)
    return result


def audit_reference(hessian, mixed, gradient, stationarity_tolerance=1e-8):
    """Numerical diagnostics, not a proof of exact stationarity/optimality.

    Check the full parameterization: softmax gauge null directions must not be
    hidden by a pseudoinverse. A different parameterization must be explicit.
    """
    h, mixed, g = map(lambda a: np.asarray(a, float), (hessian, mixed, gradient))
    if h.ndim != 2 or h.shape[0] != h.shape[1] or mixed.shape[0] != h.shape[0] or g.shape != (h.shape[0],):
        raise ValueError('Incompatible derivative shapes.')
    if not all(np.all(np.isfinite(a)) for a in (h, mixed, g)):
        raise ValueError('Nonfinite derivatives.')
    symmetry = float(np.linalg.norm(h - h.T, 2))
    symmetric = symmetry <= 1e-10 * max(1.0, float(np.linalg.norm(h, 2)))
    eig = np.linalg.eigvalsh((h + h.T) / 2)
    m = float(-eig[-1])
    stationary = bool(np.linalg.norm(g) <= stationarity_tolerance)
    accepted = symmetric and m > 0 and stationary
    s = -np.linalg.solve(h, mixed) if symmetric and m > 0 else None
    return dict(gradient_norm=float(np.linalg.norm(g)), symmetry_error=symmetry,
                m=m, numerically_stationary=stationary,
                numerical_conditions_pass=accepted, sensitivity=s)
