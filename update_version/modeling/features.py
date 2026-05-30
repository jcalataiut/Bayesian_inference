"""Per-cell features for the logistic prior on Z.

Given the accident location ``x_E`` and the physical vectors (plane velocity,
wind, water drift), each cell ``j`` is described by two coordinates relative
to an expected landing point ``mu(tau_fall, tau_drift)``:

    d_long(j)  = signed projection of (cell_j - mu) onto the trajectory axis
    d_trans(j) = signed projection onto the perpendicular axis

The linear predictor of the logistic prior is then

    eta_j(beta, tau) = b0 + b1*d_long + b2*d_trans - b3*d_long**2 - b4*d_trans**2

so the prior is a softmax-Gaussian (lineal + cuadratic) controlled by 5 betas
and the two times tau_fall, tau_drift.

All functions are pure numpy and operate on length-N arrays of cell centers
(see ``simulator.grid.GridInfo``).
"""
from __future__ import annotations

import numpy as np


# Fixed accident inputs (from deliverable2.ipynb).
ACCIDENT_POINT = np.array([7.0, 20.0])
V_PLANE = np.array([6.0, -3.5])
V_WIND = np.array([-1.0, -1.5])
V_DRIFT = np.array([0.5, -1.5])
ALPHA_WIND = 0.5  # wind coupling during the fall (fixed, not Bayesian)


def expected_landing(tau_fall: float, tau_drift: float, alpha_wind: float = ALPHA_WIND) -> np.ndarray:
    """mu = x_E + tau_fall * (v_plane + alpha_wind * v_wind) + tau_drift * v_drift."""
    return (
        ACCIDENT_POINT
        + tau_fall * (V_PLANE + alpha_wind * V_WIND)
        + tau_drift * V_DRIFT
    )


def trajectory_axes(alpha_wind: float = ALPHA_WIND) -> tuple[np.ndarray, np.ndarray]:
    """Return (d_norm, n_norm): unit vectors along and across the trajectory.

    Built from the falling-trajectory direction ``v_plane + alpha_wind * v_wind``.
    The transverse axis is a 90-degree rotation of the trajectory axis.
    """
    direction = V_PLANE + alpha_wind * V_WIND
    d_norm = direction / (np.linalg.norm(direction) + 1e-12)
    n_norm = np.array([-d_norm[1], d_norm[0]])
    return d_norm, n_norm


def cell_distances(
    x: np.ndarray,
    y: np.ndarray,
    tau_fall: float,
    tau_drift: float,
    alpha_wind: float = ALPHA_WIND,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (d_long, d_trans) per cell relative to mu(tau_fall, tau_drift)."""
    mu = expected_landing(tau_fall, tau_drift, alpha_wind)
    d_norm, n_norm = trajectory_axes(alpha_wind)
    dx = x - mu[0]
    dy = y - mu[1]
    d_long = dx * d_norm[0] + dy * d_norm[1]
    d_trans = dx * n_norm[0] + dy * n_norm[1]
    return d_long, d_trans


def linear_predictor(
    beta: np.ndarray,
    d_long: np.ndarray,
    d_trans: np.ndarray,
) -> np.ndarray:
    """eta_j = b0 + b1*d_long + b2*d_trans - b3*d_long^2 - b4*d_trans^2.

    ``beta`` is length-5: [b0, b1, b2, b3, b4]. The quadratic coefficients
    must be non-negative for a Gaussian-like prior (HalfNormal prior in PyMC).
    """
    b0, b1, b2, b3, b4 = beta
    return b0 + b1 * d_long + b2 * d_trans - b3 * d_long ** 2 - b4 * d_trans ** 2


def softmax_pi(eta: np.ndarray) -> np.ndarray:
    """Stable softmax over cells producing a valid probability distribution."""
    eta = eta - eta.max()
    p = np.exp(eta)
    return p / p.sum()


def evaluate_pi(
    beta: np.ndarray,
    tau_fall: float,
    tau_drift: float,
    x: np.ndarray,
    y: np.ndarray,
    alpha_wind: float = ALPHA_WIND,
) -> np.ndarray:
    """Convenience: evaluate the prior pi_j for a single (beta, tau) draw."""
    d_long, d_trans = cell_distances(x, y, tau_fall, tau_drift, alpha_wind)
    eta = linear_predictor(beta, d_long, d_trans)
    return softmax_pi(eta)
