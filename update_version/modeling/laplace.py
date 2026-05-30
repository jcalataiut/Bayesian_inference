"""Laplace approximation of the posterior on (beta, tau).

For our model the latent space has only 7 free parameters
``(b0, b1, b2, b3, b4, tau_fall, tau_drift)``. Running full NUTS on this
posterior takes 5-10 seconds; for a 1000-trial Monte Carlo with ~5 missions
each that means hours. The Laplace approximation -- *find the mode and
locally approximate the posterior by a Gaussian using the Hessian* -- is
the same idea that INLA exploits for high-dimensional latent Gaussian
models. In 7 dimensions it is essentially indistinguishable from INLA's
"simplified Laplace strategy".

What we compute, for each posterior update:

    theta_unc          unconstrained parameterisation
                       (b0, b1, b2, log_b3, log_b4, log_tau_fall, log_tau_drift)
    theta_MAP          = argmax  log P(theta_unc | history)
    H                  = -nabla^2 log P  evaluated at theta_MAP
    p(theta_unc | h)  ~~  N(theta_MAP, H^{-1})

We then propagate samples through ``softmax`` to obtain
``E[pi_j | history]`` which is the marginal cell prior used by the
strategy to choose the next mission.

Why unconstrained parameterisation?
-----------------------------------
``b3, b4`` have a HalfNormal prior (must be >= 0) and ``tau_fall, tau_drift``
have a LogNormal prior (must be > 0). Without reparameterisation the
Hessian at the boundary is degenerate. Working with ``log_b3, log_b4,
log_tau_fall, log_tau_drift`` removes the constraint and makes the
posterior more nearly Gaussian.

Inactive parameters
-------------------
Depending on ``PriorSpec.form``, some betas are fixed at 0 (linear: no
quadratic terms; quadratic: no linear terms). The dimension of theta_unc
adapts automatically: only active params enter the optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from ..simulator.grid import GridInfo
from .detection import DetectionModel
from .features import (
    ACCIDENT_POINT,
    ALPHA_WIND,
    V_DRIFT,
    V_PLANE,
    V_WIND,
    trajectory_axes,
)
from .priors_logistic import PriorSpec


# --------------------------------------------------------------------------- #
# Active parameter bookkeeping
# --------------------------------------------------------------------------- #
def _active_params(spec: PriorSpec) -> list[str]:
    """Names of free parameters in canonical order, given the prior spec."""
    out = ["b0"]
    if spec.b1_active:
        out.append("b1")
    if spec.b2_active:
        out.append("b2")
    if spec.b3_active:
        out.append("log_b3")
    if spec.b4_active:
        out.append("log_b4")
    out.append("log_tau_f")
    out.append("log_tau_d")
    return out


def _initial_theta(spec: PriorSpec) -> np.ndarray:
    """Reasonable starting point (prior means / Mahalanobis-aware modes)."""
    th = []
    for name in _active_params(spec):
        if name == "b0":
            th.append(0.0)
        elif name == "b1":
            th.append(float(spec.b1_mean))
        elif name == "b2":
            th.append(float(spec.b2_mean))
        elif name == "log_b3":
            # MAP of HalfNormal is at 0 -> log(0) singular; start at the
            # HalfNormal mean = sigma * sqrt(2/pi).
            th.append(float(np.log(max(spec.b3_sigma * np.sqrt(2 / np.pi), 1e-6))))
        elif name == "log_b4":
            th.append(float(np.log(max(spec.b4_sigma * np.sqrt(2 / np.pi), 1e-6))))
        elif name == "log_tau_f":
            th.append(0.0)
        elif name == "log_tau_d":
            th.append(0.0)
    return np.asarray(th, dtype=float)


def _unpack(theta: np.ndarray, spec: PriorSpec) -> tuple[float, ...]:
    """Convert unconstrained theta -> (b0, b1, b2, b3, b4, tau_f, tau_d)."""
    names = _active_params(spec)
    b0 = 0.0; b1 = 0.0; b2 = 0.0
    b3 = 0.0; b4 = 0.0
    tau_f = 1.0; tau_d = 1.0
    for n, v in zip(names, theta):
        if n == "b0":
            b0 = float(v)
        elif n == "b1":
            b1 = float(v)
        elif n == "b2":
            b2 = float(v)
        elif n == "log_b3":
            b3 = float(np.exp(v))
        elif n == "log_b4":
            b4 = float(np.exp(v))
        elif n == "log_tau_f":
            tau_f = float(np.exp(v))
        elif n == "log_tau_d":
            tau_d = float(np.exp(v))
    return b0, b1, b2, b3, b4, tau_f, tau_d


# --------------------------------------------------------------------------- #
# Vectorised pi(theta)
# --------------------------------------------------------------------------- #
def compute_pi(
    b0: float, b1: float, b2: float, b3: float, b4: float,
    tau_f: float, tau_d: float,
    grid: GridInfo, spec: PriorSpec,
) -> np.ndarray:
    """Softmax cell distribution honouring ``spec.uniform_mix``."""
    d_norm, n_norm = trajectory_axes()
    mu_x = ACCIDENT_POINT[0] + tau_f * (V_PLANE[0] + ALPHA_WIND * V_WIND[0]) + tau_d * V_DRIFT[0]
    mu_y = ACCIDENT_POINT[1] + tau_f * (V_PLANE[1] + ALPHA_WIND * V_WIND[1]) + tau_d * V_DRIFT[1]
    dx = grid.x - mu_x
    dy = grid.y - mu_y
    d_long = dx * d_norm[0] + dy * d_norm[1]
    d_trans = dx * n_norm[0] + dy * n_norm[1]
    eta = b0 + b1 * d_long + b2 * d_trans - b3 * d_long ** 2 - b4 * d_trans ** 2
    eta = eta - eta.max()
    pi = np.exp(eta)
    pi = pi / pi.sum()
    if spec.uniform_mix > 0:
        pi = (1.0 - spec.uniform_mix) * pi + spec.uniform_mix / pi.size
        pi = pi / pi.sum()
    return pi


# --------------------------------------------------------------------------- #
# Negative log posterior (target for minimisation)
# --------------------------------------------------------------------------- #
def neg_log_posterior(
    theta: np.ndarray,
    spec: PriorSpec,
    detector: DetectionModel,
    history: Sequence,
    grid: GridInfo,
) -> float:
    """Negative log joint = -(log prior + log likelihood)."""
    b0, b1, b2, b3, b4, tau_f, tau_d = _unpack(theta, spec)
    names = _active_params(spec)
    idx = {n: i for i, n in enumerate(names)}

    log_p = 0.0
    log_p += -0.5 * (theta[idx["b0"]] / spec.b0_sigma) ** 2
    if "b1" in idx:
        log_p += -0.5 * ((theta[idx["b1"]] - spec.b1_mean) / spec.b1_sigma) ** 2
    if "b2" in idx:
        log_p += -0.5 * ((theta[idx["b2"]] - spec.b2_mean) / spec.b2_sigma) ** 2
    if "log_b3" in idx:
        # HalfNormal jacobian: log p(log_b3) = log p_HN(b3) + log_b3
        log_p += -0.5 * (b3 / spec.b3_sigma) ** 2 + theta[idx["log_b3"]]
    if "log_b4" in idx:
        log_p += -0.5 * (b4 / spec.b4_sigma) ** 2 + theta[idx["log_b4"]]
    log_p += -0.5 * (theta[idx["log_tau_f"]] / spec.tau_fall_sigma) ** 2
    log_p += -0.5 * (theta[idx["log_tau_d"]] / spec.tau_drift_sigma) ** 2

    pi = compute_pi(b0, b1, b2, b3, b4, tau_f, tau_d, grid, spec)
    temp = float(spec.likelihood_temperature)
    for m in history:
        rho = detector.rho(grid.depth, grid.roughness, m.effort)
        covered = grid.coverage_mask(m.x_min, m.x_max, m.y_min, m.y_max)
        D = float(np.sum(pi * rho * covered))
        D = min(max(D, 1e-12), 1.0 - 1e-12)
        if m.s_t == 1:
            log_p += temp * np.log(D)
        else:
            log_p += temp * np.log(1.0 - D)
    return -log_p


# --------------------------------------------------------------------------- #
# Numerical Hessian via centred differences
# --------------------------------------------------------------------------- #
def _numerical_hessian(f, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            f_pp = f(x + ei + ej)
            f_pn = f(x + ei - ej)
            f_np = f(x - ei + ej)
            f_nn = f(x - ei - ej)
            H[i, j] = (f_pp - f_pn - f_np + f_nn) / (4.0 * h * h)
            H[j, i] = H[i, j]
    # Diagonal correction is more accurate with centred second derivative
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        H[i, i] = (f(x + ei) - 2.0 * fx + f(x - ei)) / (h * h)
    return H


# --------------------------------------------------------------------------- #
# Public: Laplace approximation
# --------------------------------------------------------------------------- #
@dataclass
class LaplacePosterior:
    """Output of ``laplace_approx``."""

    param_names: list[str]
    theta_map: np.ndarray
    cov: np.ndarray
    spec: PriorSpec
    converged: bool


def laplace_approx(
    spec: PriorSpec,
    detector: DetectionModel,
    history: Sequence,
    grid: GridInfo,
    *,
    init_theta: np.ndarray | None = None,
    max_iter: int = 80,
    hessian_h: float = 1e-3,
    cov_floor: float = 1e-6,
) -> LaplacePosterior:
    """Compute MAP and posterior covariance via Laplace approximation.

    The optimiser used is L-BFGS-B with finite-difference gradients. For
    our 7-dimensional problem this converges in ~30-60 ms per call. The
    Hessian is then estimated with centred finite differences (~10-20 ms
    extra) and inverted with eigenvalue regularisation to obtain a valid
    covariance.
    """
    if init_theta is None:
        init_theta = _initial_theta(spec)
    else:
        init_theta = np.asarray(init_theta, dtype=float)
        if init_theta.size != len(_active_params(spec)):
            init_theta = _initial_theta(spec)

    def neg_lp(th):
        return neg_log_posterior(th, spec, detector, history, grid)

    res = minimize(
        neg_lp,
        init_theta,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-4},
    )
    theta_map = res.x
    converged = bool(res.success)

    H = _numerical_hessian(neg_lp, theta_map, h=hessian_h)
    H_sym = 0.5 * (H + H.T) + cov_floor * np.eye(len(theta_map))
    try:
        cov = np.linalg.inv(H_sym)
        # Force PSD via eigenvalue floor.
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, cov_floor)
        cov = (V * w) @ V.T
    except np.linalg.LinAlgError:
        cov = np.eye(len(theta_map))

    return LaplacePosterior(
        param_names=_active_params(spec),
        theta_map=theta_map,
        cov=cov,
        spec=spec,
        converged=converged,
    )


def laplace_pi_mean(
    lp: LaplacePosterior,
    grid: GridInfo,
    n_samples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Estimate E_{theta ~ N(theta_MAP, cov)}[pi_j(theta)] by Monte Carlo."""
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(lp.cov)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(lp.cov), 1e-8)))
    accum = np.zeros(grid.n_cells, dtype=float)
    for _ in range(n_samples):
        z = rng.normal(size=len(lp.theta_map))
        theta = lp.theta_map + L @ z
        b0, b1, b2, b3, b4, tau_f, tau_d = _unpack(theta, lp.spec)
        accum += compute_pi(b0, b1, b2, b3, b4, tau_f, tau_d, grid, lp.spec)
    return accum / n_samples


def laplace_pi_map(lp: LaplacePosterior, grid: GridInfo) -> np.ndarray:
    """Cheap alternative: pi_j evaluated at the MAP only (no MC, ~0 cost)."""
    b0, b1, b2, b3, b4, tau_f, tau_d = _unpack(lp.theta_map, lp.spec)
    return compute_pi(b0, b1, b2, b3, b4, tau_f, tau_d, grid, lp.spec)
