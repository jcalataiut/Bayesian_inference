"""Convert MCMC samples into the posterior over Z and replay helpers.

Two posteriors live in this module:

1. **Posterior over the latent coefficients beta (and tau)**: stored directly
   in the ``arviz.InferenceData`` returned by ``run_mcmc``.

2. **Posterior over the unknown cell Z**: derived from (1) by averaging the
   per-sample categorical distribution over cells.

For each MCMC sample ``s`` we have a draw of (beta^s, tau^s). Each draw
defines a per-cell prior pi_j^s = softmax_j(eta_j(beta^s, tau^s)). Given the
mission history s_{1:T}, the per-cell likelihood

    L_j = prod_t  q_{t,j}^{s_t} * (1 - q_{t,j})^{1-s_t}

is *the same for every sample* (it does not depend on beta), so the per-sample
posterior over Z is

    p_j^s = pi_j^s * L_j / Z^s,    with  Z^s = sum_k pi_k^s * L_k.

The marginal posterior over Z is the average of p_j^s over MCMC samples.

This module also exposes ``replay_posteriors`` which computes the sequence
[posterior_after_0_missions, after_1, ..., after_T] using a fresh MCMC run
per step. That's expensive but it's how we make heatmap animations for the
notebook and the app.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import arviz as az  # noqa: F401
except ImportError:  # pragma: no cover
    az = None

from ..simulator.environment import MissionRecord
from ..simulator.grid import GridInfo
from .detection import DetectionModel
from .pymc_model import run_mcmc
from .priors_logistic import PriorSpec


# --------------------------------------------------------------------------- #
# Posterior over Z given an existing MCMC trace
# --------------------------------------------------------------------------- #
def posterior_pi(idata) -> np.ndarray:
    """Mean prior pi_j across all MCMC samples (length-N array).

    This is E_{beta, tau ~ posterior}[pi_j], i.e. the marginal cell prior
    implied by the current posterior over the parameters -- BEFORE applying
    the mission-likelihood reweighting. Useful for comparing prior vs
    parameter-posterior-induced prior.
    """
    log_pi = idata.posterior["log_pi"].values  # (chain, draw, cell)
    log_pi = log_pi.reshape(-1, log_pi.shape[-1])
    pi = np.exp(log_pi)
    return pi.mean(axis=0)


def _per_cell_log_likelihood(
    grid: GridInfo,
    detector: DetectionModel,
    history: Sequence[MissionRecord],
) -> np.ndarray:
    """log L_j = sum_t [ s_t * log q_{t,j} + (1 - s_t) * log (1 - q_{t,j}) ].

    Cells outside R_t have q_{t,j} = 0. For a s_t = 1 mission this gives
    log 0 = -inf for those cells (correctly ruling them out). For s_t = 0 it
    contributes 0, leaving those cells untouched.
    """
    log_L = np.zeros(grid.n_cells, dtype=float)
    for m in history:
        rho = np.asarray(detector.rho(grid.depth, grid.roughness, m.effort), dtype=float)
        covered = grid.coverage_mask(m.x_min, m.x_max, m.y_min, m.y_max)
        q = np.where(covered, rho, 0.0)
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        if m.s_t == 1:
            with np.errstate(divide="ignore"):
                log_L = log_L + np.where(covered, np.log(q), -np.inf)
        else:
            log_L = log_L + np.log1p(-q)
    return log_L


def posterior_over_Z(
    idata,
    grid: GridInfo,
    detector: DetectionModel,
    history: Sequence[MissionRecord],
) -> np.ndarray:
    """Compute P(Z=j | history) = E_{beta ~ posterior}[pi_j^s L_j / Z^s].

    Numerically stable: works in log-space per sample, normalises each
    sample's distribution, then averages probabilities across samples.
    """
    log_pi_samples = idata.posterior["log_pi"].values  # (chain, draw, N)
    log_pi_samples = log_pi_samples.reshape(-1, log_pi_samples.shape[-1])
    log_L = _per_cell_log_likelihood(grid, detector, history)

    accum = np.zeros(grid.n_cells, dtype=float)
    for log_pi in log_pi_samples:
        log_post = log_pi + log_L
        m = log_post.max()
        if not np.isfinite(m):
            # All cells ruled out under this sample: skip.
            continue
        p = np.exp(log_post - m)
        s = p.sum()
        if s <= 0:
            continue
        accum += p / s
    if accum.sum() <= 0:
        # Should not happen, but fall back to uniform.
        return np.full(grid.n_cells, 1.0 / grid.n_cells)
    return accum / accum.sum()


# --------------------------------------------------------------------------- #
# Replay: re-run MCMC after each mission to build a sequence of posteriors
# --------------------------------------------------------------------------- #
def replay_posteriors(
    spec: PriorSpec,
    detector: DetectionModel,
    history: Sequence[MissionRecord],
    grid: GridInfo,
    draws: int = 800,
    tune: int = 800,
    chains: int = 2,
    target_accept: float = 0.95,
    progressbar: bool = False,
) -> list[np.ndarray]:
    """Return [P(Z | nothing), P(Z | mission_1), ..., P(Z | mission_1..T)].

    Each step re-runs MCMC with the corresponding prefix of the history. Disk
    cache (in pymc_model.CACHE_DIR) means repeated calls are cheap.
    """
    posteriors = []
    for k in range(len(history) + 1):
        partial = list(history[:k])
        idata = run_mcmc(
            spec,
            detector,
            partial,
            grid,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            progressbar=progressbar,
        )
        posteriors.append(posterior_over_Z(idata, grid, detector, partial))
    return posteriors


def summarise_posterior_over_Z(p_Z: np.ndarray, grid: GridInfo, k_top: int = 5) -> dict:
    """Quick scalar summary of a P(Z) distribution for diagnostics."""
    p_Z = np.asarray(p_Z)
    sort_idx = np.argsort(-p_Z)
    top = sort_idx[:k_top]
    return {
        "argmax_cell": int(p_Z.argmax()),
        "argmax_xy": (float(grid.x[p_Z.argmax()]), float(grid.y[p_Z.argmax()])),
        "max_prob": float(p_Z.max()),
        "entropy_nats": float(-(p_Z * np.log(np.clip(p_Z, 1e-300, None))).sum()),
        "top_cells": [
            {
                "cell_id": int(c),
                "xy": (float(grid.x[c]), float(grid.y[c])),
                "p": float(p_Z[c]),
            }
            for c in top
        ],
    }
