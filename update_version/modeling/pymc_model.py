"""PyMC model + MCMC inference for the logistic prior on Z.

The model
---------

Latent parameters (per ``PriorSpec``):
    b0       ~ Normal(0, spec.b0_sigma)
    b1, b2   ~ Normal(spec.b{i}_mean, spec.b{i}_sigma)   (if active)
    b3, b4   ~ HalfNormal(spec.b{i}_sigma)               (if active)
    tau_fall ~ LogNormal(0, spec.tau_fall_sigma)
    tau_drift~ LogNormal(0, spec.tau_drift_sigma)

Deterministic transforms:
    mu        = x_E + tau_fall * (v_plane + alpha_wind * v_wind)
                    + tau_drift * v_drift
    d_long(j) = ((x_j, y_j) - mu) . d_norm
    d_trans(j)= ((x_j, y_j) - mu) . n_norm
    eta_j     = b0 + b1*d_long + b2*d_trans - b3*d_long**2 - b4*d_trans**2
    log_pi_j  = log_softmax_j(eta_j)

Mission likelihood (closed-form marginalization over the unknown Z):
    For each mission t with rectangle R_t, effort e_t, outcome s_t:
        s_in_R = sum_{j in R_t} pi_j * rho_{t,j}     (in [0, 1])
        log P(s_t | beta, tau) = log s_in_R              if s_t == 1
                               = log (1 - s_in_R)        if s_t == 0
    Added as ``pm.Potential`` to the joint log-density.

The detector ``rho_{t,j}`` is *fixed* (not Bayesian) so we precompute the
per-cell rho array for each unique (effort) once and reuse it inside the
PyMC graph.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt

try:
    import arviz as az
except ImportError:  # pragma: no cover
    az = None

from ..simulator.environment import MissionRecord
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


CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "cache"


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #
def build_model(
    spec: PriorSpec,
    detector: DetectionModel,
    history: Sequence[MissionRecord],
    grid: GridInfo,
    alpha_wind: float = ALPHA_WIND,
) -> pm.Model:
    """Construct the PyMC model for one (prior_spec, detector, history) combo."""
    d_norm, n_norm = trajectory_axes(alpha_wind)
    x_cells = np.asarray(grid.x, dtype=float)
    y_cells = np.asarray(grid.y, dtype=float)

    # Precompute per-mission (covered mask, rho array) -- constants from PyMC's
    # point of view. Stored as numpy arrays of length N.
    mission_data = []
    for m in history:
        covered = grid.coverage_mask(m.x_min, m.x_max, m.y_min, m.y_max).astype(float)
        rho_arr = np.asarray(
            detector.rho(grid.depth, grid.roughness, m.effort), dtype=float
        )
        rho_in_R = rho_arr * covered
        mission_data.append((m, rho_in_R))

    with pm.Model() as model:
        # ---- priors on the linear-predictor coefficients ---- #
        b0 = pm.Normal("b0", mu=0.0, sigma=spec.b0_sigma)
        b1 = (
            pm.Normal("b1", mu=spec.b1_mean, sigma=spec.b1_sigma)
            if spec.b1_active
            else pt.zeros(())
        )
        b2 = (
            pm.Normal("b2", mu=spec.b2_mean, sigma=spec.b2_sigma)
            if spec.b2_active
            else pt.zeros(())
        )
        b3 = (
            pm.HalfNormal("b3", sigma=spec.b3_sigma)
            if spec.b3_active
            else pt.zeros(())
        )
        b4 = (
            pm.HalfNormal("b4", sigma=spec.b4_sigma)
            if spec.b4_active
            else pt.zeros(())
        )

        # ---- priors on the latency times ---- #
        tau_fall = pm.LogNormal("tau_fall", mu=0.0, sigma=spec.tau_fall_sigma)
        tau_drift = pm.LogNormal("tau_drift", mu=0.0, sigma=spec.tau_drift_sigma)

        # ---- expected landing point mu(tau) ---- #
        mu_x = (
            ACCIDENT_POINT[0]
            + tau_fall * (V_PLANE[0] + alpha_wind * V_WIND[0])
            + tau_drift * V_DRIFT[0]
        )
        mu_y = (
            ACCIDENT_POINT[1]
            + tau_fall * (V_PLANE[1] + alpha_wind * V_WIND[1])
            + tau_drift * V_DRIFT[1]
        )

        # ---- per-cell linear predictor ---- #
        dx = pt.as_tensor_variable(x_cells) - mu_x
        dy = pt.as_tensor_variable(y_cells) - mu_y
        d_long = dx * d_norm[0] + dy * d_norm[1]
        d_trans = dx * n_norm[0] + dy * n_norm[1]

        eta = (
            b0
            + b1 * d_long
            + b2 * d_trans
            - b3 * d_long ** 2
            - b4 * d_trans ** 2
        )
        # log-softmax for numerical stability
        log_pi_pure = eta - pm.math.logsumexp(eta)
        pi_pure = pt.exp(log_pi_pure)

        # Optional uniform mixture: pi = (1-u)*pi_pure + u/N. Honoured at the
        # *cell-level* so the spec.uniform_mix and the prior_predictive_pi
        # produce the same distribution.
        u = float(spec.uniform_mix)
        if u > 0:
            pi = (1.0 - u) * pi_pure + u / grid.n_cells
            log_pi = pt.log(pi)
        else:
            pi = pi_pure
            log_pi = log_pi_pure

        pm.Deterministic("log_pi", log_pi)

        # ---- mission likelihoods (Potential, marginalized over Z) ---- #
        temp = float(spec.likelihood_temperature)
        for m, rho_in_R in mission_data:
            s_in_R = pt.sum(pi * pt.as_tensor_variable(rho_in_R))
            s_in_R = pt.clip(s_in_R, 1e-12, 1.0 - 1e-12)
            if m.s_t == 1:
                ll = pt.log(s_in_R)
            else:
                ll = pt.log(1.0 - s_in_R)
            # Tempered likelihood: temperature > 1 amplifies the influence
            # of each observation -> posterior shifts more aggressively.
            pm.Potential(f"mission_{m.mission_id}", temp * ll)

    return model


# --------------------------------------------------------------------------- #
# MCMC runner with disk cache
# --------------------------------------------------------------------------- #
def _history_signature(history: Sequence[MissionRecord]) -> str:
    payload = [
        {
            "x_min": m.x_min,
            "x_max": m.x_max,
            "y_min": m.y_min,
            "y_max": m.y_max,
            "effort": m.effort,
            "s_t": m.s_t,
        }
        for m in history
    ]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def cache_key(spec: PriorSpec, detector: DetectionModel, history: Sequence[MissionRecord]) -> str:
    sig = _history_signature(history)
    return f"{spec.name}__{detector.name}__{sig}"


def run_mcmc(
    spec: PriorSpec,
    detector: DetectionModel,
    history: Sequence[MissionRecord],
    grid: GridInfo,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    cores: int = 1,
    random_seed: int = 0,
    use_cache: bool = True,
    progressbar: bool = False,
) -> "az.InferenceData":
    """Build the model, sample with NUTS, cache to disk, return InferenceData."""
    if az is None:
        raise ImportError("arviz is required to run MCMC.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(spec, detector, history)
    cache_path = CACHE_DIR / f"{key}.nc"

    if use_cache and cache_path.exists():
        return az.from_netcdf(cache_path)

    model = build_model(spec, detector, history, grid)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    if use_cache:
        idata.to_netcdf(cache_path)
    return idata


def diagnostics_summary(idata: "az.InferenceData") -> "pd.DataFrame":  # noqa: F821
    """Return arviz summary with R-hat, ESS, mean, sd for the free parameters."""
    if az is None:
        raise ImportError("arviz is required.")
    var_names = [v for v in idata.posterior.data_vars if v != "log_pi"]
    return az.summary(idata, var_names=var_names, round_to=3)
