"""Bayesian posterior update given mission outcomes.

Setup
-----
Let Z be the (unknown) cell containing the object, with prior pi_j = P(Z=j).
A mission ``t`` with searched region R_t, effort e_t, and outcome s_t in
{0, 1} induces the per-cell detection probability

    q_{t,j} = c_{t,j} * rho_{t,j},        c_{t,j} = 1 if j in R_t else 0.

Assuming the detector is conditionally independent across missions given Z,
the likelihood of the full history s_{1:T} given Z = j is

    L_j = prod_t  q_{t,j}^{s_t} * (1 - q_{t,j})^{1 - s_t}.

The posterior is then pi_j * L_j renormalized.

Numerically we work in log space to avoid underflow for long histories.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..modeling.detection import DetectionModel
from ..simulator.environment import MissionRecord
from ..simulator.grid import GridInfo


def _coverage_mask(grid: GridInfo, m: MissionRecord) -> np.ndarray:
    return (
        (grid.x >= m.x_min)
        & (grid.x <= m.x_max)
        & (grid.y >= m.y_min)
        & (grid.y <= m.y_max)
    )


def log_posterior_update(
    prior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    history: Iterable[MissionRecord],
) -> np.ndarray:
    """Return log p(Z=j | history) up to an additive constant."""
    log_post = np.log(np.clip(prior, 1e-300, None))

    rho_all = None  # we recompute rho per mission since effort varies.
    for m in history:
        covered = _coverage_mask(grid, m)
        rho = model.rho(grid.depth, grid.roughness, m.effort)  # length-N
        q = np.where(covered, rho, 0.0)
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        if m.s_t == 1:
            # Only cells with q>0 can produce a detection. Cells outside R_t
            # have q=0, log(q)=-inf, which correctly zeros them out.
            with np.errstate(divide="ignore"):
                log_post = log_post + np.log(q)
        else:
            log_post = log_post + np.log1p(-q)
    return log_post


def posterior_update(
    prior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    history: Iterable[MissionRecord],
) -> np.ndarray:
    log_post = log_posterior_update(prior, grid, model, history)
    log_post = log_post - log_post.max()
    p = np.exp(log_post)
    s = p.sum()
    if s <= 0:
        # Degenerate: every hypothesis was ruled out. Fall back to prior.
        return np.asarray(prior) / np.asarray(prior).sum()
    return p / s


def mission_likelihood(
    grid: GridInfo,
    model: DetectionModel,
    mission: MissionRecord,
) -> np.ndarray:
    """Per-cell likelihood L_j = P(s_t | Z = j) for a single mission.

    For each cell j:
        q_j = c_{t,j} * rho_{t,j}        (q_j = 0 if cell not covered)
        L_j = q_j         if s_t == 1
        L_j = 1 - q_j     if s_t == 0
    """
    covered = _coverage_mask(grid, mission)
    rho = model.rho(grid.depth, grid.roughness, mission.effort)
    q = np.where(covered, rho, 0.0)
    if mission.s_t == 1:
        return q
    return 1.0 - q


def step_by_step_posterior(
    prior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    history: Iterable[MissionRecord],
) -> list[np.ndarray]:
    """Return [prior, posterior_after_mission_1, posterior_after_mission_2, ...].

    Useful for the Streamlit replay so the user can see the cumulative effect
    of each mission as a separate distribution.
    """
    states = [np.asarray(prior) / np.asarray(prior).sum()]
    accumulated: list[MissionRecord] = []
    for m in history:
        accumulated.append(m)
        states.append(posterior_update(prior, grid, model, accumulated))
    return states
