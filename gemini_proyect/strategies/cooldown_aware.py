"""Cooldown-aware search: few, large, decisive missions.

The professor's real webapp enforces a 12 h cooldown between missions, so
within a 230-coin budget you can realistically run only ~5–10 missions per
campaign. Strategies that propose 1×1 probes (info_gain / max_expected_
detection) need 50–200 missions to converge and are simply unusable in
practice.

This strategy is designed for that regime:

* **Floor the rectangle size.** Each rectangle must cover at least
  ``min_cells`` cells (default = budget / n_missions_target / max_effort).
  Tiny probes are excluded by construction.
* **Default to high effort.** Effort levels ``(2, 3)`` only — within a
  cell, more passes is the cheapest way to actually detect the object;
  a single low-effort pass on a deep / rough cell is wasted.
* **Maximise raw expected detection.** Score = Σ_j π_j · ρ(j, e). We do
  *not* divide by cost, because cost is already controlled by the
  rectangle-size floor and the effort range.
* **Maximum rectangle cost.** Each mission must also fit in the
  remaining budget — and, optionally, in *budget_remaining /
  remaining_missions_target* so we always leave room for the planned
  number of subsequent missions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..modeling.detection import DetectionModel
from ..simulator.grid import GridInfo
from .strategies import (
    MissionProposal,
    _enumerate_index_rectangles,
    _integral_image,
    _rect_sums,
)


DEFAULT_COOLDOWN_WIDTHS = (3, 4, 5, 6, 8, 10)
DEFAULT_COOLDOWN_HEIGHTS = (3, 4, 5, 6, 8, 10)
DEFAULT_COOLDOWN_EFFORTS = (2, 3)


def propose_cooldown_aware(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    budget_remaining: float,
    n_missions_target: int = 5,
    min_cells: int | None = None,
    max_cost_fraction: float = 0.60,
    effort_levels: Iterable[int] = DEFAULT_COOLDOWN_EFFORTS,
    widths: Iterable[int] = DEFAULT_COOLDOWN_WIDTHS,
    heights: Iterable[int] = DEFAULT_COOLDOWN_HEIGHTS,
) -> MissionProposal:
    """Pick a large, high-effort rectangle that maximises P(detect | mission).

    Parameters
    ----------
    n_missions_target
        How many missions the strategy still wants to run (informs the
        budget cap per mission). With ``budget_remaining = 230`` and
        ``n_missions_target = 5`` each mission is capped at ~46 coins by
        default — large enough to cover ~16 cells at effort 3.
    min_cells
        Floor on rectangle size. ``None`` → computed from the budget
        cap and the maximum effort level so it is *feasible* to pick a
        rectangle that uses the budget cap. Typically 9–25.
    max_cost_fraction
        Per-mission budget cap as a fraction of the remaining budget,
        divided across ``n_missions_target`` planned missions. The hard
        cap is ``budget_remaining``; this gives a soft cap so the
        strategy does not blow its plan on the first mission.
    """
    effort_levels = sorted(effort_levels)
    e_max = effort_levels[-1]

    per_mission_budget = (
        max_cost_fraction
        * budget_remaining
        / max(1, n_missions_target)
    )
    cost_cap = min(budget_remaining, max(per_mission_budget, e_max * 4))  # at least a 2x2 at e_max

    if min_cells is None:
        # Aim for medium-large rectangles by default.
        min_cells = max(9, int(per_mission_budget // e_max))

    posterior_2d = posterior.reshape(grid.Ny, grid.Nx)
    depth_2d = grid.depth.reshape(grid.Ny, grid.Nx)
    roughness_2d = grid.roughness.reshape(grid.Ny, grid.Nx)

    ix0, ix1, iy0, iy1 = _enumerate_index_rectangles(grid.Nx, grid.Ny, widths, heights)
    r0, r1, c0, c1 = iy0, iy1 + 1, ix0, ix1 + 1
    n_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    size_ok = n_cells >= min_cells

    best: MissionProposal | None = None
    best_score = -np.inf
    fallback_proposal: MissionProposal | None = None
    fallback_score = -np.inf

    for e in effort_levels:
        rho_2d = model.rho(depth_2d, roughness_2d, e)
        S_pi_rho = _integral_image(posterior_2d * rho_2d)
        expected_det = _rect_sums(S_pi_rho, r0, r1, c0, c1)
        cost = e * n_cells

        soft_feasible = (cost <= cost_cap) & (cost <= budget_remaining) & size_ok & (cost > 0)
        hard_feasible = (cost <= budget_remaining) & (cost > 0)

        for mask, score_arr, target_proposal, score_thresh in [
            (soft_feasible, expected_det, "best", "best_score"),
            (hard_feasible, expected_det, "fallback", "fallback_score"),
        ]:
            if not mask.any():
                continue
            score = np.where(mask, score_arr, -np.inf)
            k = int(np.argmax(score))
            if score[k] <= -np.inf:
                continue
            prop = MissionProposal(
                x_min=float(ix0[k]) + 0.5,
                x_max=float(ix1[k]) + 0.5,
                y_min=float(iy0[k]) + 0.5,
                y_max=float(iy1[k]) + 0.5,
                effort=int(e),
                n_cells=int(n_cells[k]),
                cost=float(cost[k]),
                score=float(score[k]),
            )
            if target_proposal == "best" and score[k] > best_score:
                best_score = float(score[k])
                best = prop
            elif target_proposal == "fallback" and score[k] > fallback_score:
                fallback_score = float(score[k])
                fallback_proposal = prop

    if best is not None:
        return best
    if fallback_proposal is not None:
        return fallback_proposal
    raise RuntimeError("No candidate rectangle fits the remaining budget.")
