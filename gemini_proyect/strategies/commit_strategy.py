"""Commit-and-verify strategy.

Motivation
----------
In deep / rough cells the conditional detectability ``rho`` can be small.
A single negative observation then carries very little evidence: the
posterior probability that the object is in that cell only drops by a
factor ``(1 - rho)``. A purely greedy strategy moves to the next cell
after one miss, so it never spends enough probes on hard cells to either
detect the object or convincingly rule the cell out.

The commit-and-verify strategy fixes this:

1. Pick an initial rectangle ``R`` with a base proposer
   (default: ``max_expected_detection``).
2. **Keep probing the same rectangle** while the posterior mass inside
   ``R`` is still above ``1 - confidence``.
   * "I am confidence-fraction sure the object is not in ``R``" means
     ``P(Z in R | data) <= 1 - confidence``.
3. As soon as ``P(Z in R | data) < 1 - confidence`` (the zone is
   "ruled out"), or the object is detected, release the commitment and
   pick a fresh rectangle using the base proposer.

Because the strategy is stateful (it remembers the committed rectangle
between calls), it is implemented as a class. The Streamlit app keeps an
instance in ``st.session_state``; offline experiments can create one per
trial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from ..modeling.detection import DetectionModel
from ..simulator.grid import GridInfo
from .strategies import (
    DEFAULT_EFFORTS,
    DEFAULT_HEIGHTS,
    DEFAULT_WIDTHS,
    MissionProposal,
    _enumerate_index_rectangles,
    _integral_image,
    _rect_sums,
    propose_info_gain,
    propose_max_expected_detection,
    propose_max_posterior_rect,
)


_BASE_PROPOSERS: dict[str, Callable] = {
    "max_expected_detection": propose_max_expected_detection,
    "info_gain": propose_info_gain,
    "max_posterior_rect": propose_max_posterior_rect,
}


def _pick_commitable_rectangle(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    budget_remaining: float,
    min_mass: float,
    effort_levels: Iterable[int],
    widths: Iterable[int],
    heights: Iterable[int],
    min_cells: int = 9,
) -> MissionProposal:
    """Pick the rectangle to commit to.

    Among all candidate rectangles whose posterior mass is at least
    ``min_mass``, return the one that maximises expected detection per
    unit cost. If no rectangle satisfies the mass constraint, fall back
    to the rectangle with the largest available mass (so the strategy
    still produces a proposal when the posterior is too diffuse to
    commit at the requested confidence).
    """
    posterior_2d = posterior.reshape(grid.Ny, grid.Nx)
    depth_2d = grid.depth.reshape(grid.Ny, grid.Nx)
    roughness_2d = grid.roughness.reshape(grid.Ny, grid.Nx)
    S_pi = _integral_image(posterior_2d)

    ix0, ix1, iy0, iy1 = _enumerate_index_rectangles(grid.Nx, grid.Ny, widths, heights)
    r0, r1, c0, c1 = iy0, iy1 + 1, ix0, ix1 + 1
    n_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    mass = _rect_sums(S_pi, r0, r1, c0, c1)

    best: MissionProposal | None = None
    best_score = -np.inf
    fallback_idx = int(np.argmax(mass))
    fallback_effort = sorted(effort_levels)[0]

    for e in effort_levels:
        rho_2d = model.rho(depth_2d, roughness_2d, e)
        S_pi_rho = _integral_image(posterior_2d * rho_2d)
        expected_det = _rect_sums(S_pi_rho, r0, r1, c0, c1)
        cost = e * n_cells
        feasible = (
            (cost <= budget_remaining)
            & (mass >= min_mass)
            & (cost > 0)
            & (n_cells >= min_cells)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(feasible, expected_det / cost, -np.inf)
        k = int(np.argmax(score))
        if np.isfinite(score[k]) and score[k] > best_score:
            best_score = float(score[k])
            best = MissionProposal(
                x_min=float(ix0[k]) + 0.5,
                x_max=float(ix1[k]) + 0.5,
                y_min=float(iy0[k]) + 0.5,
                y_max=float(iy1[k]) + 0.5,
                effort=int(e),
                n_cells=int(n_cells[k]),
                cost=float(cost[k]),
                score=float(mass[k]),
            )

    if best is not None:
        return best

    # No rectangle satisfied min_mass within budget: commit to whichever
    # rectangle holds the largest mass at the cheapest affordable effort.
    k = fallback_idx
    chosen_effort = None
    for e in sorted(effort_levels):
        if e * n_cells[k] <= budget_remaining:
            chosen_effort = e
            break
    if chosen_effort is None:
        raise RuntimeError("No candidate rectangle fits the remaining budget.")
    return MissionProposal(
        x_min=float(ix0[k]) + 0.5,
        x_max=float(ix1[k]) + 0.5,
        y_min=float(iy0[k]) + 0.5,
        y_max=float(iy1[k]) + 0.5,
        effort=int(chosen_effort),
        n_cells=int(n_cells[k]),
        cost=float(chosen_effort * n_cells[k]),
        score=float(mass[k]),
    )


@dataclass
class CommitAndVerifyStrategy:
    """Stateful policy: commit to a rectangle, verify it is empty, then move."""

    confidence: float = 0.80
    base_proposer_name: str = "max_expected_detection"
    effort_levels: Iterable[int] = (2, 3)
    widths: Iterable[int] = (3, 4, 5, 6, 8)
    heights: Iterable[int] = (3, 4, 5, 6, 8)
    min_cells: int = 9

    # internal state (persisted across calls)
    committed_rect: Optional[tuple[float, float, float, float, int]] = None
    committed_cells: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def base_proposer(self) -> Callable:
        return _BASE_PROPOSERS[self.base_proposer_name]

    def reset(self) -> None:
        self.committed_rect = None
        self.committed_cells = None

    def _propose_new(
        self,
        posterior: np.ndarray,
        grid: GridInfo,
        model: DetectionModel,
        budget_remaining: float,
    ) -> MissionProposal:
        # The new commitment must hold at least (1 − confidence) of the mass,
        # otherwise the "verify it is empty" condition would already be true
        # the moment we picked it, which defeats the purpose.
        min_mass = 1.0 - self.confidence
        prop = _pick_commitable_rectangle(
            posterior=posterior,
            grid=grid,
            model=model,
            budget_remaining=budget_remaining,
            min_mass=min_mass,
            effort_levels=self.effort_levels,
            widths=self.widths,
            heights=self.heights,
            min_cells=self.min_cells,
        )
        self.committed_rect = (prop.x_min, prop.x_max, prop.y_min, prop.y_max, prop.effort)
        self.committed_cells = grid.cells_in_rectangle(
            prop.x_min, prop.x_max, prop.y_min, prop.y_max
        )
        return prop

    def mass_in_commitment(self, posterior: np.ndarray) -> float:
        if self.committed_cells is None or self.committed_cells.size == 0:
            return 0.0
        return float(posterior[self.committed_cells].sum())

    def propose(
        self,
        posterior: np.ndarray,
        grid: GridInfo,
        model: DetectionModel,
        budget_remaining: float,
    ) -> MissionProposal:
        threshold = 1.0 - self.confidence

        # Re-use the committed rectangle while it still has enough probability.
        if self.committed_rect is not None and self.committed_cells is not None:
            mass = self.mass_in_commitment(posterior)
            if mass > threshold:
                x_min, x_max, y_min, y_max, effort = self.committed_rect
                n = int(self.committed_cells.size)
                cost = effort * n
                if cost <= budget_remaining:
                    return MissionProposal(
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        effort=effort,
                        n_cells=n,
                        cost=cost,
                        score=mass,
                    )
                # Same rectangle but cheaper effort if we can no longer afford the commit.
                cheaper_efforts = sorted(
                    e for e in self.effort_levels if e < effort and e * n <= budget_remaining
                )
                if cheaper_efforts:
                    e = cheaper_efforts[-1]  # the highest still-affordable level
                    self.committed_rect = (x_min, x_max, y_min, y_max, e)
                    return MissionProposal(
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        effort=e,
                        n_cells=n,
                        cost=e * n,
                        score=mass,
                    )
                # Can't afford the committed rectangle at any allowed effort;
                # fall through to picking a new one.

        # No live commitment (or zone ruled out / unaffordable). Pick anew.
        return self._propose_new(posterior, grid, model, budget_remaining)


def propose_commit_and_verify(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    budget_remaining: float,
    state: CommitAndVerifyStrategy,
    **_unused,
) -> MissionProposal:
    """Functional wrapper around ``CommitAndVerifyStrategy.propose``.

    ``state`` is the persistent object that holds the current commitment.
    """
    return state.propose(posterior, grid, model, budget_remaining)
