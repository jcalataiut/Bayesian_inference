"""Posterior-sampling (Thompson-style) search strategy.

Motivation
----------
All the deterministic strategies (max-expected-detection, info-gain,
max-posterior, commit-and-verify) pick the *same* rectangle whenever the
posterior is the same. With small posteriors this is fine; with broad,
multi-modal posteriors it can lock the search into one local mode.

A stochastic strategy injects randomness in a probabilistically principled
way:

1. **Sample** a candidate cell ``j*`` from the current posterior
   ``π``. This is "Thompson sampling" on the latent variable ``Z``: we
   act as if ``j*`` were the true location.
2. **Build** a small set of rectangles centred near ``j*`` (clamped to
   the grid bounds, with a few allowed widths and heights).
3. **Choose** the rectangle that maximises *expected detection per unit
   cost* under the posterior — i.e. we still spend the budget wisely
   conditional on the sampled hypothesis.

Because step 1 is random, two consecutive calls with the same posterior
can propose different rectangles. The frequency with which a region is
explored matches the posterior probability that the object lives there
— a classic Thompson-sampling property.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..modeling.detection import DetectionModel
from ..simulator.grid import GridInfo
from .strategies import (
    DEFAULT_EFFORTS,
    DEFAULT_HEIGHTS,
    DEFAULT_WIDTHS,
    MissionProposal,
)


def _build_rectangle_around(
    grid: GridInfo, center_ix: int, center_iy: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """Smallest index rectangle of size (w, h) containing the centre cell,
    clamped to the grid bounds."""
    ix0 = center_ix - w // 2
    iy0 = center_iy - h // 2
    ix0 = max(0, min(grid.Nx - w, ix0))
    iy0 = max(0, min(grid.Ny - h, iy0))
    return ix0, ix0 + w - 1, iy0, iy0 + h - 1


@dataclass
class StochasticPosteriorSampler:
    """Stateful so the same instance keeps a reproducible RNG across calls.

    The default rectangle pool is intentionally biased toward medium sizes
    (3..5) and the default effort levels exclude 1: a single low-effort
    probe centred on a sampled cell rarely produces evidence in deep/rough
    cells, so Thompson sampling with that setting degenerates into a
    diffuse random walk that never detects anything. Using slightly larger
    rectangles and higher effort gives each random sample a real shot.
    """

    seed: int = 0
    widths: Iterable[int] = (3, 4, 5)
    heights: Iterable[int] = (3, 4, 5)
    effort_levels: Iterable[int] = (2, 3)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _sample_centre_cell(self, posterior: np.ndarray, grid: GridInfo) -> int:
        p = np.clip(posterior, 0.0, None)
        s = p.sum()
        if s <= 0:
            # Degenerate: fall back to uniform sampling.
            return int(self.rng.integers(0, grid.n_cells))
        return int(self.rng.choice(grid.n_cells, p=p / s))

    def propose(
        self,
        posterior: np.ndarray,
        grid: GridInfo,
        model: DetectionModel,
        budget_remaining: float,
    ) -> MissionProposal:
        # Step 1: sample the candidate cell from the posterior.
        j_star = self._sample_centre_cell(posterior, grid)
        center_ix = int(round(grid.x[j_star] - 0.5))
        center_iy = int(round(grid.y[j_star] - 0.5))

        # Step 2: enumerate rectangles centred near j*.
        best: MissionProposal | None = None
        best_score = -np.inf
        for w in self.widths:
            for h in self.heights:
                ix0, ix1, iy0, iy1 = _build_rectangle_around(grid, center_ix, center_iy, w, h)
                cells = grid.cells_in_rectangle(ix0 + 0.5, ix1 + 0.5, iy0 + 0.5, iy1 + 0.5)
                n = int(cells.size)
                if n == 0:
                    continue
                rho = model.rho(grid.depth[cells], grid.roughness[cells], 1)
                mass = float(posterior[cells].sum())
                for e in self.effort_levels:
                    cost = e * n
                    if cost <= 0 or cost > budget_remaining:
                        continue
                    rho_e = model.rho(grid.depth[cells], grid.roughness[cells], e)
                    expected_det = float(np.sum(posterior[cells] * rho_e))
                    # Score = raw expected detection probability (not per
                    # unit cost). Thompson already injects exploration via
                    # the random j*, so the per-rectangle decision should
                    # favour rectangles that maximise the chance of a hit.
                    score = expected_det
                    if score > best_score:
                        best_score = score
                        best = MissionProposal(
                            x_min=ix0 + 0.5,
                            x_max=ix1 + 0.5,
                            y_min=iy0 + 0.5,
                            y_max=iy1 + 0.5,
                            effort=int(e),
                            n_cells=n,
                            cost=float(cost),
                            score=float(mass),
                        )
        if best is None:
            raise RuntimeError("No candidate rectangle around j* fits the remaining budget.")
        return best


def propose_thompson(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    budget_remaining: float,
    state: Optional[StochasticPosteriorSampler] = None,
    **_unused,
) -> MissionProposal:
    """Stateless-looking wrapper; creates a one-shot sampler if no state given.

    For Monte Carlo benchmarks, pass a persistent ``state`` so the RNG is
    advanced consistently across the campaign.
    """
    sampler = state or StochasticPosteriorSampler(seed=int(np.random.SeedSequence().entropy & 0xFFFFFFFF))
    return sampler.propose(posterior, grid, model, budget_remaining)
