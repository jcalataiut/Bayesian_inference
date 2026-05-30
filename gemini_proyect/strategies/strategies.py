"""Search strategies that propose the next rectangular mission.

The professor's UI requires the searched region to be an axis-aligned
rectangle ``[x_min, x_max] x [y_min, y_max]``. All strategies below
respect this constraint.

Each strategy returns a ``MissionProposal`` with the rectangle, an effort
level, and the predicted cost (effort * n_cells).

Performance
-----------
With Nx=50, Ny=35 and many candidate rectangle sizes, the candidate space
contains tens of thousands of rectangles. We use 2-D integral images
(summed-area tables) so the sum of any quantity over any rectangle is
O(1), and the inner loop runs in pure NumPy vectorised form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..modeling.detection import DetectionModel
from ..simulator.grid import GridInfo


@dataclass
class MissionProposal:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    effort: int
    n_cells: int
    cost: float
    score: float

    def as_kwargs(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "effort": self.effort,
        }


# --------------------------------------------------------------------- #
#  Integral-image helpers
# --------------------------------------------------------------------- #
def _integral_image(arr2d: np.ndarray) -> np.ndarray:
    """Return a summed-area table padded with one zero row/column.

    For an input of shape (Ny, Nx), the output has shape (Ny+1, Nx+1) and
    satisfies ``S[i, j] = sum_{a<i, b<j} arr[a, b]``. The sum over rows
    ``[r0, r1)`` and cols ``[c0, c1)`` is then
    ``S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]``.
    """
    S = arr2d.cumsum(axis=0).cumsum(axis=1)
    out = np.zeros((arr2d.shape[0] + 1, arr2d.shape[1] + 1), dtype=arr2d.dtype)
    out[1:, 1:] = S
    return out


def _rect_sums(S: np.ndarray, r0: np.ndarray, r1: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Vectorised rectangle sums given an integral image ``S``."""
    return S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]


def _enumerate_index_rectangles(
    Nx: int,
    Ny: int,
    widths: Iterable[int],
    heights: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return four 1-D arrays (ix0, ix1, iy0, iy1) describing every rectangle
    of allowed widths and heights, with inclusive integer indices into the
    cell grid.
    """
    ix0_all, ix1_all, iy0_all, iy1_all = [], [], [], []
    for w in widths:
        for h in heights:
            ix0 = np.arange(0, Nx - w + 1)
            iy0 = np.arange(0, Ny - h + 1)
            IX0, IY0 = np.meshgrid(ix0, iy0, indexing="xy")
            ix0_all.append(IX0.ravel())
            ix1_all.append(IX0.ravel() + (w - 1))
            iy0_all.append(IY0.ravel())
            iy1_all.append(IY0.ravel() + (h - 1))
    return (
        np.concatenate(ix0_all),
        np.concatenate(ix1_all),
        np.concatenate(iy0_all),
        np.concatenate(iy1_all),
    )


def enumerate_candidate_rectangles(
    grid: GridInfo,
    widths: Iterable[int] = (2, 3, 4, 5, 6, 8),
    heights: Iterable[int] = (2, 3, 4, 5, 6, 8),
) -> list[tuple[int, int, int, int]]:
    """Convenience wrapper returning a Python list (used by tests / notebooks)."""
    ix0, ix1, iy0, iy1 = _enumerate_index_rectangles(grid.Nx, grid.Ny, widths, heights)
    return list(zip(ix0.tolist(), ix1.tolist(), iy0.tolist(), iy1.tolist()))


def _bounds_from_indices(ix0: int, ix1: int, iy0: int, iy1: int) -> tuple[float, float, float, float]:
    """Cell centres are at integer + 0.5, so the index rectangle ``[ix0, ix1]``
    is exactly the bounds ``[ix0 + 0.5, ix1 + 0.5]`` for the webapp.
    """
    return (ix0 + 0.5, ix1 + 0.5, iy0 + 0.5, iy1 + 0.5)


# --------------------------------------------------------------------- #
#  Core vectorised scorer shared by all three strategies.
# --------------------------------------------------------------------- #
def _score_all_rectangles(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    effort_levels: Iterable[int],
    budget_remaining: float,
    widths: Iterable[int],
    heights: Iterable[int],
    score_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (rect indices, effort, n_cells, cost, score) for all candidates
    that fit in the remaining budget.

    ``score_kind`` is one of:
        * "expected_detection"      : sum_j pi_j q_j / cost
        * "info_gain"               : (H0 - E[H | s]) / cost
        * "posterior_mass_x_rho"    : sum_j pi_j q_j (no cost normalisation)
    """
    posterior_2d = posterior.reshape(grid.Ny, grid.Nx)
    depth_2d = grid.depth.reshape(grid.Ny, grid.Nx)
    roughness_2d = grid.roughness.reshape(grid.Ny, grid.Nx)

    ix0, ix1, iy0, iy1 = _enumerate_index_rectangles(grid.Nx, grid.Ny, widths, heights)
    r0 = iy0
    r1 = iy1 + 1
    c0 = ix0
    c1 = ix1 + 1
    n_rect = ix0.size
    n_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)

    if score_kind == "info_gain":
        H0 = -np.sum(posterior_2d * np.log(np.clip(posterior_2d, 1e-300, None)))

    best_score = np.full(n_rect, -np.inf)
    best_effort = np.zeros(n_rect, dtype=int)
    best_cost = np.zeros(n_rect)

    for e in effort_levels:
        rho_2d = model.rho(depth_2d, roughness_2d, e)
        pi_rho = posterior_2d * rho_2d
        pi_log_rho = posterior_2d * np.log(np.clip(rho_2d, 1e-300, None))
        pi_log_1mrho = posterior_2d * np.log(np.clip(1.0 - rho_2d, 1e-300, None))

        S_pi_rho = _integral_image(pi_rho)
        # Expected detection per rectangle.
        expected_det = _rect_sums(S_pi_rho, r0, r1, c0, c1)

        cost = e * n_cells
        feasible = cost <= budget_remaining

        if score_kind == "expected_detection":
            with np.errstate(divide="ignore", invalid="ignore"):
                score = expected_det / cost
        elif score_kind == "posterior_mass_x_rho":
            score = expected_det.copy()
        elif score_kind == "info_gain":
            S_log_rho = _integral_image(pi_log_rho)
            S_log_1mrho = _integral_image(pi_log_1mrho)
            # Total posterior entropy contribution from cells *inside* the rect
            # under each branch. Cells outside R have q=0:
            #   s=1 -> their posterior weight is multiplied by 0 (irrelevant
            #          for renormalisation, contributes nothing to entropy
            #          numerator). We only need the numerator inside.
            #   s=0 -> their posterior weight unchanged.
            sum_pi_log_rho = _rect_sums(S_log_rho, r0, r1, c0, c1)
            sum_pi_log_1mrho = _rect_sums(S_log_1mrho, r0, r1, c0, c1)

            p_s1 = expected_det                          # P(s=1) = sum_j pi_j q_j
            p_s0 = 1.0 - p_s1

            # H(post | s=1):
            #   post1(j) ∝ pi_j q_j for j in R, 0 outside.
            #   H = -sum post1 log post1
            #     = -sum_{j in R} (pi_j q_j / p_s1) [log(pi_j q_j) - log p_s1]
            #     = log p_s1 - (1/p_s1) sum_{j in R} pi_j q_j log(pi_j q_j)
            #     = log p_s1 - (1/p_s1) (E_R[pi log pi within R weighted by q] + E_R[pi q log q])
            # We need sum_{j in R} pi_j q_j * log(pi_j q_j). That requires
            # pi_j q_j log(pi_j q_j) = pi_j rho_j (log pi_j + log rho_j). The
            # log pi_j term changes when posterior changes, so we cache its
            # integral image once outside the loop.
            # (Computed below.)

            log_pi = np.log(np.clip(posterior_2d, 1e-300, None))
            pi_rho_log_pi = pi_rho * log_pi
            pi_rho_log_rho = pi_rho * np.log(np.clip(rho_2d, 1e-300, None))
            S_pi_rho_log_pi = _integral_image(pi_rho_log_pi)
            S_pi_rho_log_rho = _integral_image(pi_rho_log_rho)
            sum_pi_q_log_piq = (
                _rect_sums(S_pi_rho_log_pi, r0, r1, c0, c1)
                + _rect_sums(S_pi_rho_log_rho, r0, r1, c0, c1)
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                H_s1 = np.where(
                    p_s1 > 0,
                    np.log(np.clip(p_s1, 1e-300, None)) - sum_pi_q_log_piq / np.where(p_s1 > 0, p_s1, 1.0),
                    0.0,
                )

            # H(post | s=0):
            #   post0(j) ∝ pi_j (1 - q_j). For j outside R, q_j = 0, so post0(j) ∝ pi_j.
            #   Splitting the entropy into in-R and out-R parts:
            #     H = log p_s0
            #         - (1/p_s0) [ sum_{j out} pi_j log pi_j
            #                    + sum_{j in R} pi_j (1 - rho_j) (log pi_j + log(1 - rho_j)) ]
            # We need:
            #     A = sum_{j out} pi_j log pi_j  =  total - sum_{j in R} pi_j log pi_j
            #     B = sum_{j in R} pi_j (1 - rho_j) (log pi_j + log(1 - rho_j))
            S_pi_log_pi = _integral_image(posterior_2d * log_pi)
            sum_pi_log_pi_in = _rect_sums(S_pi_log_pi, r0, r1, c0, c1)
            total_pi_log_pi = (posterior_2d * log_pi).sum()
            A = total_pi_log_pi - sum_pi_log_pi_in

            pi_1mrho = posterior_2d * (1.0 - rho_2d)
            pi_1mrho_log_pi = pi_1mrho * log_pi
            pi_1mrho_log_1mrho = pi_1mrho * np.log(np.clip(1.0 - rho_2d, 1e-300, None))
            S_pi_1mrho_log_pi = _integral_image(pi_1mrho_log_pi)
            S_pi_1mrho_log_1mrho = _integral_image(pi_1mrho_log_1mrho)
            B = (
                _rect_sums(S_pi_1mrho_log_pi, r0, r1, c0, c1)
                + _rect_sums(S_pi_1mrho_log_1mrho, r0, r1, c0, c1)
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                H_s0 = np.where(
                    p_s0 > 0,
                    np.log(np.clip(p_s0, 1e-300, None)) - (A + B) / np.where(p_s0 > 0, p_s0, 1.0),
                    0.0,
                )

            expected_H = p_s1 * H_s1 + p_s0 * H_s0
            info_gain = H0 - expected_H
            score = info_gain / cost
        else:
            raise ValueError(f"Unknown score_kind {score_kind!r}")

        # Apply budget mask.
        score = np.where(feasible, score, -np.inf)
        # Keep the best effort per rectangle.
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_effort = np.where(better, e, best_effort)
        best_cost = np.where(better, cost, best_cost)

    return ix0, ix1, iy0, iy1, n_cells, best_effort, best_cost, best_score


def _best_proposal(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    effort_levels: Iterable[int],
    budget_remaining: float,
    widths: Iterable[int],
    heights: Iterable[int],
    score_kind: str,
) -> MissionProposal:
    ix0, ix1, iy0, iy1, n_cells, best_effort, best_cost, best_score = _score_all_rectangles(
        posterior=posterior,
        grid=grid,
        model=model,
        effort_levels=effort_levels,
        budget_remaining=budget_remaining,
        widths=widths,
        heights=heights,
        score_kind=score_kind,
    )
    if not np.isfinite(best_score).any():
        raise RuntimeError("No candidate rectangle fits the remaining budget.")
    k = int(np.argmax(best_score))
    xb = _bounds_from_indices(int(ix0[k]), int(ix1[k]), int(iy0[k]), int(iy1[k]))
    return MissionProposal(
        x_min=xb[0],
        x_max=xb[1],
        y_min=xb[2],
        y_max=xb[3],
        effort=int(best_effort[k]),
        n_cells=int(n_cells[k]),
        cost=float(best_cost[k]),
        score=float(best_score[k]),
    )


# --------------------------------------------------------------------- #
#  Public strategies
# --------------------------------------------------------------------- #
DEFAULT_WIDTHS = (1, 2, 3, 4, 5, 6, 8)
DEFAULT_HEIGHTS = (1, 2, 3, 4, 5, 6, 8)
DEFAULT_EFFORTS = (1, 2, 3)


def propose_max_expected_detection(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    budget_remaining: float,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
    widths: Iterable[int] = DEFAULT_WIDTHS,
    heights: Iterable[int] = DEFAULT_HEIGHTS,
) -> MissionProposal:
    """Maximise expected detection per unit cost: argmax sum_j pi_j q_j / cost."""
    return _best_proposal(
        posterior, grid, model, effort_levels, budget_remaining, widths, heights,
        score_kind="expected_detection",
    )


def propose_info_gain(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    budget_remaining: float,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
    widths: Iterable[int] = DEFAULT_WIDTHS,
    heights: Iterable[int] = DEFAULT_HEIGHTS,
) -> MissionProposal:
    """Maximise expected entropy reduction per unit cost (one-step lookahead)."""
    return _best_proposal(
        posterior, grid, model, effort_levels, budget_remaining, widths, heights,
        score_kind="info_gain",
    )


def propose_max_posterior_rect(
    posterior: np.ndarray,
    grid: GridInfo,
    model: DetectionModel,
    budget_remaining: float,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
    widths: Iterable[int] = DEFAULT_WIDTHS,
    heights: Iterable[int] = DEFAULT_HEIGHTS,
) -> MissionProposal:
    """Maximise raw expected detection sum_j pi_j q_j (not per-cost).

    This is closer to "shrink-wrap the highest posterior mass and inspect
    it as well as possible" without explicitly rewarding small rectangles.
    """
    return _best_proposal(
        posterior, grid, model, effort_levels, budget_remaining, widths, heights,
        score_kind="posterior_mass_x_rho",
    )
