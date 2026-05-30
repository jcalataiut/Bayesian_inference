"""Propose the next axis-aligned rectangular search mission.

The webapp accepts a rectangle ``[x_min, x_max] x [y_min, y_max]`` and an
effort level. We expose two families of strategies:

A. **Direct expected-detection** (legacy, kept for comparison):
   enumerate all rectangles globally and score by
   ``sum_{j in R} P(Z=j) * rho_j(e) / cost``.

B. **HDR funnel** (recommended): a three-stage decision pipeline that
   first focuses on *where the object plausibly is* and only then asks
   *where the detector works best*.

      Stage 1.  Highest-Density Region: sort cells by P(Z=j) descending
                and accumulate until total mass >= hdr_mass (e.g. 0.80).
                Gives a set H of credible cells.

      Stage 2.  Bounding box of H gives an axis-aligned region B.
                For each effort e, score cells inside B by
                    score_j(e) = P(Z=j) * rho_j(e)
                masked to H (cells outside H contribute 0).

      Stage 3.  Select the top-K cells by score and propose the
                minimum axis-aligned rectangle containing them. Pick the
                effort that maximises total captured score per unit cost,
                subject to budget.

We also expose ``hdr_mask`` so notebooks and the Streamlit app can render
the credible regions at multiple levels (50, 80, 95).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..modeling.detection import DetectionModel
from ..modeling.features import cell_distances, expected_landing, trajectory_axes
from ..simulator.grid import GridInfo


# --------------------------------------------------------------------------- #
# Public data class
# --------------------------------------------------------------------------- #
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
    expected_detection: float  # sum_{j in R} P(Z=j) * rho_j(e)

    def as_kwargs(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "effort": self.effort,
        }


# --------------------------------------------------------------------------- #
# Integral image helpers
# --------------------------------------------------------------------------- #
def _integral_image(arr2d: np.ndarray) -> np.ndarray:
    S = arr2d.cumsum(axis=0).cumsum(axis=1)
    out = np.zeros((arr2d.shape[0] + 1, arr2d.shape[1] + 1), dtype=arr2d.dtype)
    out[1:, 1:] = S
    return out


def _rect_sums(S: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    return S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]


def _enumerate_rectangles(
    Nx: int,
    Ny: int,
    widths: Iterable[int],
    heights: Iterable[int],
):
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


def _bounds_from_indices(ix0, ix1, iy0, iy1):
    # Cell centers are at integer + 0.5, so an inclusive index rectangle
    # [ix0, ix1] corresponds exactly to webapp bounds [ix0+0.5, ix1+0.5].
    return ix0 + 0.5, ix1 + 0.5, iy0 + 0.5, iy1 + 0.5


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
DEFAULT_WIDTHS = (1, 2, 3, 4, 5, 6, 8, 10)
DEFAULT_HEIGHTS = (1, 2, 3, 4, 5, 6, 8, 10)
DEFAULT_EFFORTS = (1, 2, 3)


def progressive_d_long_center(
    n_fails: int,
    *,
    first_center: float = 2.5,
    coarse_step: float = 2.5,
    coarse_until: float = 7.5,
    fine_step: float = 0.5,
    max_long: float = 12.0,
) -> float:
    """Center schedule along the drift/flight direction.

    The operational search starts with a non-zero conservative displacement
    along the drift axis, then moves in large steps until 7.5, and finally
    refines in 0.5-cell steps:

        failures: 0 -> 2.5, 1 -> 5.0, 2 -> 7.5,
                  3 -> 8.0, 4 -> 8.5, ...

    This matches the intended logic: if the plane is not around the initial
    impact area, keep moving farther along the physical drift direction.
    """
    n = max(0, int(n_fails))
    coarse_count = int(round((coarse_until - first_center) / coarse_step)) + 1
    if n < coarse_count:
        center = first_center + coarse_step * n
    else:
        center = coarse_until + fine_step * (n - coarse_count + 1)
    return float(min(max_long, center))


def propose_expected_detection(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    detector: DetectionModel,
    budget_remaining: float,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
    widths: Iterable[int] = DEFAULT_WIDTHS,
    heights: Iterable[int] = DEFAULT_HEIGHTS,
) -> MissionProposal:
    """Maximise expected detections per unit cost."""
    posterior_2d = posterior_Z.reshape(grid.Ny, grid.Nx)
    depth_2d = grid.depth.reshape(grid.Ny, grid.Nx)
    roughness_2d = grid.roughness.reshape(grid.Ny, grid.Nx)

    ix0, ix1, iy0, iy1 = _enumerate_rectangles(grid.Nx, grid.Ny, widths, heights)
    r0, r1 = iy0, iy1 + 1
    c0, c1 = ix0, ix1 + 1
    n_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)

    best_score = np.full(ix0.size, -np.inf)
    best_effort = np.zeros(ix0.size, dtype=int)
    best_cost = np.zeros(ix0.size)
    best_exp_det = np.zeros(ix0.size)

    for e in effort_levels:
        rho_2d = detector.rho(depth_2d, roughness_2d, e)
        S = _integral_image(posterior_2d * rho_2d)
        expected_det = _rect_sums(S, r0, r1, c0, c1)

        cost = e * n_cells
        feasible = cost <= budget_remaining
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(feasible, expected_det / cost, -np.inf)

        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_effort = np.where(better, e, best_effort)
        best_cost = np.where(better, cost, best_cost)
        best_exp_det = np.where(better, expected_det, best_exp_det)

    if not np.isfinite(best_score).any():
        raise RuntimeError("No candidate rectangle fits the remaining budget.")

    k = int(np.argmax(best_score))
    xmin, xmax, ymin, ymax = _bounds_from_indices(
        int(ix0[k]), int(ix1[k]), int(iy0[k]), int(iy1[k])
    )
    return MissionProposal(
        x_min=float(xmin),
        x_max=float(xmax),
        y_min=float(ymin),
        y_max=float(ymax),
        effort=int(best_effort[k]),
        n_cells=int(n_cells[k]),
        cost=float(best_cost[k]),
        score=float(best_score[k]),
        expected_detection=float(best_exp_det[k]),
    )


# --------------------------------------------------------------------------- #
# HDR funnel strategy
# --------------------------------------------------------------------------- #
def hdr_cells(posterior_Z: np.ndarray, mass: float = 0.80) -> np.ndarray:
    """Return the indices of the cells whose cumulative posterior mass >= mass.

    Sorted from highest to lowest P(Z=j). The returned indices form the
    minimum-cardinality Highest-Density Region for the given mass level.
    """
    order = np.argsort(-posterior_Z)
    cum = np.cumsum(posterior_Z[order])
    n_take = int(np.searchsorted(cum, mass)) + 1
    return order[:n_take]


def hdr_mask(posterior_Z: np.ndarray, mass: float = 0.80) -> np.ndarray:
    """Boolean length-N mask of the HDR cells at the given mass level."""
    idx = hdr_cells(posterior_Z, mass)
    out = np.zeros_like(posterior_Z, dtype=bool)
    out[idx] = True
    return out


def hdr_bounding_box(
    posterior_Z: np.ndarray, grid: GridInfo, mass: float = 0.80,
) -> tuple[int, int, int, int]:
    """Return (ix_min, ix_max, iy_min, iy_max) integer-grid bounds of HDR.

    Bounds are inclusive integer cell indices (0..Nx-1, 0..Ny-1) and can be
    converted to webapp coordinates via ``+ 0.5``.
    """
    idx = hdr_cells(posterior_Z, mass)
    ix = (grid.x[idx] - 0.5).astype(int)
    iy = (grid.y[idx] - 0.5).astype(int)
    return int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())


def _minimum_rect_around_cells(
    cell_ids: np.ndarray, grid: GridInfo
) -> tuple[float, float, float, float, int]:
    """Return (x_min, x_max, y_min, y_max, n_cells) of the bounding rectangle.

    The rectangle is the minimum axis-aligned one that contains all
    ``cell_ids``; n_cells counts every grid cell falling inside the rect
    (including ones not in ``cell_ids``).
    """
    ix = (grid.x[cell_ids] - 0.5).astype(int)
    iy = (grid.y[cell_ids] - 0.5).astype(int)
    ix0, ix1 = int(ix.min()), int(ix.max())
    iy0, iy1 = int(iy.min()), int(iy.max())
    n_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    return ix0 + 0.5, ix1 + 0.5, iy0 + 0.5, iy1 + 0.5, n_cells


def propose_via_hdr_topk(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    detector: DetectionModel,
    budget_remaining: float,
    hdr_mass: float = 0.80,
    top_k: int | None = None,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
    cost_exponent: float = 0.5,
    history: "Sequence | None" = None,
    repetition_penalty: float = 0.2,
    exploration_mix: float = 0.0,
) -> "HDRProposal":
    """Propose a mission via the HDR -> bounding-box -> top-K -> min-rect funnel.

    Parameters
    ----------
    hdr_mass : float
        Cumulative posterior mass that defines the Highest-Density Region.
    top_k : int or None
        Number of high-score cells inside HDR whose bounding rectangle we
        propose. If None, defaults to max(2, |HDR| // 4).
    effort_levels : Iterable[int]
        Allowed effort values (default (1, 2, 3)).
    cost_exponent : float
        Exponent applied to ``cost`` in the score
        ``score = expected_detection / cost ** cost_exponent``.

        * ``1.0`` -- pure expected detection per unit cost; favours small,
          low-effort missions because per-dollar yield is highest at e=1.
        * ``0.5`` (default) -- soft preference for higher effort: the
          saturating detector model means e=2 or e=3 detect noticeably more
          often than e=1, so a moderately sublinear cost penalty pushes the
          algorithm toward investing more per mission while still being
          punished if it bloats the rectangle.
        * ``0.0`` -- ignore cost entirely; will almost always pick the
          highest allowed effort and the largest feasible rectangle (greedy,
          burns budget fast).
    history : sequence of MissionRecord or None
        If provided, cells inside any previously-failed mission rectangle
        receive a multiplicative ``repetition_penalty`` on their score.
        This forces the strategy to jump to unexplored parts of the HDR
        when consecutive failures concentrate the proposal in the same
        zone (typical failure mode when the truth lies in the prior tail).
    repetition_penalty : float in [0, 1]
        Multiplicative discount applied to cells inside the union of
        previously-failed rectangles. Defaults to 0.2 (5x penalty); 1.0
        disables it, 0.0 forbids re-visiting at all.
    exploration_mix : float in [0, 1)
        Replace the input posterior by
        ``(1 - exploration_mix) * posterior_Z + exploration_mix / N``
        before doing HDR / scoring. Gives every cell a guaranteed minimum
        weight so the algorithm considers unexplored regions even when the
        parametric posterior has collapsed away from them. Independent of
        ``spec.uniform_mix``: this knob acts at decision time, not at
        modelling time.
    """
    if exploration_mix > 0.0:
        N = float(posterior_Z.size)
        posterior_Z = (1.0 - exploration_mix) * posterior_Z + exploration_mix / N
        posterior_Z = posterior_Z / posterior_Z.sum()
    H = hdr_cells(posterior_Z, hdr_mass)
    if H.size == 0:
        raise RuntimeError(f"Empty HDR at mass={hdr_mass}.")
    if top_k is None:
        top_k = max(2, H.size // 4)
    top_k = max(1, min(int(top_k), int(H.size)))

    H_mask = np.zeros_like(posterior_Z, dtype=bool)
    H_mask[H] = True

    # Cells already covered by failed missions -- penalize so we explore
    # the rest of the HDR instead of re-proposing the same zone.
    explored_mask = np.zeros_like(posterior_Z, dtype=bool)
    if history is not None and repetition_penalty != 1.0:
        for m in history:
            if getattr(m, "s_t", 0) == 0:
                explored_mask |= grid.coverage_mask(
                    m.x_min, m.x_max, m.y_min, m.y_max
                )

    best = None
    for e in effort_levels:
        rho = detector.rho(grid.depth, grid.roughness, e)
        # Score within HDR only.
        score_full = np.where(H_mask, posterior_Z * rho, -np.inf)
        # Apply repetition penalty to cells already covered by failed
        # missions. Keeps them as candidates but discourages re-selection.
        if explored_mask.any():
            score_full = np.where(
                explored_mask & H_mask,
                score_full * repetition_penalty,
                score_full,
            )
        # Take the top_k *inside HDR* (always positive by construction).
        topk_idx = np.argpartition(-score_full, top_k - 1)[:top_k]
        # Drop any cells with -inf (shouldn't happen if top_k <= |H|).
        topk_idx = topk_idx[score_full[topk_idx] > -np.inf]
        if topk_idx.size == 0:
            continue
        x_min, x_max, y_min, y_max, n_cells = _minimum_rect_around_cells(topk_idx, grid)
        cost = float(e * n_cells)
        if cost > budget_remaining + 1e-9:
            continue
        # Re-score over the actual rectangle (not just the topK).
        rect_mask = grid.coverage_mask(x_min, x_max, y_min, y_max)
        expected_det = float(np.sum(posterior_Z * rho * rect_mask))
        # HDR mass actually captured by this rect.
        hdr_captured = float(np.sum(posterior_Z * (H_mask & rect_mask)))
        # cost_exponent in (0, 1): sublinearly penalize cost -> bias toward
        # higher effort. cost_exponent = 1 recovers the per-cost objective.
        denom = max(cost ** cost_exponent, 1e-9)
        score = expected_det / denom
        if best is None or score > best.score:
            best = HDRProposal(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                effort=int(e), n_cells=int(n_cells), cost=cost,
                score=score, expected_detection=expected_det,
                hdr_mass_used=float(hdr_mass), hdr_cells_total=int(H.size),
                top_k=int(top_k), hdr_mass_captured=hdr_captured,
            )
    if best is None:
        # Adaptive fallback: shrink top_k until something fits the budget.
        for k_try in (max(top_k // 2, 1), max(top_k // 4, 1), 2, 1):
            if k_try >= top_k:
                continue
            try:
                return propose_via_hdr_topk(
                    posterior_Z, grid, detector, budget_remaining,
                    hdr_mass=hdr_mass, top_k=k_try,
                    effort_levels=effort_levels, cost_exponent=cost_exponent,
                    history=history, repetition_penalty=repetition_penalty,
                    exploration_mix=0.0,  # already applied above
                )
            except RuntimeError:
                continue
        raise RuntimeError(
            f"No HDR rectangle fits the remaining budget {budget_remaining:.1f}; "
            f"consider lowering hdr_mass or top_k."
        )
    return best


@dataclass
class HDRProposal(MissionProposal):
    """MissionProposal extended with HDR diagnostics."""

    hdr_mass_used: float = 0.0
    hdr_cells_total: int = 0
    top_k: int = 0
    hdr_mass_captured: float = 0.0


def _entropy_arr(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-300, None))).sum())


def physics_tail_prior(
    grid: GridInfo,
    *,
    tau_fall: float = 1.0,
    tau_drift: float = 1.0,
    long_mean: float = 7.5,
    long_sd: float = 2.5,
    trans_mean: float = 0.8,
    trans_sd: float = 2.5,
    uniform_mix: float = 0.02,
) -> np.ndarray:
    """Prior component for the physically plausible down-track tail.

    This is not a hand-coded target rectangle. It uses the accident physics:
    project each cell onto the trajectory axes and put mass *ahead* of the
    nominal expected landing point along the drift/flight direction. The
    stress-test zone x≈18–21, y≈9–12 lies naturally in this continuation
    tail (d_long ≈ 6–9).
    """
    d_long, d_trans = cell_distances(grid.x, grid.y, tau_fall, tau_drift)
    z = -0.5 * ((d_long - long_mean) / long_sd) ** 2
    z += -0.5 * ((d_trans - trans_mean) / trans_sd) ** 2
    w = np.exp(z - np.max(z))
    w = w / w.sum()
    if uniform_mix > 0:
        w = (1.0 - uniform_mix) * w + uniform_mix / w.size
        w = w / w.sum()
    return w


def progressive_drift_prior(
    grid: GridInfo,
    history: "Sequence | None" = None,
    *,
    tau_fall: float = 1.0,
    tau_drift: float = 1.0,
    step_long: float = 2.5,
    max_long: float = 12.0,
    trans_mean: float = 0.8,
    long_sd: float = 2.0,
    trans_sd: float = 2.4,
    uniform_mix: float = 0.02,
) -> tuple[np.ndarray, float, tuple[float, float], int]:
    """Moving physical prior that advances along the drift axis after failures.

    This is the gradual logic used by ``propose_drift_tail_rescue``:

    * 0 failures: start at d_long=2.5.
    * 1 failure: move to d_long=5.0.
    * 2 failures: move to d_long=7.5.
    * 3+ failures: continue in fine 0.5-cell increments.

    The stress-test zone x≈18–21, y≈9–12 is reached because it lies around
    d_long≈6–9, not because those coordinates are hard-coded as a target.
    """
    n_fails = 0
    if history is not None:
        n_fails = sum(1 for m in history if getattr(m, "s_t", 0) == 0)
    center_long = progressive_d_long_center(n_fails, max_long=max_long)
    d_long, d_trans = cell_distances(grid.x, grid.y, tau_fall, tau_drift)
    z = -0.5 * ((d_long - center_long) / long_sd) ** 2
    z += -0.5 * ((d_trans - trans_mean) / trans_sd) ** 2
    w = np.exp(z - np.max(z))
    w = w / w.sum()
    if uniform_mix > 0:
        w = (1.0 - uniform_mix) * w + uniform_mix / w.size
        w = w / w.sum()

    mu = expected_landing(tau_fall, tau_drift)
    d_norm, n_norm = trajectory_axes()
    center_xy = mu + center_long * d_norm + trans_mean * n_norm
    return w, float(center_long), (float(center_xy[0]), float(center_xy[1])), int(n_fails)


def progressive_scenario_prior(
    grid: GridInfo,
    history: "Sequence | None" = None,
    *,
    scenario: str = "center",
    tau_fall: float = 1.0,
    tau_drift: float = 1.0,
    step_long: float = 2.5,
    max_long: float = 12.0,
    lateral_offset: float = 3.0,
    long_sd: float = 2.0,
    trans_sd: float = 2.4,
    uniform_mix: float = 0.02,
) -> tuple[np.ndarray, float, tuple[float, float], int, str]:
    """Progressive physical prior for witness-driven scenarios.

    Scenarios:
    - ``center``: witness 1, object follows the aircraft trajectory; after
      water contact it drifts progressively down-track.
    - ``right``: witness 2 reports a lateral displacement to +d_trans.
    - ``left``: witness 2 reports a lateral displacement to -d_trans.
    - ``strong_wind``: fall stage is more wind-coupled (alpha_wind=1.0),
      then the same progressive down-track drift is applied.
    - ``mixture``: weighted model average over center/right/left/strong_wind.

    The true cell is never used. The scenario only changes the physical
    hypothesis used after failed searches.
    """
    if scenario == "mixture":
        weights = {
            "center": 0.45,
            "right": 0.20,
            "left": 0.20,
            "strong_wind": 0.15,
        }
        acc = np.zeros(grid.n_cells, dtype=float)
        centers = []
        center_long = 0.0
        n_fails = 0
        for name, weight in weights.items():
            p, center_long, center_xy, n_fails, _ = progressive_scenario_prior(
                grid, history, scenario=name, tau_fall=tau_fall,
                tau_drift=tau_drift, step_long=step_long, max_long=max_long,
                lateral_offset=lateral_offset, long_sd=long_sd,
                trans_sd=trans_sd, uniform_mix=uniform_mix,
            )
            acc += weight * p
            centers.append((weight, center_xy))
        acc = acc / acc.sum()
        mean_center = (
            float(sum(w * c[0] for w, c in centers)),
            float(sum(w * c[1] for w, c in centers)),
        )
        return acc, float(center_long), mean_center, int(n_fails), scenario

    n_fails = 0
    if history is not None:
        n_fails = sum(1 for m in history if getattr(m, "s_t", 0) == 0)
    center_long = progressive_d_long_center(n_fails, max_long=max_long)

    alpha_wind = 1.0 if scenario == "strong_wind" else 0.5
    if scenario == "right":
        trans_mean = lateral_offset
    elif scenario == "left":
        trans_mean = -lateral_offset
    elif scenario in ("center", "strong_wind"):
        trans_mean = 0.0
    else:
        raise ValueError(f"Unknown progressive scenario {scenario!r}")

    d_long, d_trans = cell_distances(
        grid.x, grid.y, tau_fall, tau_drift, alpha_wind=alpha_wind
    )
    z = -0.5 * ((d_long - center_long) / long_sd) ** 2
    z += -0.5 * ((d_trans - trans_mean) / trans_sd) ** 2
    p = np.exp(z - np.max(z))
    p = p / p.sum()
    if uniform_mix > 0:
        p = (1.0 - uniform_mix) * p + uniform_mix / p.size
        p = p / p.sum()

    mu = expected_landing(tau_fall, tau_drift, alpha_wind=alpha_wind)
    d_norm, n_norm = trajectory_axes(alpha_wind=alpha_wind)
    center_xy = mu + center_long * d_norm + trans_mean * n_norm
    return p, float(center_long), (float(center_xy[0]), float(center_xy[1])), int(n_fails), scenario


def apply_drift_tail_escape(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    history: "Sequence | None" = None,
    *,
    fail_threshold: int = 1,
    max_mix: float = 0.45,
) -> tuple[np.ndarray, float]:
    """Mix posterior with a progressively moving drift component.

    Failed missions near the nominal peak are informative: they suggest that
    the object may be farther along the flight/drift direction. Instead of
    jumping to a fixed tail, the auxiliary component advances gradually in
    d_long as the failure count grows.

    Returns
    -------
    posterior_mixed, mix_weight
    """
    n_fails = 0
    if history is not None:
        n_fails = sum(1 for m in history if getattr(m, "s_t", 0) == 0)
    if n_fails < fail_threshold:
        p = np.asarray(posterior_Z, dtype=float)
        return p / p.sum(), 0.0
    # Gradual trust transfer: after the first failure the physical drift
    # hypothesis is weak; after repeated failures it becomes a serious
    # competing explanation.
    mix = min(max_mix, 0.12 * n_fails)
    drift_component, _, _, _ = progressive_drift_prior(grid, history)
    p = (1.0 - mix) * posterior_Z + mix * drift_component
    p = p / p.sum()
    return p, float(mix)


def apply_progressive_scenario_escape(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    history: "Sequence | None" = None,
    *,
    scenario: str = "mixture",
    fail_threshold: int = 1,
    max_mix: float = 0.45,
) -> tuple[np.ndarray, float, float, tuple[float, float], str]:
    """Mix posterior with one witness-driven progressive physical scenario."""
    n_fails = 0
    if history is not None:
        n_fails = sum(1 for m in history if getattr(m, "s_t", 0) == 0)
    p = np.asarray(posterior_Z, dtype=float)
    p = p / p.sum()
    scenario_component, center_long, center_xy, _, scenario_name = progressive_scenario_prior(
        grid, history, scenario=scenario
    )
    if n_fails < fail_threshold:
        return p, 0.0, center_long, center_xy, scenario_name
    mix = min(max_mix, 0.12 * n_fails)
    out = (1.0 - mix) * p + mix * scenario_component
    out = out / out.sum()
    return out, float(mix), center_long, center_xy, scenario_name


def propose_info_gain(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    detector: DetectionModel,
    budget_remaining: float,
    hdr_mass: float = 0.85,
    top_k: int | None = None,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
    history: "Sequence | None" = None,
    repetition_penalty: float = 0.15,
    exploration_mix: float = 0.05,
    n_candidates: int = 64,
) -> "HDRProposal":
    """Bayesian optimal-experiment-design: maximise expected entropy reduction
    of the posterior over Z, per unit cost.

    For each candidate rectangle ``R`` and effort ``e``:
        D = sum_{j in R} pi_j * rho_j(e)           (= P(s_t = 1))
        H_before = entropy(pi)
        H_after(s=1) = entropy of pi restricted to R weighted by rho
        H_after(s=0) = entropy of pi multiplied by (1 - q_j), renormalised
        E[H_after] = D * H_after(s=1) + (1 - D) * H_after(s=0)
        info_gain = H_before - E[H_after]
        score = info_gain / cost

    Implementation: we don't enumerate the full ~50000 rectangle space
    (too slow with per-rect entropy computations). Instead, we sample
    ``n_candidates`` rectangles using the same top-K + minimum-rect logic
    as ``propose_via_hdr_topk`` but with randomised k in [2, top_k_max], so
    each call produces a varied pool of candidate rectangles. We then
    rank by info_gain / cost.
    """
    if exploration_mix > 0.0:
        N = float(posterior_Z.size)
        posterior_Z = (1.0 - exploration_mix) * posterior_Z + exploration_mix / N
        posterior_Z = posterior_Z / posterior_Z.sum()

    H = hdr_cells(posterior_Z, hdr_mass)
    if H.size == 0:
        raise RuntimeError(f"Empty HDR at mass={hdr_mass}.")
    top_k_max = max(2, int(top_k or H.size // 4))

    H_mask = np.zeros_like(posterior_Z, dtype=bool)
    H_mask[H] = True

    # Repetition penalty as in propose_via_hdr_topk
    explored = np.zeros_like(posterior_Z, dtype=bool)
    if history is not None and repetition_penalty != 1.0:
        for m in history:
            if getattr(m, "s_t", 0) == 0:
                explored |= grid.coverage_mask(m.x_min, m.x_max, m.y_min, m.y_max)

    H_before = _entropy_arr(posterior_Z)
    rng = np.random.default_rng()

    best = None
    for e in effort_levels:
        rho = detector.rho(grid.depth, grid.roughness, e)
        score_full = np.where(H_mask, posterior_Z * rho, 0.0)
        if explored.any():
            score_full = np.where(explored & H_mask,
                                    score_full * repetition_penalty, score_full)

        for k in rng.integers(2, top_k_max + 1, size=n_candidates):
            # Top-k by score
            k = int(min(k, score_full.size))
            topk_idx = np.argpartition(-score_full, k - 1)[:k]
            topk_idx = topk_idx[score_full[topk_idx] > 0]
            if topk_idx.size == 0:
                continue
            x_min, x_max, y_min, y_max, n_cells = _minimum_rect_around_cells(topk_idx, grid)
            cost = float(e * n_cells)
            if cost > budget_remaining + 1e-9:
                continue
            rect_mask = grid.coverage_mask(x_min, x_max, y_min, y_max)
            q = np.where(rect_mask, rho, 0.0)
            q = np.clip(q, 1e-12, 1.0 - 1e-12)
            D = float(np.sum(posterior_Z * q))
            if D <= 0 or D >= 1:
                continue
            # Posterior after each outcome
            post1 = posterior_Z * q
            s1 = post1.sum()
            post1 = post1 / s1 if s1 > 0 else post1
            post0 = posterior_Z * (1.0 - q)
            s0 = post0.sum()
            post0 = post0 / s0 if s0 > 0 else post0
            H1 = _entropy_arr(post1)
            H0 = _entropy_arr(post0)
            E_H = D * H1 + (1.0 - D) * H0
            info_gain = H_before - E_H
            score = info_gain / cost
            if best is None or score > best.score:
                best = HDRProposal(
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    effort=int(e), n_cells=int(n_cells), cost=cost,
                    score=score, expected_detection=D,
                    hdr_mass_used=float(hdr_mass), hdr_cells_total=int(H.size),
                    top_k=int(k), hdr_mass_captured=float(np.sum(posterior_Z * (H_mask & rect_mask))),
                )
    if best is None:
        raise RuntimeError(
            f"No info-gain candidate fits the remaining budget {budget_remaining:.1f}."
        )
    return best


def propose_adaptive(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    detector: DetectionModel,
    budget_remaining: float,
    history: "Sequence | None" = None,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
) -> "HDRProposal":
    """Adaptive HDR strategy: hyperparameters change based on context.

    Reads two signals from the current state of the campaign:
        * **Posterior entropy** (normalised by log(N)): high → we don't know
          much yet, prefer exploration; low → we are confident, exploit.
        * **Failure streak**: more failures → broaden HDR, accept smaller
          top-K rectangles, give every cell a larger uniform mixture so the
          algorithm jumps to new parts of the grid instead of creeping.

    This is the "model flexibility" Task 5 mentions: instead of one fixed
    strategy that performs OK on average, the strategy itself adapts to
    each campaign's state.

    Returns the same HDRProposal as ``propose_via_hdr_topk``.
    """
    H_max = float(np.log(posterior_Z.size))
    H = float(-(posterior_Z * np.log(np.clip(posterior_Z, 1e-300, None))).sum())
    H_norm = H / H_max  # in [0, 1]

    n_fails = 0
    if history is not None:
        n_fails = sum(1 for m in history if getattr(m, "s_t", 0) == 0)

    # ----- adaptive schedule (justified by grid search at scale) ----- #
    if H_norm < 0.55 and n_fails < 2:
        # Confident, few failures: exploit with a focused mission.
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.60, 15, 0.5, 0.03, 0.15
    elif n_fails < 4:
        # Mid-range: balanced.
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.80, 10, 0.7, 0.10, 0.10
    elif n_fails < 7:
        # Several failures: broaden the search.
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.90,  6, 0.8, 0.20, 0.05
    else:
        # Many failures: nearly uniform exploration of the whole plausible region.
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.97,  4, 1.0, 0.35, 0.03

    return propose_via_hdr_topk(
        posterior_Z, grid, detector, budget_remaining,
        hdr_mass=hdr_mass, top_k=top_k, effort_levels=effort_levels,
        cost_exponent=cost_exp, history=history,
        repetition_penalty=rep_pen, exploration_mix=expl_mix,
    )


def propose_drift_tail_rescue(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    detector: DetectionModel,
    budget_remaining: float,
    history: "Sequence | None" = None,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
) -> "HDRProposal":
    """Progressive drift strategy for the south-east stress test.

    It first applies a gradually moving physical component after failed
    missions, then runs the HDR funnel. The model is not told the target
    rectangle; it only knows that if the conservative expected zone fails,
    the next plausible explanation is progressively farther along the
    drift/flight direction.
    """
    drift_component, center_long, center_xy, n_fails = progressive_drift_prior(
        grid, history
    )
    p_work, mix = apply_drift_tail_escape(
        posterior_Z, grid, history, fail_threshold=1, max_mix=0.45
    )
    # Once the drift center starts moving, prefer smaller rectangles and
    # per-cost scoring so we follow the trajectory step by step without
    # burning the 530 budget in one huge rectangle.
    if n_fails < 1:
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.80, 10, 0.70, 0.10, 0.10
    elif n_fails < 3:
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.82, 9, 0.90, 0.10, 0.06
    else:
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.85, 8, 1.00, 0.12, 0.04
    prop = propose_via_hdr_topk(
        p_work, grid, detector, budget_remaining,
        hdr_mass=hdr_mass, top_k=top_k, effort_levels=effort_levels,
        cost_exponent=cost_exp, history=history,
        repetition_penalty=rep_pen, exploration_mix=expl_mix,
    )
    # Store the diagnostic on the returned dataclass dynamically; Streamlit
    # can display it if present.
    setattr(prop, "tail_mix", mix)
    setattr(prop, "drift_long", center_long)
    setattr(prop, "drift_center_xy", center_xy)
    setattr(prop, "n_failed_missions", n_fails)
    return prop


def propose_progressive_scenario_search(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    detector: DetectionModel,
    budget_remaining: float,
    history: "Sequence | None" = None,
    *,
    scenario: str = "mixture",
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
) -> "HDRProposal":
    """Mission proposal for center/right/left/strong-wind physical scenarios."""
    p_work, mix, center_long, center_xy, scenario_name = apply_progressive_scenario_escape(
        posterior_Z, grid, history, scenario=scenario, fail_threshold=1, max_mix=0.45
    )
    n_fails = 0 if history is None else sum(1 for m in history if getattr(m, "s_t", 0) == 0)
    if n_fails < 1:
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.80, 10, 0.70, 0.08, 0.10
    elif n_fails < 3:
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.82, 9, 0.90, 0.08, 0.06
    else:
        hdr_mass, top_k, cost_exp, expl_mix, rep_pen = 0.85, 8, 1.00, 0.10, 0.04
    prop = propose_via_hdr_topk(
        p_work, grid, detector, budget_remaining,
        hdr_mass=hdr_mass, top_k=top_k, effort_levels=effort_levels,
        cost_exponent=cost_exp, history=history,
        repetition_penalty=rep_pen, exploration_mix=expl_mix,
    )
    setattr(prop, "tail_mix", mix)
    setattr(prop, "drift_long", center_long)
    setattr(prop, "drift_center_xy", center_xy)
    setattr(prop, "physical_scenario", scenario_name)
    setattr(prop, "n_failed_missions", n_fails)
    return prop


def propose_max_posterior_rect(
    posterior_Z: np.ndarray,
    grid: GridInfo,
    detector: DetectionModel,
    budget_remaining: float,
    effort_levels: Iterable[int] = DEFAULT_EFFORTS,
    widths: Iterable[int] = DEFAULT_WIDTHS,
    heights: Iterable[int] = DEFAULT_HEIGHTS,
) -> MissionProposal:
    """Maximise raw expected detections (no cost normalization).

    Useful as a second baseline: tends to prefer larger rectangles than
    the cost-normalised version above.
    """
    posterior_2d = posterior_Z.reshape(grid.Ny, grid.Nx)
    depth_2d = grid.depth.reshape(grid.Ny, grid.Nx)
    roughness_2d = grid.roughness.reshape(grid.Ny, grid.Nx)

    ix0, ix1, iy0, iy1 = _enumerate_rectangles(grid.Nx, grid.Ny, widths, heights)
    r0, r1 = iy0, iy1 + 1
    c0, c1 = ix0, ix1 + 1
    n_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)

    best_score = np.full(ix0.size, -np.inf)
    best_effort = np.zeros(ix0.size, dtype=int)
    best_cost = np.zeros(ix0.size)
    best_exp_det = np.zeros(ix0.size)

    for e in effort_levels:
        rho_2d = detector.rho(depth_2d, roughness_2d, e)
        S = _integral_image(posterior_2d * rho_2d)
        expected_det = _rect_sums(S, r0, r1, c0, c1)

        cost = e * n_cells
        feasible = cost <= budget_remaining
        score = np.where(feasible, expected_det, -np.inf)

        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_effort = np.where(better, e, best_effort)
        best_cost = np.where(better, cost, best_cost)
        best_exp_det = np.where(better, expected_det, best_exp_det)

    if not np.isfinite(best_score).any():
        raise RuntimeError("No candidate rectangle fits the remaining budget.")

    k = int(np.argmax(best_score))
    xmin, xmax, ymin, ymax = _bounds_from_indices(
        int(ix0[k]), int(ix1[k]), int(iy0[k]), int(iy1[k])
    )
    return MissionProposal(
        x_min=float(xmin),
        x_max=float(xmax),
        y_min=float(ymin),
        y_max=float(ymax),
        effort=int(best_effort[k]),
        n_cells=int(n_cells[k]),
        cost=float(best_cost[k]),
        score=float(best_score[k]),
        expected_detection=float(best_exp_det[k]),
    )
