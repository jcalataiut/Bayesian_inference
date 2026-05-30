"""Fast Monte Carlo evaluation over many plausible bomb scenarios.

Why fast: full NUTS per mission is prohibitively expensive at 1000 trials
(weeks of compute). Instead we use the **direct per-cell posterior update**
which is the closed-form solution for a fixed parametric family — the prior
predictive distribution treated as a pointwise prior over Z, multiplied by
the per-cell likelihood after each mission:

    pi_post(j) ∝ pi_prior(j) * prod_t  q_{t,j}^{s_t} (1-q_{t,j})^{1-s_t}

This drops the "learning about beta" component of the full hierarchical
model but keeps the cell-level Bayesian update intact. It is the right tool
for ranking priors by how fast they converge on the true cell across many
physically-plausible scenarios.

The bomb is planted by sampling ``true_cell ~ Categorical(reference_prior)``
so the test pool is concentrated on scenarios that are consistent with the
accident's physical inputs.

Usage:

    python -m update_version.experiments.fast_monte_carlo --n-trials 1000
"""
from __future__ import annotations

import argparse
import zlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..modeling.detection import DETECTORS, DetectionModel
from ..modeling.priors_logistic import PRIORS, PriorSpec, make_grid_priors
from ..simulator.grid import GridInfo, load_grid
from ..simulator.true_detection import TrueDetector
from ..strategies.next_mission import propose_via_hdr_topk


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class CampaignResult:
    detected: bool
    n_missions: int
    cost_used: float
    final_p_true: float
    final_entropy: float
    p_true_evolution: list[float]
    entropy_evolution: list[float]


def _entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-300, None))).sum())


def fast_campaign(
    prior_pred: np.ndarray,
    grid: GridInfo,
    detector_model: DetectionModel,
    true_detector: TrueDetector,
    true_cell: int,
    budget: float = 230.0,
    *,
    hdr_mass: float = 0.80,
    top_k: int = 10,
    cost_exponent: float = 0.7,
    repetition_penalty: float = 0.15,
    exploration_mix: float = 0.05,
    update_temperature: float = 1.0,
    max_missions: int = 10,
    seed: int = 0,
) -> CampaignResult:
    """Run one simulated campaign WITHOUT MCMC, using the direct cell-level
    Bayesian update.

    The strategy honours `exploration_mix` so a small uniform component
    keeps every cell as a viable candidate even after several failed
    missions.
    """
    rng = np.random.default_rng(seed)
    current = prior_pred / prior_pred.sum()
    p_true_evol = [float(current[true_cell])]
    entropy_evol = [_entropy(current)]
    budget_used = 0.0
    history: list = []
    detected = False

    @dataclass
    class _Rec:
        mission_id: int
        x_min: float; x_max: float
        y_min: float; y_max: float
        effort: int
        s_t: int

    mid = 0
    for _ in range(max_missions):
        if budget - budget_used < 4:
            break
        try:
            prop = propose_via_hdr_topk(
                current, grid, detector_model, budget - budget_used,
                hdr_mass=hdr_mass, top_k=top_k, cost_exponent=cost_exponent,
                history=history, repetition_penalty=repetition_penalty,
                exploration_mix=exploration_mix,
            )
        except RuntimeError:
            break
        mid += 1
        # Simulate detection if the (true) cell falls inside the rectangle.
        covered = grid.coverage_mask(prop.x_min, prop.x_max, prop.y_min, prop.y_max)
        if covered[true_cell]:
            rho_true = float(true_detector.rho(
                grid.depth[true_cell], grid.roughness[true_cell], prop.effort
            ))
            s_t = int(rng.random() < rho_true)
        else:
            s_t = 0
        history.append(_Rec(mid, prop.x_min, prop.x_max, prop.y_min, prop.y_max,
                             prop.effort, s_t))
        budget_used += prop.cost
        if s_t == 1:
            detected = True
            p_true_evol.append(p_true_evol[-1])
            entropy_evol.append(entropy_evol[-1])
            break
        # Direct cell-level Bayesian update with the modelling detector.
        rho_model = np.asarray(detector_model.rho(
            grid.depth, grid.roughness, prop.effort
        ))
        q = np.where(covered, rho_model, 0.0)
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        L = (1.0 - q) ** update_temperature  # since s_t == 0
        current = current * L
        s = current.sum()
        if s <= 0:
            current = np.full_like(current, 1.0 / current.size)
        else:
            current = current / s
        p_true_evol.append(float(current[true_cell]))
        entropy_evol.append(_entropy(current))

    return CampaignResult(
        detected=detected,
        n_missions=len(history),
        cost_used=float(budget_used),
        final_p_true=float(current[true_cell]),
        final_entropy=_entropy(current),
        p_true_evolution=p_true_evol,
        entropy_evolution=entropy_evol,
    )


def laplace_campaign(
    spec,
    grid: GridInfo,
    detector_model: DetectionModel,
    true_detector: TrueDetector,
    true_cell: int,
    budget: float = 230.0,
    *,
    hdr_mass: float = 0.80,
    top_k: int = 10,
    cost_exponent: float = 0.7,
    repetition_penalty: float = 0.15,
    exploration_mix: float = 0.05,
    max_missions: int = 10,
    laplace_n_samples: int = 60,
    seed: int = 0,
) -> CampaignResult:
    """Same as ``fast_campaign`` but updates pi via a Laplace approximation
    of the posterior on (beta, tau) after every mission.

    "INLA-style" backend: ~50x slower than the cell-level update but it
    DOES learn the latent physics (β posterior) between missions instead
    of merely reweighting cells. For our 7-dim posterior the Laplace
    approximation is essentially what INLA's simplified-Laplace strategy
    would yield.

    Note ``spec`` here MUST be a ``PriorSpec`` (the empirical KDE prior
    is not parametric and has no β to update with Laplace).
    """
    from ..modeling.laplace import laplace_approx, laplace_pi_mean
    rng = np.random.default_rng(seed)

    @dataclass
    class _Rec:
        mission_id: int
        x_min: float; x_max: float
        y_min: float; y_max: float
        effort: int
        s_t: int

    current = spec.prior_predictive_pi(grid.x, grid.y, n_samples=500, seed=0)
    p_true_evol = [float(current[true_cell])]
    entropy_evol = [_entropy(current)]
    budget_used = 0.0
    history: list = []
    detected = False
    theta_warm: np.ndarray | None = None
    mid = 0

    for _ in range(max_missions):
        if budget - budget_used < 4:
            break
        try:
            prop = propose_via_hdr_topk(
                current, grid, detector_model, budget - budget_used,
                hdr_mass=hdr_mass, top_k=top_k, cost_exponent=cost_exponent,
                history=history, repetition_penalty=repetition_penalty,
                exploration_mix=exploration_mix,
            )
        except RuntimeError:
            break
        mid += 1
        covered = grid.coverage_mask(prop.x_min, prop.x_max, prop.y_min, prop.y_max)
        if covered[true_cell]:
            rho_true = float(true_detector.rho(
                grid.depth[true_cell], grid.roughness[true_cell], prop.effort
            ))
            s_t = int(rng.random() < rho_true)
        else:
            s_t = 0
        history.append(_Rec(mid, prop.x_min, prop.x_max, prop.y_min, prop.y_max,
                             prop.effort, s_t))
        budget_used += prop.cost
        if s_t == 1:
            detected = True
            p_true_evol.append(p_true_evol[-1])
            entropy_evol.append(entropy_evol[-1])
            break
        # Laplace update on (beta, tau) using the whole history so far.
        lp = laplace_approx(spec, detector_model, history, grid,
                             init_theta=theta_warm)
        theta_warm = lp.theta_map
        current = laplace_pi_mean(lp, grid, n_samples=laplace_n_samples, seed=seed)
        p_true_evol.append(float(current[true_cell]))
        entropy_evol.append(_entropy(current))

    return CampaignResult(
        detected=detected,
        n_missions=len(history),
        cost_used=float(budget_used),
        final_p_true=float(current[true_cell]),
        final_entropy=_entropy(current),
        p_true_evolution=p_true_evol,
        entropy_evolution=entropy_evol,
    )


def sample_plausible_true_cells(
    reference_pi: np.ndarray, n_trials: int, seed: int = 0
) -> np.ndarray:
    """Sample bomb positions from a reference prior_predictive.

    Doing this rather than uniform sampling keeps the test pool focused on
    scenarios that are physically consistent with the accident's inputs.
    """
    rng = np.random.default_rng(seed)
    p = reference_pi / reference_pi.sum()
    return rng.choice(len(p), size=n_trials, p=p)


def run_fast_mc(
    priors: dict[str, PriorSpec],
    grid: GridInfo,
    detector_model: DetectionModel,
    true_detector: TrueDetector,
    true_cells: Sequence[int],
    budget: float = 230.0,
    n_prior_samples: int = 800,
    backend: str = "cell_update",
    **strategy_kwargs,
) -> pd.DataFrame:
    """Run a campaign for each (prior, true_cell), return long-form DataFrame.

    ``backend`` selects the per-mission update mechanism:
        * ``"cell_update"`` — closed-form pointwise Bayesian reweighting of
          pi_j. Fast (~5 sec for 1000 trials) but does NOT update the
          posterior on (β, τ). Default.
        * ``"laplace"`` — INLA-style: re-fits the 7-dim posterior on
          (β, τ) by MAP + Hessian after every mission and recomputes the
          marginal cell prior. ~50x slower but learns the latent physics.
    """
    if backend not in {"cell_update", "laplace"}:
        raise ValueError(f"Unknown backend {backend!r}; expected 'cell_update' or 'laplace'.")
    print(f"Pre-computing prior_predictive for {len(priors)} priors...")
    prior_preds = {name: spec.prior_predictive_pi(grid.x, grid.y, n_samples=n_prior_samples, seed=0)
                    for name, spec in priors.items()}

    rows = []
    t0 = time.time()
    for prior_name in priors:
        for trial_idx, true_cell in enumerate(true_cells):
            seed_txt = f"{prior_name}|{int(true_cell)}|{trial_idx}".encode("utf-8")
            seed = zlib.crc32(seed_txt)
            if backend == "cell_update":
                r = fast_campaign(
                    prior_preds[prior_name], grid, detector_model, true_detector,
                    int(true_cell), budget=budget, seed=seed, **strategy_kwargs,
                )
            else:
                # laplace: needs spec (parametric); empirical priors fall back to cell_update
                spec = priors[prior_name]
                if hasattr(spec, "b0_sigma"):
                    r = laplace_campaign(
                        spec, grid, detector_model, true_detector,
                        int(true_cell), budget=budget, seed=seed, **strategy_kwargs,
                    )
                else:
                    # EmpiricalPriorSpec or similar: no β to update -> cell_update.
                    r = fast_campaign(
                        prior_preds[prior_name], grid, detector_model, true_detector,
                        int(true_cell), budget=budget, seed=seed, **strategy_kwargs,
                    )
            rows.append({
                "prior": prior_name,
                "trial": trial_idx,
                "true_cell": int(true_cell),
                "detected": r.detected,
                "n_missions": r.n_missions,
                "cost_used": r.cost_used,
                "final_p_true": r.final_p_true,
                "final_entropy": r.final_entropy,
                "backend": backend,
            })
        elapsed = time.time() - t0
        rate = (rows_done := sum(1 for r in rows if r['prior'] == prior_name)) / max(elapsed, 1e-3)
        print(f"  [{elapsed:6.1f}s] {prior_name:35s}  done {rows_done}/{len(true_cells)}  "
              f"({rate:.1f} trials/s)")
    return pd.DataFrame(rows)


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("prior")
        .agg(
            detection_rate=("detected", "mean"),
            mean_n_missions=("n_missions", "mean"),
            median_n_missions=("n_missions", "median"),
            mean_cost_used=("cost_used", "mean"),
            mean_final_p_true=("final_p_true", "mean"),
            mean_final_entropy=("final_entropy", "mean"),
            n_trials=("detected", "size"),
        )
        .reset_index()
        .sort_values("detection_rate", ascending=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Number of plausible-bomb trials per prior")
    parser.add_argument("--budget", type=float, default=230.0)
    parser.add_argument("--reference-prior", type=str, default="P6_mixed_wit_informative",
                        help="Prior used to sample plausible bomb positions")
    parser.add_argument("--include-grid", action="store_true",
                        help="Include hyperparameter grid priors (make_grid_priors)")
    parser.add_argument("--out", type=str, default="fast_mc_summary.csv")
    parser.add_argument("--hdr-mass", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cost-exponent", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=0.15)
    parser.add_argument("--exploration-mix", type=float, default=0.05)
    parser.add_argument("--update-temperature", type=float, default=1.0,
                        help="Exponent applied to the failed-search likelihood (1-q).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = load_grid(Path(__file__).resolve().parent.parent / "data" / "grid_dataset.csv")
    detector_model = DETECTORS["D1_saturating_exponential"]
    true_detector = TrueDetector()

    priors = dict(PRIORS)
    if args.include_grid:
        priors.update(make_grid_priors())

    ref_pi = priors[args.reference_prior].prior_predictive_pi(grid.x, grid.y, n_samples=2000, seed=0)
    true_cells = sample_plausible_true_cells(ref_pi, args.n_trials, seed=42)
    print(f"Reference prior for sampling: {args.reference_prior}")
    print(f"Sampled {len(true_cells)} unique-ish bomb positions "
          f"(unique cells: {len(set(true_cells.tolist()))})")

    df = run_fast_mc(
        priors, grid, detector_model, true_detector, true_cells,
        budget=args.budget,
        hdr_mass=args.hdr_mass, top_k=args.top_k,
        cost_exponent=args.cost_exponent,
        repetition_penalty=args.repetition_penalty,
        exploration_mix=args.exploration_mix,
        update_temperature=args.update_temperature,
    )
    df_path = RESULTS_DIR / args.out
    df.to_csv(df_path, index=False)
    print(f"\nWrote per-trial results: {df_path}")

    summary = aggregate_summary(df)
    sum_path = RESULTS_DIR / args.out.replace(".csv", "_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(f"Wrote aggregated summary: {sum_path}")
    print("\nTop priors by detection rate:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
