"""Monte Carlo comparison of (prior, detector) combinations.

For each combination we run a full simulated search campaign multiple times,
each time:
    * planting the object somewhere on the grid,
    * iteratively running NUTS + proposing missions until detection or budget
      exhaustion.

Aggregated metrics: detection rate, average missions until detection, average
cost used, average final-posterior entropy. Output: CSV in ../results/.

This is slow because every mission triggers an MCMC run. The disk cache
(see ``modeling.pymc_model.CACHE_DIR``) is reused across trials when the
same history prefix appears.

Usage:
    python -m update_version.experiments.compare_priors --trials 10
"""
from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ..modeling.detection import DETECTORS
from ..modeling.posterior import posterior_over_Z, summarise_posterior_over_Z
from ..modeling.priors_logistic import PRIORS
from ..modeling.pymc_model import run_mcmc
from ..simulator.environment import SearchEnvironment
from ..simulator.grid import load_grid
from ..simulator.true_detection import TrueDetector
from ..strategies.next_mission import propose_expected_detection


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_campaign(
    spec,
    detector,
    grid,
    true_detector,
    budget,
    seed,
    max_missions=8,
    draws=400,
    tune=400,
    chains=2,
):
    """Run one simulated campaign and return per-trial metrics."""
    env = SearchEnvironment(
        grid=grid,
        detector=true_detector,
        budget_total=float(budget),
        rng=np.random.default_rng(seed),
    )
    # Plant from the prior-predictive so true cell is plausibly likely.
    pi_prior = spec.prior_predictive_pi(grid.x, grid.y, n_samples=300, seed=seed)
    env.plant_object(prior=pi_prior)

    p_Z = pi_prior.copy()
    detected = False
    for _ in range(max_missions):
        if env.budget_remaining < 4:
            break
        proposal = propose_expected_detection(p_Z, grid, detector, env.budget_remaining)
        rec = env.run_mission(**proposal.as_kwargs())
        if rec.s_t == 1:
            detected = True
            break
        idata = run_mcmc(spec, detector, env.history, grid,
                          draws=draws, tune=tune, chains=chains, cores=1,
                          progressbar=False, use_cache=True)
        p_Z = posterior_over_Z(idata, grid, detector, env.history)

    summ = summarise_posterior_over_Z(p_Z, grid)
    return {
        "detected": int(detected),
        "n_missions": len(env.history),
        "cost_used": env.budget_used,
        "final_entropy_nats": summ["entropy_nats"],
        "final_max_prob": summ["max_prob"],
        "true_cell": env.true_cell,
        "argmax_cell": summ["argmax_cell"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per (prior, detector) combo")
    parser.add_argument("--budget", type=float, default=230.0)
    parser.add_argument("--max-missions", type=int, default=8)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--tune", type=int, default=400)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--priors", nargs="*", default=None,
                        help="Restrict to a subset of priors by name")
    parser.add_argument("--detectors", nargs="*", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = load_grid(Path(__file__).resolve().parent.parent / "data" / "grid_dataset.csv")
    true_detector = TrueDetector()

    prior_names = args.priors or list(PRIORS.keys())
    det_names = args.detectors or list(DETECTORS.keys())

    rows = []
    t0 = time.time()
    for prior_name, det_name in product(prior_names, det_names):
        spec = PRIORS[prior_name]
        detector = DETECTORS[det_name]
        for trial in range(args.trials):
            seed = hash((prior_name, det_name, trial)) % (2 ** 32)
            metrics = run_campaign(
                spec, detector, grid, true_detector,
                budget=args.budget, seed=seed,
                max_missions=args.max_missions,
                draws=args.draws, tune=args.tune, chains=args.chains,
            )
            metrics.update({
                "prior": prior_name,
                "detector": det_name,
                "trial": trial,
            })
            rows.append(metrics)
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s] {prior_name:35s} {det_name:30s} trial {trial}: "
                  f"detected={metrics['detected']} n_missions={metrics['n_missions']}")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "comparison_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")

    summary = (
        df.groupby(["prior", "detector"])
        .agg(
            detection_rate=("detected", "mean"),
            mean_missions=("n_missions", "mean"),
            mean_cost_used=("cost_used", "mean"),
            mean_final_entropy=("final_entropy_nats", "mean"),
        )
        .reset_index()
        .sort_values("detection_rate", ascending=False)
    )
    print("\nAggregated summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
