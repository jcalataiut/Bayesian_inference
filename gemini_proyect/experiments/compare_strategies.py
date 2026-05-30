"""Monte Carlo comparison of search strategies and priors.

For each (strategy, prior) combination, run ``n_trials`` independent search
campaigns. The hidden object is sampled from ``truth_prior`` (held fixed
across all strategies for fairness).

Outputs a summary table with detection rate, mean missions to detect,
mean budget used, and mean posterior-argmax error.

Run with:

    python -m gemini_proyect.experiments.compare_strategies
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..modeling import (
    DetectionModel,
    drift_prior,
    drift_prior_with_witnesses,
    posterior_update,
    sinking_adjusted_prior,
    uniform_prior,
)
from ..simulator import SearchEnvironment, TrueDetector, load_grid
from ..strategies import (
    propose_info_gain,
    propose_max_expected_detection,
    propose_max_posterior_rect,
)


GRID_CSV = Path(__file__).resolve().parents[1] / "data" / "grid_dataset.csv"


@dataclass
class TrialResult:
    strategy: str
    prior: str
    seed: int
    found: bool
    n_missions: int
    budget_used: float
    argmax_error: float  # Euclidean distance between argmax(posterior) and truth


STRATEGIES: dict[str, Callable] = {
    "info_gain": propose_info_gain,
    "max_expected_detection": propose_max_expected_detection,
    "max_posterior_rect": propose_max_posterior_rect,
}


def _build_prior(name: str, grid) -> np.ndarray:
    if name == "uniform":
        return uniform_prior(grid)
    if name == "drift":
        return drift_prior(grid)
    if name == "drift+witnesses":
        return drift_prior_with_witnesses(grid)
    if name == "drift+witnesses+deep":
        return sinking_adjusted_prior(drift_prior_with_witnesses(grid), grid, depth_bias=1.0)
    raise ValueError(f"Unknown prior name {name!r}")


def run_trial(
    strategy: str,
    prior_name: str,
    seed: int,
    max_missions: int = 250,
    budget_total: float = 230.0,
) -> TrialResult:
    env = SearchEnvironment.from_csv(
        GRID_CSV, seed=seed, detector=TrueDetector(), budget_total=budget_total
    )
    grid = env.grid

    # Truth is *always* drawn from the witness-informed prior so the ranking
    # reflects modeling quality, not whether the inference prior happens to
    # match the truth-generating prior.
    truth_prior = drift_prior_with_witnesses(grid)
    true_cell = env.plant_object(prior=truth_prior)

    prior = _build_prior(prior_name, grid)
    model = DetectionModel()
    posterior = prior.copy()
    propose = STRATEGIES[strategy]

    found = False
    for _ in range(max_missions):
        # Minimum mission cost is 1x1 cells * effort 1 = 1.
        if env.budget_remaining < 1:
            break
        try:
            proposal = propose(
                posterior=posterior,
                grid=grid,
                model=model,
                budget_remaining=env.budget_remaining,
            )
        except RuntimeError:
            break
        record = env.run_mission(**proposal.as_kwargs())
        posterior = posterior_update(prior, grid, model, env.history)
        if record.s_t == 1:
            found = True
            break

    argmax_cell = int(np.argmax(posterior))
    err = float(
        np.hypot(grid.x[argmax_cell] - grid.x[true_cell], grid.y[argmax_cell] - grid.y[true_cell])
    )
    return TrialResult(
        strategy=strategy,
        prior=prior_name,
        seed=seed,
        found=found,
        n_missions=len(env.history),
        budget_used=env.budget_used,
        argmax_error=err,
    )


def run_grid(
    strategies: list[str],
    priors: list[str],
    n_trials: int = 50,
    base_seed: int = 1000,
) -> pd.DataFrame:
    rows: list[TrialResult] = []
    for strategy in strategies:
        for prior_name in priors:
            for k in range(n_trials):
                rows.append(
                    run_trial(strategy=strategy, prior_name=prior_name, seed=base_seed + k)
                )
    return pd.DataFrame([r.__dict__ for r in rows])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["strategy", "prior"], as_index=False)
    return g.agg(
        detection_rate=("found", "mean"),
        mean_missions=("n_missions", "mean"),
        mean_budget_used=("budget_used", "mean"),
        mean_argmax_error=("argmax_error", "mean"),
    ).sort_values("detection_rate", ascending=False)


if __name__ == "__main__":
    strategies = ["info_gain", "max_expected_detection", "max_posterior_rect"]
    priors = ["uniform", "drift", "drift+witnesses", "drift+witnesses+deep"]
    df = run_grid(strategies=strategies, priors=priors, n_trials=10)
    summary = summarize(df)
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "trials.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
