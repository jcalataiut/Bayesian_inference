"""Multi-campaign Monte Carlo: same truth across strategies, varies per campaign.

The professor's deliverable evaluates only the single real campaign you run
on the webapp. To choose a strategy with confidence, we want to know how
each strategy performs across *many* possible truths.

For each campaign k in 0..n_campaigns-1:
    * sample one truth location ``z_k`` from ``truth_prior`` (deterministic
      via campaign seed) — fixed across strategies so the comparison is
      paired, not unpaired (lower variance estimates).
    * for each strategy s:
        - rebuild a fresh ``SearchEnvironment``,
        - plant the same ``z_k``,
        - run the strategy until detection or the budget is exhausted,
        - record (found, n_missions, budget_used, final_argmax_error).

Returns a tidy DataFrame; a helper aggregates it into per-strategy summary
statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from ..modeling import (
    DetectionModel,
    drift_prior,
    drift_prior_with_witnesses,
    physics_prior,
    posterior_update,
    sinking_adjusted_prior,
    uniform_prior,
)
from ..simulator import SearchEnvironment, TrueDetector
from ..strategies import (
    CommitAndVerifyStrategy,
    StochasticPosteriorSampler,
    propose_cooldown_aware,
    propose_info_gain,
    propose_max_expected_detection,
    propose_max_posterior_rect,
)


GRID_CSV = Path(__file__).resolve().parents[1] / "data" / "grid_dataset.csv"


@dataclass
class CampaignResult:
    campaign: int
    strategy: str
    found: bool
    n_missions: int
    budget_used: float
    final_argmax_error: float


def _build_prior(name: str, grid, depth_bias: float = 1.0, mc_seed: int = 0) -> np.ndarray:
    if name == "uniform":
        return uniform_prior(grid)
    if name == "drift":
        return drift_prior(grid)
    if name == "drift+witnesses":
        return drift_prior_with_witnesses(grid)
    if name == "drift+witnesses+deep":
        return sinking_adjusted_prior(drift_prior_with_witnesses(grid), grid, depth_bias=depth_bias)
    if name == "physics_mc":
        return physics_prior(grid, n_samples=2000, seed=mc_seed, depth_bias=depth_bias)
    raise ValueError(name)


def _make_proposer(
    name: str,
    *,
    confidence: float,
    stochastic_seed: int,
    n_missions_target: int = 5,
) -> Callable:
    if name == "max_expected_detection":
        return propose_max_expected_detection
    if name == "info_gain":
        return propose_info_gain
    if name == "max_posterior_rect":
        return propose_max_posterior_rect
    if name == "commit_and_verify":
        cav = CommitAndVerifyStrategy(confidence=confidence)
        return lambda p, g, m, b: cav.propose(p, g, m, b)
    if name == "thompson":
        samp = StochasticPosteriorSampler(seed=stochastic_seed)
        return lambda p, g, m, b: samp.propose(p, g, m, b)
    if name == "cooldown_aware":
        return lambda p, g, m, b: propose_cooldown_aware(
            p, g, m, b, n_missions_target=n_missions_target
        )
    raise ValueError(name)


def run_multi_campaign(
    strategies: Iterable[str],
    prior_name: str = "drift+witnesses",
    n_campaigns: int = 100,
    budget_total: float = 230.0,
    max_missions: int = 250,
    depth_bias: float = 1.0,
    detector_kwargs: Optional[dict] = None,
    detection_model_kwargs: Optional[dict] = None,
    base_seed: int = 12345,
    confidence: float = 0.80,
    n_missions_target: int = 5,
    mc_seed: int = 0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Run ``n_campaigns`` campaigns; same truth across strategies per campaign.

    ``progress_cb(done, total)`` is called after each (campaign, strategy)
    pair so Streamlit can show a progress bar.
    """
    detector_kwargs = detector_kwargs or {}
    detection_model_kwargs = detection_model_kwargs or {}

    # Build the grid once and reuse it (the dataset is fixed).
    truth_env_template = SearchEnvironment.from_csv(
        GRID_CSV, seed=0, budget_total=budget_total, detector=TrueDetector(**detector_kwargs)
    )
    grid = truth_env_template.grid
    truth_prior = _build_prior(prior_name, grid, depth_bias=depth_bias, mc_seed=mc_seed)

    # Pre-sample the truth cell for each campaign deterministically.
    truth_rng = np.random.default_rng(base_seed)
    truth_cells = truth_rng.choice(grid.n_cells, size=n_campaigns, p=truth_prior)

    strategies = list(strategies)
    total_steps = n_campaigns * len(strategies)
    step = 0
    rows: list[CampaignResult] = []

    for k in range(n_campaigns):
        true_cell = int(truth_cells[k])
        for s_name in strategies:
            env = SearchEnvironment.from_csv(
                GRID_CSV,
                seed=base_seed + k * 1000 + hash(s_name) % 1000,
                budget_total=budget_total,
                detector=TrueDetector(**detector_kwargs),
            )
            env.plant_object(cell_id=true_cell)

            prior = _build_prior(prior_name, grid, depth_bias=depth_bias, mc_seed=mc_seed)
            model = DetectionModel(**detection_model_kwargs)
            posterior = prior.copy()

            propose = _make_proposer(
                s_name,
                confidence=confidence,
                stochastic_seed=base_seed + k * 7 + 1,
                n_missions_target=n_missions_target,
            )

            found = False
            for _ in range(max_missions):
                if env.budget_remaining < 1:
                    break
                try:
                    prop = propose(posterior, grid, model, env.budget_remaining)
                except RuntimeError:
                    break
                rec = env.run_mission(**prop.as_kwargs())
                posterior = posterior_update(prior, grid, model, env.history)
                if rec.s_t == 1:
                    found = True
                    break

            argmax = int(np.argmax(posterior))
            err = float(
                np.hypot(grid.x[argmax] - grid.x[true_cell], grid.y[argmax] - grid.y[true_cell])
            )
            rows.append(
                CampaignResult(
                    campaign=k,
                    strategy=s_name,
                    found=found,
                    n_missions=len(env.history),
                    budget_used=env.budget_used,
                    final_argmax_error=err,
                )
            )
            step += 1
            if progress_cb is not None:
                progress_cb(step, total_steps)

    return pd.DataFrame([r.__dict__ for r in rows])


def summarize_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("strategy", as_index=False)
    summary = g.agg(
        n_campaigns=("campaign", "nunique"),
        detection_rate=("found", "mean"),
        mean_missions=("n_missions", "mean"),
        median_missions=("n_missions", "median"),
        mean_budget_used=("budget_used", "mean"),
        mean_argmax_error=("final_argmax_error", "mean"),
    ).sort_values("detection_rate", ascending=False)
    return summary
