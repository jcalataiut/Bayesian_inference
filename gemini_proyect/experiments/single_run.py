"""One end-to-end simulated search run.

Run with:

    python -m gemini_proyect.experiments.single_run

(from inside the Delivery_2 directory).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..modeling import (
    DetectionModel,
    drift_prior_with_witnesses,
    posterior_update,
)
from ..simulator import SearchEnvironment, TrueDetector
from ..strategies import (
    propose_info_gain,
    propose_max_expected_detection,
)


GRID_CSV = Path(__file__).resolve().parents[1] / "data" / "grid_dataset.csv"


def run_one(
    strategy: str = "info_gain",
    max_missions: int = 250,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    env = SearchEnvironment.from_csv(
        GRID_CSV, seed=seed, detector=TrueDetector(), budget_total=230.0
    )
    grid = env.grid

    # Use the *true* generating distribution to plant the object so that
    # our prior is informative but not omniscient.
    truth_prior = drift_prior_with_witnesses(grid)
    true_cell = env.plant_object(prior=truth_prior)

    # Our modeling layer.
    prior = drift_prior_with_witnesses(grid)
    model = DetectionModel()
    posterior = prior.copy()

    propose_fn = {
        "info_gain": propose_info_gain,
        "max_expected_detection": propose_max_expected_detection,
    }[strategy]

    found = False
    for _ in range(max_missions):
        if env.budget_remaining < 1:
            break
        try:
            proposal = propose_fn(
                posterior=posterior,
                grid=grid,
                model=model,
                budget_remaining=env.budget_remaining,
            )
        except RuntimeError:
            break
        record = env.run_mission(**proposal.as_kwargs())
        posterior = posterior_update(prior, grid, model, env.history)
        if verbose:
            print(
                f"mission {record.mission_id}: "
                f"rect=({record.x_min:.1f},{record.x_max:.1f})x"
                f"({record.y_min:.1f},{record.y_max:.1f}) "
                f"effort={record.effort} cost={record.cost:.0f} "
                f"s_t={record.s_t} budget_left={record.budget_remaining:.0f}"
            )
        if record.s_t == 1:
            found = True
            break

    return {
        "found": found,
        "n_missions": len(env.history),
        "budget_used": env.budget_used,
        "true_cell": true_cell,
        "true_xy": (float(grid.x[true_cell]), float(grid.y[true_cell])),
        "argmax_posterior": int(np.argmax(posterior)),
    }


if __name__ == "__main__":
    out = run_one()
    print(out)
