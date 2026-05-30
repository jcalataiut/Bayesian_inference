"""Iterative Empirical Bayes calibration under the budget cap.

This experiment answers:

    Can we improve the search model by using cells where previous simulated
    campaigns found the plane to define a better prior for the next round?

The loop is intentionally train/validation split:

1. Sample physically plausible train and validation bomb positions.
2. Evaluate a pool of priors under several posterior-update temperatures.
3. Use successful train detections from the best combinations to build KDE
   empirical priors.
4. Repeat for ``rounds`` iterations, always reporting validation performance.

Budget is hard-capped at 530, matching the operational constraint.

Example:

    python -m update_version.experiments.iterative_empirical_bayes \\
        --rounds 10 --n-train 200 --n-val 400 --budget 530
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..modeling.detection import DETECTORS
from ..modeling.priors_logistic import EmpiricalPriorSpec, PRIORS
from ..simulator.grid import load_grid
from ..simulator.true_detection import TrueDetector
from .fast_monte_carlo import (
    RESULTS_DIR,
    aggregate_summary,
    run_fast_mc,
    sample_plausible_true_cells,
)


BUDGET_CAP = 530.0


@dataclass(frozen=True)
class UpdateConfig:
    name: str
    update_temperature: float
    exploration_mix: float
    repetition_penalty: float
    hdr_mass: float = 0.80
    top_k: int = 10
    cost_exponent: float = 0.70
    max_missions: int = 12

    def kwargs(self) -> dict[str, Any]:
        return {
            "hdr_mass": self.hdr_mass,
            "top_k": self.top_k,
            "cost_exponent": self.cost_exponent,
            "repetition_penalty": self.repetition_penalty,
            "exploration_mix": self.exploration_mix,
            "update_temperature": self.update_temperature,
            "max_missions": self.max_missions,
        }


UPDATE_CONFIGS = [
    UpdateConfig("std", update_temperature=1.00, exploration_mix=0.05, repetition_penalty=0.15),
    UpdateConfig("lik_x1_5", update_temperature=1.50, exploration_mix=0.05, repetition_penalty=0.15),
    UpdateConfig("lik_x2", update_temperature=2.00, exploration_mix=0.05, repetition_penalty=0.15),
    UpdateConfig("explore_x1_5", update_temperature=1.50, exploration_mix=0.10, repetition_penalty=0.10),
    UpdateConfig("soft_x0_75", update_temperature=0.75, exploration_mix=0.05, repetition_penalty=0.15),
]


ANCHOR_PRIORS = [
    "P3_quadratic_nowit_informative",
    "P4_quadratic_wit_informative",
    "P5_mixed_nowit_informative",
    "P6_mixed_wit_informative",
    "P7_mixed_wit_weak",
    "P8_mixed_wit_vague",
    "P9_mixed_wit_uniform30",
    "P10_mixed_wit_uniform50_temp2",
]


def _summary_by_update(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["prior", "update_form"])
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
        .sort_values(
            ["detection_rate", "mean_final_p_true", "mean_cost_used"],
            ascending=[False, False, True],
        )
    )


def evaluate_pool(
    priors: dict[str, Any],
    grid,
    detector,
    true_detector,
    true_cells: np.ndarray,
    budget: float,
    update_configs: list[UpdateConfig],
    round_idx: int,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all (prior, update_config) combinations."""
    dfs = []
    for cfg in update_configs:
        print(f"    {split} cfg={cfg.name:12s} priors={len(priors)}")
        df = run_fast_mc(
            priors,
            grid,
            detector,
            true_detector,
            true_cells,
            budget=budget,
            **cfg.kwargs(),
        )
        df["update_form"] = cfg.name
        df["round"] = round_idx
        df["split"] = split
        dfs.append(df)
    long_df = pd.concat(dfs, ignore_index=True)
    summary = _summary_by_update(long_df)
    summary["round"] = round_idx
    summary["split"] = split
    return long_df, summary


def build_empirical_priors(
    success_cells: np.ndarray,
    grid,
    round_idx: int,
) -> dict[str, EmpiricalPriorSpec]:
    """Create several KDE priors from detected cells."""
    if success_cells.size == 0:
        return {}
    out = {}
    for bandwidth, uniform_mix in [
        (1.75, 0.05),
        (2.50, 0.10),
        (3.50, 0.15),
        (4.50, 0.20),
    ]:
        name = f"EB_r{round_idx:02d}_bw{bandwidth:g}_u{int(uniform_mix * 100):02d}"
        out[name] = EmpiricalPriorSpec(
            name=name,
            detected_cells=success_cells,
            grid_x=grid.x,
            grid_y=grid.y,
            bandwidth=bandwidth,
            uniform_mix=uniform_mix,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--budget", type=float, default=BUDGET_CAP)
    parser.add_argument("--reference-prior", type=str, default="P8_mixed_wit_vague")
    parser.add_argument("--success-top-combos", type=int, default=3)
    parser.add_argument("--max-pool", type=int, default=12)
    parser.add_argument("--out-prefix", type=str, default="iterative_eb")
    args = parser.parse_args()

    if args.budget > BUDGET_CAP:
        raise ValueError(f"Budget cannot exceed {BUDGET_CAP}; got {args.budget}.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = load_grid(Path(__file__).resolve().parent.parent / "data" / "grid_dataset.csv")
    detector = DETECTORS["D1_saturating_exponential"]
    true_detector = TrueDetector()

    ref_pi = PRIORS[args.reference_prior].prior_predictive_pi(
        grid.x, grid.y, n_samples=3000, seed=0
    )
    train_cells = sample_plausible_true_cells(ref_pi, args.n_train, seed=42)
    val_cells = sample_plausible_true_cells(ref_pi, args.n_val, seed=4242)

    priors_all: dict[str, Any] = dict(PRIORS)
    pool: dict[str, Any] = {name: priors_all[name] for name in ANCHOR_PRIORS}

    all_train, all_val, all_train_summary, all_val_summary = [], [], [], []
    best_history = []
    t0 = time.time()

    for r in range(args.rounds):
        print("\n" + "=" * 78)
        print(f"ROUND {r + 1}/{args.rounds} | pool={len(pool)} | budget={args.budget}")

        train_df, train_summary = evaluate_pool(
            pool, grid, detector, true_detector, train_cells,
            args.budget, UPDATE_CONFIGS, r, "train",
        )
        val_df, val_summary = evaluate_pool(
            pool, grid, detector, true_detector, val_cells,
            args.budget, UPDATE_CONFIGS, r, "val",
        )

        all_train.append(train_df)
        all_val.append(val_df)
        all_train_summary.append(train_summary)
        all_val_summary.append(val_summary)

        best_train = train_summary.iloc[0]
        best_val = val_summary.iloc[0]
        best_history.append({
            "round": r,
            "train_best_prior": best_train["prior"],
            "train_best_update": best_train["update_form"],
            "train_detection_rate": best_train["detection_rate"],
            "val_best_prior": best_val["prior"],
            "val_best_update": best_val["update_form"],
            "val_detection_rate": best_val["detection_rate"],
            "val_mean_cost_used": best_val["mean_cost_used"],
            "elapsed_s": time.time() - t0,
        })

        print(
            f"  TRAIN best: {best_train['prior']} + {best_train['update_form']} "
            f"det={best_train['detection_rate']:.1%}"
        )
        print(
            f"  VAL   best: {best_val['prior']} + {best_val['update_form']} "
            f"det={best_val['detection_rate']:.1%}"
        )

        # Collect detected cells from the top-K train combinations. Duplicates
        # are intentionally kept: they act as KDE weights.
        top = train_summary.head(args.success_top_combos)
        success_chunks = []
        for _, row in top.iterrows():
            mask = (
                (train_df["prior"] == row["prior"])
                & (train_df["update_form"] == row["update_form"])
                & (train_df["detected"])
            )
            success_chunks.append(train_df.loc[mask, "true_cell"].to_numpy(dtype=int))
        success_cells = (
            np.concatenate(success_chunks) if success_chunks else np.array([], dtype=int)
        )
        print(f"  Success cells for EB: {success_cells.size}")

        new_eb = build_empirical_priors(success_cells, grid, r + 1)
        priors_all.update(new_eb)

        # Keep anchors + best current priors + new EB priors. This prevents the
        # pool from exploding while preserving strong older empirical priors.
        top_names = list(dict.fromkeys(
            train_summary.head(args.max_pool)["prior"].tolist()
            + val_summary.head(args.max_pool)["prior"].tolist()
        ))
        keep_names = list(dict.fromkeys(ANCHOR_PRIORS + top_names + list(new_eb)))
        pool = {name: priors_all[name] for name in keep_names if name in priors_all}
        if len(pool) > args.max_pool:
            # Always keep anchors that are still among the original strong
            # baselines; fill the rest by top validation performance.
            selected = []
            for name in ANCHOR_PRIORS:
                if name in pool and name not in selected:
                    selected.append(name)
            for name in val_summary["prior"]:
                if name in pool and name not in selected:
                    selected.append(name)
                if len(selected) >= args.max_pool:
                    break
            for name in new_eb:
                if name in pool and name not in selected:
                    selected.append(name)
                if len(selected) >= args.max_pool:
                    break
            pool = {name: priors_all[name] for name in selected[:args.max_pool]}

    train_all = pd.concat(all_train, ignore_index=True)
    val_all = pd.concat(all_val, ignore_index=True)
    train_summary_all = pd.concat(all_train_summary, ignore_index=True)
    val_summary_all = pd.concat(all_val_summary, ignore_index=True)
    history = pd.DataFrame(best_history)

    prefix = RESULTS_DIR / args.out_prefix
    train_all.to_csv(f"{prefix}_train_long.csv", index=False)
    val_all.to_csv(f"{prefix}_val_long.csv", index=False)
    train_summary_all.to_csv(f"{prefix}_train_summary.csv", index=False)
    val_summary_all.to_csv(f"{prefix}_val_summary.csv", index=False)
    history.to_csv(f"{prefix}_history.csv", index=False)

    best = val_summary_all.sort_values(
        ["detection_rate", "mean_final_p_true", "mean_cost_used"],
        ascending=[False, False, True],
    ).iloc[0]
    print("\n" + "=" * 78)
    print("BEST VALIDATION COMBINATION")
    print(best.to_string())
    print("\nHistory:")
    print(history.to_string(index=False))
    print(f"\nWrote CSVs with prefix: {prefix}")


if __name__ == "__main__":
    main()
