"""Local replica of the professor's search webapp.

The real webapp accepts (x_min, x_max, y_min, y_max, effort), charges
``cost = effort * n_cells``, debits a shared budget, and returns
``s_t in {0, 1}`` (detection / no detection).

We mirror that interface exactly so that strategies developed against this
class can be plugged into the real submission with no code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .grid import GridInfo, load_grid
from .true_detection import TrueDetector


@dataclass
class MissionRecord:
    mission_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    effort: int
    n_cells: int
    cost: float
    s_t: int
    budget_remaining: float


@dataclass
class SearchEnvironment:
    """Headless replica of the search-missions Streamlit app."""

    grid: GridInfo
    detector: TrueDetector
    budget_total: float = 230.0
    team_id: str = "SIM"
    true_cell: Optional[int] = None  # set via plant_object
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    history: list[MissionRecord] = field(default_factory=list)
    _next_mission_id: int = 1

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        seed: int = 0,
        budget_total: float = 230.0,
        detector: TrueDetector | None = None,
        team_id: str = "SIM",
    ) -> "SearchEnvironment":
        grid = load_grid(csv_path)
        return cls(
            grid=grid,
            detector=detector or TrueDetector(),
            budget_total=budget_total,
            team_id=team_id,
            rng=np.random.default_rng(seed),
        )

    @property
    def budget_used(self) -> float:
        return float(sum(m.cost for m in self.history))

    @property
    def budget_remaining(self) -> float:
        return self.budget_total - self.budget_used

    def plant_object(self, cell_id: int | None = None, prior: np.ndarray | None = None) -> int:
        """Place the hidden object. If ``prior`` is given, sample from it.

        Returns the planted cell_id (also stored as ``self.true_cell``).
        """
        if cell_id is None:
            if prior is None:
                cell_id = int(self.rng.integers(0, self.grid.n_cells))
            else:
                p = np.asarray(prior, dtype=float)
                p = p / p.sum()
                cell_id = int(self.rng.choice(self.grid.n_cells, p=p))
        self.true_cell = cell_id
        return cell_id

    def run_mission(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        effort: int,
    ) -> MissionRecord:
        if self.true_cell is None:
            raise RuntimeError("plant_object() must be called before run_mission().")

        covered = self.grid.cells_in_rectangle(x_min, x_max, y_min, y_max)
        n_cells = int(covered.size)
        cost = float(effort) * n_cells

        if cost > self.budget_remaining + 1e-9:
            raise ValueError(
                f"Mission cost {cost:.1f} exceeds remaining budget {self.budget_remaining:.1f}."
            )

        # Determine detection.
        if self.true_cell in covered:
            depth = self.grid.depth[self.true_cell]
            roughness = self.grid.roughness[self.true_cell]
            rho = float(self.detector.rho(depth, roughness, effort))
            s_t = int(self.rng.random() < rho)
        else:
            s_t = 0

        record = MissionRecord(
            mission_id=self._next_mission_id,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            effort=int(effort),
            n_cells=n_cells,
            cost=cost,
            s_t=s_t,
            budget_remaining=self.budget_remaining - cost,
        )
        self._next_mission_id += 1
        self.history.append(record)
        return record

    def history_to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "team_id": self.team_id,
                "mission_id": m.mission_id,
                "x_min": m.x_min,
                "x_max": m.x_max,
                "y_min": m.y_min,
                "y_max": m.y_max,
                "effort": m.effort,
                "n_cells": m.n_cells,
                "cost": m.cost,
                "s_t": m.s_t,
                "budget_remaining": m.budget_remaining,
                "timestamp": "",
            }
            for m in self.history
        ]
        return pd.DataFrame(rows)
