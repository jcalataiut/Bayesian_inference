"""Grid loading and indexing utilities.

The grid is fixed (Nx=50, Ny=35). Each cell has a center (x, y), a normalized
depth in [0, 1] and a normalized roughness in [0, 1]. All per-cell quantities
are stored as flat length-N arrays keyed by ``cell_id``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NX_DEFAULT = 50
NY_DEFAULT = 35


@dataclass
class GridInfo:
    Nx: int
    Ny: int
    cell_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    roughness: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.Nx * self.Ny

    def reshape_2d(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.Ny, self.Nx)

    def cells_in_rectangle(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> np.ndarray:
        mask = (
            (self.x >= x_min)
            & (self.x <= x_max)
            & (self.y >= y_min)
            & (self.y <= y_max)
        )
        return self.cell_id[mask]

    def coverage_mask(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> np.ndarray:
        return (
            (self.x >= x_min)
            & (self.x <= x_max)
            & (self.y >= y_min)
            & (self.y <= y_max)
        )


def load_grid(csv_path: str | Path) -> GridInfo:
    df = pd.read_csv(csv_path).sort_values("cell_id").reset_index(drop=True)
    assert (df["cell_id"].to_numpy() == np.arange(len(df))).all()
    Nx = int(df["x"].nunique())
    Ny = int(df["y"].nunique())
    return GridInfo(
        Nx=Nx,
        Ny=Ny,
        cell_id=df["cell_id"].to_numpy(),
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        depth=df["depth"].to_numpy(),
        roughness=df["roughness"].to_numpy(),
    )
