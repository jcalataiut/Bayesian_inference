"""Grid loading and indexing utilities.

The grid is fixed: Nx=50, Ny=35. Each cell has a center (x, y), a normalized
depth in [0, 1], and a normalized roughness in [0, 1].
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
    """A flat-array view of the grid keyed by cell_id.

    All per-cell quantities are stored as 1-D arrays of length N = Nx*Ny.
    The 2-D layout (Ny, Nx) is recoverable via ``reshape_2d``.
    """

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
        """Reshape a length-N flat array into (Ny, Nx) for plotting."""
        return values.reshape(self.Ny, self.Nx)

    def cells_in_rectangle(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> np.ndarray:
        """Return cell_ids whose centers fall inside [x_min, x_max] x [y_min, y_max]."""
        mask = (
            (self.x >= x_min)
            & (self.x <= x_max)
            & (self.y >= y_min)
            & (self.y <= y_max)
        )
        return self.cell_id[mask]


def load_grid(csv_path: str | Path) -> GridInfo:
    df = pd.read_csv(csv_path).sort_values("cell_id").reset_index(drop=True)
    # Sanity: the dataset rows are aligned with cell_id 0..N-1.
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
