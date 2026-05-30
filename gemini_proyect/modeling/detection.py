"""Our detection model rho_{t,j} (used for inference, not for simulation).

We follow a *single-pass-equivalent* construction motivated by previous
mission reports:

    rho(j, e) = 1 - exp( -lambda(j) * e )

with

    lambda(j) = lambda_0 * (1 - depth_j)^a_d * (1 - roughness_j)^a_r .

Properties
----------
* rho in [0, 1) for all parameters.
* rho is increasing in effort, in (1 - depth), and in (1 - roughness).
* Doubling effort is equivalent to two independent passes of a sensor with
  per-pass success ``1 - exp(-lambda(j))``. This is exactly the qualitative
  pattern described in report 4 (repeated inspection eventually detected).
* Parameters are interpretable: ``lambda_0`` is the per-effort detection
  rate in the easiest possible cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionModel:
    lambda_0: float = 1.2
    a_d: float = 1.0
    a_r: float = 0.6

    def lambda_cell(self, depth: np.ndarray, roughness: np.ndarray) -> np.ndarray:
        return self.lambda_0 * (1.0 - depth) ** self.a_d * (1.0 - roughness) ** self.a_r

    def rho(
        self,
        depth: np.ndarray,
        roughness: np.ndarray,
        effort: int | np.ndarray,
    ) -> np.ndarray:
        """Per-cell detection probability given coverage and effort."""
        lam = self.lambda_cell(np.asarray(depth), np.asarray(roughness))
        return 1.0 - np.exp(-lam * np.asarray(effort, dtype=float))
