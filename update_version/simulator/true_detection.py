"""Ground-truth detection model for the local simulator.

Kept separate from the *modeling* detectors used for inference: that asymmetry
mirrors the real situation (the professor's hidden binomial is not the
detector we use for our posterior). Parameters are chosen so that a single
pass in a shallow + smooth cell is highly informative, but deep + rough cells
remain difficult even at high effort — qualitatively consistent with
``previous_missions_reports.pdf``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrueDetector:
    base: float = 1.0
    alpha_d: float = 0.4
    alpha_r: float = 0.3
    p_unit: float = 0.6

    def rho(self, depth: np.ndarray, roughness: np.ndarray, effort: int) -> np.ndarray:
        depth = np.asarray(depth)
        roughness = np.asarray(roughness)
        f_effort = 1.0 - (1.0 - self.p_unit) ** effort
        terrain = (1.0 - depth) ** self.alpha_d * (1.0 - roughness) ** self.alpha_r
        return np.clip(self.base * f_effort * terrain, 0.0, 1.0)
