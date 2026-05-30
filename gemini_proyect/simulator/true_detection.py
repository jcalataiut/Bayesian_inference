"""Ground-truth detection model used by the simulator.

This is intentionally kept separate from the *modeling* code that the student
uses for inference. The point of the simulator is to evaluate how well our
modeling assumptions hold up when the data-generating process is *not* the
same as our inference model.

Functional form
---------------
For a cell with normalized depth d and roughness r, inspected at effort
level e, the conditional detection probability is

    rho_true(d, r, e) = base * f_effort(e) * (1 - d)^alpha_d * (1 - r)^alpha_r

with f_effort(e) saturating as e grows (one-pass capped sensor).

Defaults are calibrated qualitatively from previous_missions_reports.pdf:
  * report 2 (shallow, smooth, high effort -> detected fast): rho close to 1
  * report 3 (deep, rough, high effort -> no detection): rho small
  * report 4 (intermediate, irregular -> needed repetition): rho moderate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrueDetector:
    base: float = 1.0
    alpha_d: float = 0.4
    alpha_r: float = 0.3
    # f_effort(e) = 1 - (1 - p_unit)^e, i.e. e independent passes of a sensor
    # whose per-pass effectiveness is p_unit. Setting p_unit moderately
    # high so a single pass is informative in easy cells, but extra effort
    # still adds meaningful coverage in harder cells.
    p_unit: float = 0.6

    def rho(self, depth: np.ndarray, roughness: np.ndarray, effort: int) -> np.ndarray:
        depth = np.asarray(depth)
        roughness = np.asarray(roughness)
        f_effort = 1.0 - (1.0 - self.p_unit) ** effort
        terrain = (1.0 - depth) ** self.alpha_d * (1.0 - roughness) ** self.alpha_r
        out = self.base * f_effort * terrain
        return np.clip(out, 0.0, 1.0)
