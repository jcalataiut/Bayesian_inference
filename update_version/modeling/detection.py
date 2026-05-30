"""Detection models rho_{t,j}(depth, roughness, effort).

Four families with the same interface, all returning probabilities in [0, 1].
Parameters are fixed (not Bayesian) and calibrated qualitatively from
``previous_missions_reports.pdf``:

    * Report 2: shallow + smooth + high effort  ->  rho ~= 1
    * Report 3: deep + rough + high effort      ->  rho small
    * Report 4: intermediate, needed repetition ->  rho moderate

All four families share the property that rho is monotonically increasing in
effort and in (1 - depth) and (1 - roughness). The specific functional form
differs so we can test sensitivity in the comparison experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class DetectionModel(Protocol):
    name: str

    def rho(
        self,
        depth: np.ndarray,
        roughness: np.ndarray,
        effort: int | np.ndarray,
    ) -> np.ndarray:
        ...


@dataclass
class SaturatingExponentialDetector:
    """rho = 1 - exp(-lambda_0 * (1-d)^a_d * (1-r)^a_r * e).

    Classic search-theory form (Koopman, Stone). Each unit of effort
    contributes independently to a Poisson-like detection rate. Saturates
    at 1 as effort grows.
    """

    lambda_0: float = 1.2
    a_d: float = 1.0
    a_r: float = 0.6
    name: str = "saturating_exponential"

    def rho(self, depth, roughness, effort):
        depth = np.asarray(depth, dtype=float)
        roughness = np.asarray(roughness, dtype=float)
        e = np.asarray(effort, dtype=float)
        lam = self.lambda_0 * (1.0 - depth) ** self.a_d * (1.0 - roughness) ** self.a_r
        return 1.0 - np.exp(-lam * e)


@dataclass
class IndependentPassesDetector:
    """rho = 1 - (1 - p_unit * (1-d)^a_d * (1-r)^a_r)^e.

    Effort interpreted as e independent inspections of the cell, each
    succeeding with probability p_unit * terrain. Equivalent to "the cell is
    detected unless every pass fails".
    """

    p_unit: float = 0.5
    a_d: float = 0.4
    a_r: float = 0.3
    name: str = "independent_passes"

    def rho(self, depth, roughness, effort):
        depth = np.asarray(depth, dtype=float)
        roughness = np.asarray(roughness, dtype=float)
        e = np.asarray(effort, dtype=float)
        per_pass = self.p_unit * (1.0 - depth) ** self.a_d * (1.0 - roughness) ** self.a_r
        per_pass = np.clip(per_pass, 0.0, 1.0)
        return 1.0 - (1.0 - per_pass) ** e


@dataclass
class LogisticDetector:
    """rho = sigmoid(a0 + a1*log(e+1) + a2*(1-d) + a3*(1-r)).

    GLM-style detector with effort entering on the log scale (diminishing
    returns) and terrain entering linearly. Hyperparameters chosen so that
    a single-effort inspection of a shallow + smooth cell is well above 0.5
    while a deep + rough cell stays well below 0.5 even at high effort.
    """

    a0: float = -1.5
    a1: float = 1.5
    a2: float = 2.0
    a3: float = 1.2
    name: str = "logistic"

    def rho(self, depth, roughness, effort):
        depth = np.asarray(depth, dtype=float)
        roughness = np.asarray(roughness, dtype=float)
        e = np.asarray(effort, dtype=float)
        z = (
            self.a0
            + self.a1 * np.log(e + 1.0)
            + self.a2 * (1.0 - depth)
            + self.a3 * (1.0 - roughness)
        )
        return _sigmoid(z)


@dataclass
class MultiplicativeThresholdDetector:
    """rho = (1 - (1-p_u)^e) * (1-d)^a_d * (1-r)^a_r * sigmoid(k*(thr - d*r)).

    Combines two ideas:
    * a saturating-effort term ``1 - (1-p_u)^e`` (number of independent passes),
    * a multiplicative terrain term,
    * a soft threshold ``sigmoid(k*(thr - d*r))`` that kills detection in
      cells whose combined difficulty d*r exceeds the threshold thr.
    """

    p_unit: float = 0.5
    a_d: float = 0.5
    a_r: float = 0.4
    threshold: float = 0.5
    k: float = 8.0
    base: float = 1.0
    name: str = "multiplicative_threshold"

    def rho(self, depth, roughness, effort):
        depth = np.asarray(depth, dtype=float)
        roughness = np.asarray(roughness, dtype=float)
        e = np.asarray(effort, dtype=float)
        f_e = 1.0 - (1.0 - self.p_unit) ** e
        terrain = (1.0 - depth) ** self.a_d * (1.0 - roughness) ** self.a_r
        gate = _sigmoid(self.k * (self.threshold - depth * roughness))
        return np.clip(self.base * f_e * terrain * gate, 0.0, 1.0)


# Registry used by the notebook and the Streamlit app.
DETECTORS: dict[str, DetectionModel] = {
    "D1_saturating_exponential": SaturatingExponentialDetector(),
    "D2_independent_passes": IndependentPassesDetector(),
    "D3_logistic": LogisticDetector(),
    "D4_multiplicative_threshold": MultiplicativeThresholdDetector(),
}
