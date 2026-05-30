"""Prior distributions over the grid.

Physical model
--------------
The aircraft explodes at ``x_E = (7, 20)``. From there:

  * fragments keep moving in roughly the plane-velocity direction (witness 1),
  * a wind acts on them while they fall,
  * once they hit water they are pushed by the surface current,
  * witness 2 says the object falls "somewhat farther along the trajectory
    and slightly off to one side".

Modelling choice: we treat the expected impact point on the sea surface as

    mu = x_E + tau_fall * (v_plane + w_wind * v_wind)

with ``tau_fall`` an unknown fall-time scalar and ``w_wind`` a wind-coupling
factor. After impact, the object can be displaced by the water drift while
sinking:

    mu_seabed = mu + tau_drift * v_drift

We model the prior over the seabed location as an isotropic (or oriented)
2-D Gaussian centred at ``mu_seabed``, possibly modulated by environmental
plausibility (e.g. shallow areas are slightly preferred because they imply
shorter drift while sinking — this is a soft prior, not a fact).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulator.grid import GridInfo


# ----- default accident parameters (from deliverable2.ipynb) --------------
ACCIDENT_POINT = np.array([7.0, 20.0])
V_PLANE = np.array([6.0, -3.5])
V_WIND = np.array([-1.0, -1.5])
V_DRIFT = np.array([0.5, -1.5])


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0:
        raise ValueError("Prior sums to zero; cannot normalize.")
    return p / total


def uniform_prior(grid: GridInfo) -> np.ndarray:
    return np.full(grid.n_cells, 1.0 / grid.n_cells)


@dataclass
class DriftParams:
    """Parameters controlling the impact-point displacement model."""

    tau_fall: float = 1.0       # seconds of free flight (scaled to grid units)
    w_wind: float = 0.5         # how strongly the wind couples while falling
    tau_drift: float = 1.0      # how long the object drifts while sinking
    sigma_x: float = 6.0        # along-trajectory uncertainty (grid units)
    sigma_y: float = 4.0        # cross-trajectory uncertainty


def expected_landing(p: DriftParams = DriftParams()) -> np.ndarray:
    mu = ACCIDENT_POINT + p.tau_fall * (V_PLANE + p.w_wind * V_WIND)
    mu = mu + p.tau_drift * V_DRIFT
    return mu


def _oriented_gaussian(
    grid: GridInfo,
    mu: np.ndarray,
    direction: np.ndarray,
    sigma_along: float,
    sigma_cross: float,
) -> np.ndarray:
    """Anisotropic Gaussian aligned with ``direction``.

    Larger uncertainty along the trajectory captures the fact that we don't
    know exactly how far the object travelled; smaller perpendicular
    uncertainty reflects that witness 2 only allows a slight lateral offset.
    """
    d_norm = direction / (np.linalg.norm(direction) + 1e-12)
    n_norm = np.array([-d_norm[1], d_norm[0]])  # 90 deg rotation

    dx = grid.x - mu[0]
    dy = grid.y - mu[1]
    a = dx * d_norm[0] + dy * d_norm[1]    # coord along direction
    c = dx * n_norm[0] + dy * n_norm[1]    # coord perpendicular

    z = (a / sigma_along) ** 2 + (c / sigma_cross) ** 2
    return np.exp(-0.5 * z)


def drift_prior(
    grid: GridInfo,
    params: DriftParams = DriftParams(),
) -> np.ndarray:
    """Anisotropic Gaussian prior centred on the expected landing point.

    Uses only the physical vectors (no witness information beyond the
    direction of motion implied by the velocity vector).
    """
    mu = expected_landing(params)
    direction = V_PLANE + params.w_wind * V_WIND  # the trajectory tilt
    p = _oriented_gaussian(grid, mu, direction, params.sigma_x, params.sigma_y)
    return _normalize(p)


def drift_prior_with_witnesses(
    grid: GridInfo,
    params: DriftParams = DriftParams(),
    forward_bias: float = 1.5,
    lateral_offset: float = 1.5,
    lateral_side: int = +1,  # +1 = above trajectory, -1 = below
) -> np.ndarray:
    """Drift prior with witness statements baked in.

    Witness 1: the object kept moving forward along the trajectory
        -> shift the mean farther along ``direction`` by ``forward_bias``.
    Witness 2: it fell slightly off to one side
        -> shift the mean perpendicular to ``direction`` by ``lateral_offset``.

    These shifts are intentionally small relative to ``sigma_x, sigma_y``,
    consistent with the witness statements being qualitative.
    """
    mu = expected_landing(params)
    direction = V_PLANE + params.w_wind * V_WIND
    d_norm = direction / (np.linalg.norm(direction) + 1e-12)
    n_norm = np.array([-d_norm[1], d_norm[0]])

    mu = mu + forward_bias * d_norm + lateral_side * lateral_offset * n_norm

    p = _oriented_gaussian(grid, mu, direction, params.sigma_x, params.sigma_y)
    return _normalize(p)


def sinking_adjusted_prior(
    base_prior: np.ndarray,
    grid: GridInfo,
    depth_bias: float = 0.0,
) -> np.ndarray:
    """Multiplicatively reweight a prior by a depth-related factor.

    ``depth_bias > 0`` increases the prior weight of deep cells (an object
    that has been drifting may end up in any depth — but if it's heavy it
    sinks to the deepest available cell). ``depth_bias < 0`` prefers shallow
    cells. ``0`` leaves the prior unchanged.

    The factor is ``exp(depth_bias * depth)``, so the prior is still
    well-defined for any real depth_bias.
    """
    factor = np.exp(depth_bias * grid.depth)
    return _normalize(base_prior * factor)


# ----------------------- physics-aware Monte Carlo prior ----------------- #
def physics_prior(
    grid: GridInfo,
    n_samples: int = 2000,
    seed: int = 0,
    *,
    tau_fall_log_sigma: float = 0.30,
    w_wind_low: float = 0.20,
    w_wind_high: float = 0.80,
    tau_drift_log_sigma: float = 0.30,
    sigma_along_range: tuple[float, float] = (4.0, 8.0),
    sigma_cross_range: tuple[float, float] = (3.0, 5.0),
    forward_bias_mean: float = 1.5,
    forward_bias_std: float = 1.0,
    lateral_offset_std: float = 1.5,
    uniform_weight: float = 0.05,
) -> np.ndarray:
    """Monte Carlo prior with explicit randomness over the physics constants.

    Conceptual note: the prior is the distribution over *where the object
    is*. By construction it must depend only on causes — the physics that
    took the object from the explosion point to the seabed: plane
    velocity, wind, water drift, and the witnesses. It must *not* depend
    on cell properties like depth or roughness, because those are
    properties of the landing location, not causes of it. (A deep cell
    does not "attract" debris simply because it is deep.) Depth and
    roughness do enter the model, but on the *detectability* side
    (rho_{t,j} in DetectionModel), not on the prior.

    Motivation
    ----------
    The simple ``drift_prior`` plugs in *deterministic* values for
    ``tau_fall``, ``w_wind``, ``tau_drift`` and the two witness shifts,
    which makes the resulting Gaussian unrealistically narrow. Real
    Bayesian search theory (Stone, *Bayesian search for the Scorpion /
    Air France 447 / MH370*) marginalises over those nuisance parameters.

    Sampling distributions (defaults justified inline)
    --------------------------------------------------
    * ``tau_fall ~ LogNormal(mean=0, sigma=0.30)`` → roughly 0.55…1.80,
      centred on 1 (the deterministic baseline).
    * ``w_wind ~ U(0.2, 0.8)`` — wind coupling for falling debris is
      not pinned down at all; large range.
    * ``tau_drift ~ LogNormal(mean=0, sigma=0.30)`` — surface drift time
      until the object sinks/settles.
    * ``sigma_along ~ U(4, 8)``, ``sigma_cross ~ U(3, 5)`` — uncertainty
      about how spread out the resulting impact ellipse is.
    * ``forward_bias ~ N(1.5, 1.0)`` — witness 1 says "kept moving
      forward", but how much is unknown; can be 0 or negative.
    * ``lateral_offset ~ N(0, 1.5)`` — witness 2 says "slightly off to
      one side" but does not specify the side, so the offset is
      symmetric around 0 (mixture of left/right interpretations).

    Uniform component
    -----------------
    Mixes in a ``uniform_weight`` fraction of the uniform distribution
    so the prior never collapses to exactly zero outside the drift
    cone. This is the standard way to acknowledge "our physics model
    could be wrong" without giving away the informed prior entirely.
    """
    rng = np.random.default_rng(seed)
    accum = np.zeros(grid.n_cells)

    for _ in range(n_samples):
        tau_fall = float(rng.lognormal(mean=0.0, sigma=tau_fall_log_sigma))
        w_wind = float(rng.uniform(w_wind_low, w_wind_high))
        tau_drift = float(rng.lognormal(mean=0.0, sigma=tau_drift_log_sigma))
        sigma_along = float(rng.uniform(*sigma_along_range))
        sigma_cross = float(rng.uniform(*sigma_cross_range))
        forward_bias = float(rng.normal(forward_bias_mean, forward_bias_std))
        lateral_offset = float(rng.normal(0.0, lateral_offset_std))

        direction = V_PLANE + w_wind * V_WIND
        mu = ACCIDENT_POINT + tau_fall * direction + tau_drift * V_DRIFT
        d_norm = direction / (np.linalg.norm(direction) + 1e-12)
        n_norm = np.array([-d_norm[1], d_norm[0]])
        mu = mu + forward_bias * d_norm + lateral_offset * n_norm

        accum += _oriented_gaussian(grid, mu, direction, sigma_along, sigma_cross)

    informed = _normalize(accum / n_samples)
    if uniform_weight <= 0:
        return informed
    return _normalize(
        (1.0 - uniform_weight) * informed + uniform_weight / grid.n_cells
    )
