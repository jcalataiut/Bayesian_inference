"""Prior specifications for the logistic prior on Z.

The prior over the cell containing the object is

    pi_j(beta, tau_fall, tau_drift) = softmax_j(eta_j)

with

    eta_j = b0 + b1 * d_long + b2 * d_trans - b3 * d_long^2 - b4 * d_trans^2.

Different ``PriorSpec`` instances vary the prior over ``(beta, tau)`` along
three axes:

    * ``form``      : "linear" / "quadratic" / "mixed" controls which betas
                      are free and which are forced to zero.
    * ``witnesses`` : True / False controls whether the linear betas (b1, b2)
                      have prior means consistent with the witness statements
                      (forward bias from witness 1, lateral offset from
                      witness 2). False -> linear-beta priors centered at 0.
    * ``info``      : "informative" / "weak" / "vague" scales the prior
                      standard deviations.

Each spec yields the hyperparameters needed to build a PyMC model in
``modeling.pymc_model``. Numpy-only evaluators are also provided so we can
visualise the prior (sample from it, plot heatmaps, compute prior-predictive
distributions) without ever invoking PyMC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from .features import (
    cell_distances,
    evaluate_pi,
    expected_landing,
    trajectory_axes,
)


PriorForm = Literal["linear", "quadratic", "mixed"]
PriorInfo = Literal["informative", "weak", "vague"]


# Witness-implied prior means for (b1, b2):
#   * witness 1: object kept moving forward  -> push pi mass along +d_norm
#   * witness 2: slightly off to one side    -> push pi mass along +n_norm
# Magnitudes are small relative to the spread (witnesses are qualitative).
WITNESS_B1_MEAN = 0.10   # forward bias coefficient (positive -> +d_long preferred)
WITNESS_B2_MEAN = 0.07   # lateral offset coefficient

# Reference longitudinal / transverse spreads (cells). Used to set the prior
# mean for the HalfNormal on (b3, b4) so that the implied Gaussian has
# sigma_long ~= 6 and sigma_trans ~= 4 -- consistent with the deterministic
# prior used in the legacy gemini_proyect baseline.
SIGMA_LONG_REF = 6.0
SIGMA_TRANS_REF = 4.0
B3_REF = 1.0 / (2.0 * SIGMA_LONG_REF ** 2)
B4_REF = 1.0 / (2.0 * SIGMA_TRANS_REF ** 2)


@dataclass
class PriorSpec:
    """Hyperparameters for one prior variant.

    The "uniform_mix" field controls a mixture with the uniform distribution
    that gives every cell a guaranteed minimum probability mass. Useful when
    the truth may lie in the tail of the parametric prior (where pure softmax
    would assign vanishing mass). The PyMC model and the prior-predictive
    evaluation both honour this mixture::

        pi_mixed = (1 - uniform_mix) * softmax(eta) + uniform_mix / N
    """

    name: str
    form: PriorForm
    witnesses: bool
    info: PriorInfo

    # Filled by __post_init__.
    b0_sigma: float = 5.0
    b1_mean: float = 0.0
    b2_mean: float = 0.0
    b1_sigma: float = 0.1
    b2_sigma: float = 0.1
    b3_sigma: float = B3_REF
    b4_sigma: float = B4_REF
    b3_active: bool = True
    b4_active: bool = True
    b1_active: bool = True
    b2_active: bool = True
    tau_fall_sigma: float = 0.30
    tau_drift_sigma: float = 0.30
    uniform_mix: float = 0.0          # fraction of uniform component
    likelihood_temperature: float = 1.0  # >1 amplifies likelihood updates

    # Hyperparameter overrides (None → use defaults from class-level constants).
    sigma_long_ref: float | None = None
    sigma_trans_ref: float | None = None

    def __post_init__(self) -> None:
        # Activation pattern by form.
        if self.form == "linear":
            self.b3_active = False
            self.b4_active = False
        elif self.form == "quadratic":
            self.b1_active = False
            self.b2_active = False
        elif self.form == "mixed":
            pass
        else:
            raise ValueError(f"Unknown form {self.form!r}")

        # Witness-driven prior means for the linear betas.
        if self.witnesses:
            self.b1_mean = WITNESS_B1_MEAN
            self.b2_mean = WITNESS_B2_MEAN
        else:
            self.b1_mean = 0.0
            self.b2_mean = 0.0

        # Informativeness scaling. Multiplies all *sigmas* used as priors.
        scale = {"informative": 1.0, "weak": 3.0, "vague": 10.0}[self.info]
        self.b1_sigma = 0.10 * scale
        self.b2_sigma = 0.10 * scale
        # Allow per-spec override of the reference spread used to set b3/b4.
        s_long = self.sigma_long_ref if self.sigma_long_ref is not None else SIGMA_LONG_REF
        s_trans = self.sigma_trans_ref if self.sigma_trans_ref is not None else SIGMA_TRANS_REF
        b3_ref_local = 1.0 / (2.0 * s_long ** 2)
        b4_ref_local = 1.0 / (2.0 * s_trans ** 2)
        self.b3_sigma = b3_ref_local * scale
        self.b4_sigma = b4_ref_local * scale

    # ------------------------ pure-numpy evaluators ------------------------ #
    def sample_beta(self, rng: np.random.Generator) -> np.ndarray:
        b0 = rng.normal(0.0, self.b0_sigma)
        b1 = rng.normal(self.b1_mean, self.b1_sigma) if self.b1_active else 0.0
        b2 = rng.normal(self.b2_mean, self.b2_sigma) if self.b2_active else 0.0
        b3 = abs(rng.normal(0.0, self.b3_sigma)) if self.b3_active else 0.0
        b4 = abs(rng.normal(0.0, self.b4_sigma)) if self.b4_active else 0.0
        return np.array([b0, b1, b2, b3, b4])

    def sample_tau(self, rng: np.random.Generator) -> tuple[float, float]:
        tau_fall = float(rng.lognormal(mean=0.0, sigma=self.tau_fall_sigma))
        tau_drift = float(rng.lognormal(mean=0.0, sigma=self.tau_drift_sigma))
        return tau_fall, tau_drift

    def prior_predictive_pi(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_samples: int = 2000,
        seed: int = 0,
    ) -> np.ndarray:
        """Monte Carlo estimate of E_{beta, tau ~ prior}[pi_j]   (with mix).

        Used to draw the prior heatmap before any mission is run. Honours
        ``self.uniform_mix`` so the heatmap reflects the same mixture that
        the PyMC model uses as a prior on cells.
        """
        rng = np.random.default_rng(seed)
        accum = np.zeros_like(x, dtype=float)
        for _ in range(n_samples):
            beta = self.sample_beta(rng)
            tau_fall, tau_drift = self.sample_tau(rng)
            accum += evaluate_pi(beta, tau_fall, tau_drift, x, y)
        pi = accum / n_samples
        if self.uniform_mix > 0:
            uniform = 1.0 / pi.size
            pi = (1.0 - self.uniform_mix) * pi + self.uniform_mix * uniform
            pi = pi / pi.sum()
        return pi


# --------------------------------------------------------------------------- #
# Registry: 8 priors covering the 3 axes of variation.
# Chosen to be a manageable subset of the full 3x2x2 = 12 grid: we include
# the three forms x both witness settings under "informative", plus two
# informativeness variations on the strongest (mixed + witnesses) baseline.
# --------------------------------------------------------------------------- #
PRIORS: dict[str, PriorSpec] = {
    # NOTE on excluded P1/P2: linear-form priors (no quadratic terms) were
    # implemented and tested in earlier sweeps. They were removed from the
    # registry because exp(linear)/softmax does NOT concentrate mass on a
    # peak -- its maximum lies at one of the grid corners, producing a
    # degenerate prior that fails to converge in validation (detection
    # rate ~0%). They are documented in the report as an anti-baseline
    # that motivates the quadratic family but kept out of the active
    # registry to avoid noise in figures and ranking tables.
    "P3_quadratic_nowit_informative": PriorSpec(
        name="P3_quadratic_nowit_informative",
        form="quadratic",
        witnesses=False,
        info="informative",
    ),
    "P4_quadratic_wit_informative": PriorSpec(
        name="P4_quadratic_wit_informative",
        form="quadratic",
        witnesses=True,
        info="informative",
    ),
    "P5_mixed_nowit_informative": PriorSpec(
        name="P5_mixed_nowit_informative",
        form="mixed",
        witnesses=False,
        info="informative",
    ),
    "P6_mixed_wit_informative": PriorSpec(
        name="P6_mixed_wit_informative",
        form="mixed",
        witnesses=True,
        info="informative",
    ),
    "P7_mixed_wit_weak": PriorSpec(
        name="P7_mixed_wit_weak",
        form="mixed",
        witnesses=True,
        info="weak",
    ),
    "P8_mixed_wit_vague": PriorSpec(
        name="P8_mixed_wit_vague",
        form="mixed",
        witnesses=True,
        info="vague",
    ),
    "P9_mixed_wit_uniform30": PriorSpec(
        name="P9_mixed_wit_uniform30",
        form="mixed",
        witnesses=True,
        info="informative",
        uniform_mix=0.30,
    ),
    "P10_mixed_wit_uniform50_temp2": PriorSpec(
        name="P10_mixed_wit_uniform50_temp2",
        form="mixed",
        witnesses=True,
        info="informative",
        uniform_mix=0.50,
        likelihood_temperature=2.0,
    ),
    # Operational priors for the progressive-drift strategy. These are
    # centered on the expected first water-contact point mu(tau), not on the
    # tail. If conservative missions fail, the *strategy* moves the search
    # center along drift; the prior itself stays honest about the initial
    # impact hypothesis.
    # ----- P11-P14: derived from the sweep, NOT hand-picked -----
    # Procedure (see notebook 00 §6.1):
    #   1. Run a 10-round Empirical Bayes sweep with backend=cell_update,
    #      detector=D1, strategy=FIXED, train/val split. Result CSV:
    #      results/iterative_eb_10r_val_summary.csv.
    #   2. Identify the top EB winner in validation -> EB_r01_bw2.5_u10
    #      (built from 62 detected cells in round 0, bandwidth 2.5,
    #      uniform_mix 0.10).
    #   3. Strip the uniform component and fit an anisotropic Gaussian to
    #      the pure KDE part, projecting on the trajectory axes:
    #         sigma_long_EB  = 5.52   (~ uncertainty along plane direction)
    #         sigma_trans_EB = 4.11   (~ uncertainty perpendicular)
    #      (a secondary EB winner with bw=1.75 in round 4 gives 5.24/3.86;
    #      both are within 6%, so the fit is robust).
    #   4. Define P11-P14 as parametric variations around (sigma_long_EB,
    #      sigma_trans_EB) covering the spectrum tight -> wide:
    #         P11_tight    : scale 0.70 of EB fit
    #         P12_balanced : exact EB fit (the "central" winner)
    #         P13_robust   : same sigmas + uniform_mix=0.10
    #                         (mirrors EB_r01_bw2.5_u10 verbatim)
    #         P14_wide     : scale 1.50 of EB fit
    # The form is "quadratic without witnesses" because that is the family
    # that consistently topped the validation ranking (see notebook 00 §5.1).
    "P11_impact_tight": PriorSpec(
        name="P11_impact_tight",
        form="quadratic",
        witnesses=False,
        info="informative",
        sigma_long_ref=3.9,   # 5.52 * 0.70 (rounded)
        sigma_trans_ref=2.9,  # 4.11 * 0.70
    ),
    "P12_impact_balanced": PriorSpec(
        name="P12_impact_balanced",
        form="quadratic",
        witnesses=False,
        info="informative",
        sigma_long_ref=5.5,   # exact EB winner Gaussian fit
        sigma_trans_ref=4.1,
    ),
    "P13_impact_robust_uniform10": PriorSpec(
        name="P13_impact_robust_uniform10",
        form="quadratic",
        witnesses=False,
        info="informative",
        sigma_long_ref=5.5,
        sigma_trans_ref=4.1,
        uniform_mix=0.10,        # matches u of the EB winner
        likelihood_temperature=1.5,
    ),
    "P14_impact_wide": PriorSpec(
        name="P14_impact_wide",
        form="quadratic",
        witnesses=False,
        info="informative",
        sigma_long_ref=8.3,   # 5.52 * 1.50
        sigma_trans_ref=6.2,  # 4.11 * 1.50
    ),
}


# --------------------------------------------------------------------------- #
# Empirical Bayes: build a prior from cells where the bomb was actually found
# in a previous Monte Carlo round. Implements the user's idea of "use round-1
# successful detections to define a round-2 prior".
# --------------------------------------------------------------------------- #
@dataclass
class EmpiricalPriorSpec:
    """Non-parametric prior built from a set of detection-success cells.

    Given a list of cell indices where previous campaigns successfully
    located the bomb, we apply Gaussian KDE smoothing on the grid to obtain
    a continuous probability surface. Optionally mixed with a uniform
    component for robustness.

    Duck-types ``PriorSpec`` for use in ``experiments.fast_monte_carlo``:
    only the ``name``, ``uniform_mix`` and ``prior_predictive_pi`` interface
    are required there. ``likelihood_temperature`` defaults to 1.0 since
    this prior is not used inside PyMC (no parametric β to update).
    """

    name: str
    detected_cells: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    bandwidth: float = 2.0
    uniform_mix: float = 0.05
    likelihood_temperature: float = 1.0

    def __post_init__(self) -> None:
        if len(self.detected_cells) == 0:
            raise ValueError("EmpiricalPriorSpec needs at least one detected cell.")
        self._pi = self._compute_pi()

    def _compute_pi(self) -> np.ndarray:
        pi = np.zeros_like(self.grid_x, dtype=float)
        bw2 = 2.0 * self.bandwidth ** 2
        for c in self.detected_cells:
            dx = self.grid_x - self.grid_x[c]
            dy = self.grid_y - self.grid_y[c]
            pi += np.exp(-(dx ** 2 + dy ** 2) / bw2)
        pi = pi / pi.sum()
        if self.uniform_mix > 0:
            u = 1.0 / pi.size
            pi = (1.0 - self.uniform_mix) * pi + self.uniform_mix * u
            pi = pi / pi.sum()
        return pi

    def prior_predictive_pi(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return the precomputed empirical prior (signature matches PriorSpec)."""
        return self._pi.copy()


# --------------------------------------------------------------------------- #
# Helper to build a small hyperparameter grid of priors on the fly. Used by
# the comparator to evaluate sensitivity to (sigma_long, sigma_trans, b1_mean).
# --------------------------------------------------------------------------- #
def make_grid_priors(
    sigma_longs=(4.0, 6.0, 8.0),
    sigma_transs=(3.0, 4.0, 5.0),
    b1_means=(0.0, 0.10, 0.20),
    uniform_mixes=(0.0, 0.2),
) -> dict[str, PriorSpec]:
    """Cartesian product of hyperparameters → dict of PriorSpec.

    Useful for systematic sensitivity analysis when the deliverable wants to
    justify a particular prior choice. Each generated spec is "mixed +
    witnesses + informative", varying only the geometry hyperparameters
    plus an optional uniform mixture.
    """
    out: dict[str, PriorSpec] = {}
    for sl in sigma_longs:
        for st in sigma_transs:
            for b1 in b1_means:
                for um in uniform_mixes:
                    name = f"G_sl{sl:.0f}_st{st:.0f}_b1{int(b1*100):02d}_u{int(um*100):02d}"
                    spec = PriorSpec(
                        name=name,
                        form="mixed",
                        witnesses=True,
                        info="informative",
                        uniform_mix=um,
                        sigma_long_ref=sl,
                        sigma_trans_ref=st,
                    )
                    spec.b1_mean = b1
                    out[name] = spec
    return out


def list_priors() -> Iterable[str]:
    return PRIORS.keys()
