"""Streamlit app for the Bayesian search deliverable.

A literal interactive walkthrough of the four-axis pipeline plus a one-click
'operational mode' that runs the recommended configuration (the same flow as
``notebook 04_operational_summary``) and shows the full mission sequence under
the assumption of repeated non-detection.

Run from the repository root:
    streamlit run update_version/app/streamlit_app.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Make ``update_version`` importable when ``streamlit run`` launches the app.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from update_version.simulator.grid import load_grid           # noqa: E402
from update_version.simulator.environment import SearchEnvironment   # noqa: E402
from update_version.simulator.true_detection import TrueDetector     # noqa: E402
from update_version.modeling.features import (                # noqa: E402
    ACCIDENT_POINT, V_PLANE, V_WIND, V_DRIFT, ALPHA_WIND,
    expected_landing, trajectory_axes, cell_distances,
)
from update_version.modeling.priors_logistic import PRIORS    # noqa: E402
from update_version.modeling.detection import DETECTORS       # noqa: E402
from update_version.strategies.next_mission import (          # noqa: E402
    hdr_mask, propose_via_hdr_topk, propose_adaptive, propose_info_gain,
    propose_progressive_scenario_search,
    progressive_scenario_prior, apply_progressive_scenario_escape,
)


# --------------------------------------------------------------------------- #
# App-wide configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Bayesian Search — Deliverable 2",
    layout="wide",
)

BUDGET_CAP = 530
DATA_CSV = _HERE.parent.parent / "data" / "grid_dataset.csv"


@st.cache_resource(show_spinner=False)
def _load_grid():
    return load_grid(DATA_CSV)


@st.cache_data(show_spinner=False)
def _prior_pi_cached(name: str, n_samples: int = 700, seed: int = 0) -> np.ndarray:
    """Cached prior-predictive evaluation (one per (name, n_samples, seed))."""
    g = _load_grid()
    return PRIORS[name].prior_predictive_pi(g.x, g.y, n_samples=n_samples, seed=seed)


@st.cache_data(show_spinner=False)
def _rho_cached(detector_name: str, effort: int) -> np.ndarray:
    g = _load_grid()
    return DETECTORS[detector_name].rho(g.depth, g.roughness, effort)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def heatmap(ax, values, grid, *, cmap="magma", title=None, vmin=None, vmax=None,
            mark_mu=True, true_cell=None, rect=None):
    """Standard heatmap with mu and (optional) rectangle / true cell overlays."""
    im = ax.imshow(grid.reshape_2d(values), origin="lower",
                   extent=[0, grid.Nx, 0, grid.Ny], cmap=cmap,
                   vmin=vmin, vmax=vmax)
    if mark_mu:
        mu = expected_landing(1.0, 1.0)
        ax.scatter(*mu, s=60, facecolors="none", edgecolors="cyan",
                   linewidths=1.5, label="μ")
    if true_cell is not None:
        ax.scatter(grid.x[true_cell], grid.y[true_cell],
                   color="lime", marker="X", s=85, edgecolor="black",
                   label="true cell")
    if rect is not None:
        x0, x1, y0, y1 = rect
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    edgecolor="white", facecolor="none", lw=2))
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)


def entropy_nats(p: np.ndarray) -> float:
    p = np.clip(p, 1e-300, None)
    return float(-(p * np.log(p)).sum())


def direct_update(posterior: np.ndarray, rec: Any, detector,
                   grid, temperature: float = 1.5):
    rho = np.asarray(detector.rho(grid.depth, grid.roughness, rec.effort))
    covered = grid.coverage_mask(rec.x_min, rec.x_max, rec.y_min, rec.y_max)
    q = np.where(covered, rho, 0.0)
    q = np.clip(q, 1e-12, 1.0 - 1e-12)
    L = (q if rec.s_t else 1.0 - q) ** temperature
    post = posterior * L
    post = post / post.sum()
    return post, covered, q, L


# --------------------------------------------------------------------------- #
# Sidebar — global configuration
# --------------------------------------------------------------------------- #
def render_sidebar():
    st.sidebar.title("Bayesian Search")
    st.sidebar.caption("Deliverable 2 · University of Barcelona")

    st.sidebar.markdown("### Global configuration")
    prior_name = st.sidebar.selectbox(
        "Prior", list(PRIORS.keys()),
        index=list(PRIORS.keys()).index("P12_impact_balanced"),
        help="The prior over the aircraft location. P11-P14 are the operational priors derived from the sweep.",
    )
    detector_name = st.sidebar.selectbox(
        "Detector", list(DETECTORS.keys()), index=0,
        help="The detection model ρ(depth, roughness, effort).",
    )
    update_temperature = st.sidebar.slider(
        "Update temperature T", 0.5, 3.0, 1.5, 0.1,
        help="Exponent on the per-cell likelihood. T > 1 makes failed missions reduce in-rectangle mass more aggressively.",
    )
    budget = st.sidebar.number_input(
        "Budget (cap = 530)", min_value=50, max_value=BUDGET_CAP,
        value=BUDGET_CAP, step=10,
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Pages**")
    st.sidebar.markdown(
        "1. Overview & physics\n"
        "2. Priors\n"
        "3. Detectors\n"
        "4. Strategy preview\n"
        "5. Interactive simulator\n"
        "6. **Final operational model**"
    )
    return prior_name, detector_name, update_temperature, budget


# --------------------------------------------------------------------------- #
# TAB 1 — Overview & physics
# --------------------------------------------------------------------------- #
def tab_overview():
    grid = _load_grid()
    st.header("1 · Overview & physics")
    st.markdown(
        "This app walks through the Bayesian search pipeline interactively. "
        "The problem: localise an aircraft in a $50 \\times 35$ grid using up to "
        "$B = 530$ budget units, where each mission covers an axis-aligned "
        "rectangle and a noisy detector returns $s_t \\in \\{0, 1\\}$."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Physical inputs")
        st.markdown(
            f"- Accident location $x_E = {tuple(ACCIDENT_POINT.tolist())}$\n"
            f"- Aircraft velocity $v_{{\\mathrm{{plane}}}} = {tuple(V_PLANE.tolist())}$\n"
            f"- Wind $v_{{\\mathrm{{wind}}}} = {tuple(V_WIND.tolist())}$\n"
            f"- Drift $v_{{\\mathrm{{drift}}}} = {tuple(V_DRIFT.tolist())}$\n"
            f"- Wind coupling during fall $\\alpha_{{\\mathrm{{wind}}}} = {ALPHA_WIND}$"
        )
        st.subheader("Expected impact point μ")
        tau_fall = st.slider("τ_fall", 0.5, 1.5, 1.0, 0.05, key="overview_tau_fall")
        tau_drift = st.slider("τ_drift", 0.5, 1.5, 1.0, 0.05, key="overview_tau_drift")
        mu = expected_landing(tau_fall, tau_drift)
        d_norm, n_norm = trajectory_axes()
        st.markdown(
            f"$\\mu(\\tau_{{\\mathrm{{fall}}}}={tau_fall}, \\tau_{{\\mathrm{{drift}}}}={tau_drift}) "
            f"= ({mu[0]:.2f},\\,{mu[1]:.2f})$"
        )
        st.markdown(
            f"$\\hat d = {tuple(np.round(d_norm, 3).tolist())}$ "
            f"(along trajectory), "
            f"$\\hat n = {tuple(np.round(n_norm, 3).tolist())}$ "
            f"(perpendicular)"
        )

    with c2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(grid.x, grid.y, s=4, alpha=0.15, color="lightgray")
        ax.scatter(*ACCIDENT_POINT, s=110, color="white",
                    edgecolor="black", label="x_E", zorder=5)
        ax.scatter(*mu, s=130, color="red", marker="x", label="μ",
                    zorder=5, linewidths=2.5)
        ax.arrow(mu[0], mu[1], d_norm[0]*6, d_norm[1]*6,
                  width=0.1, color="tab:blue", length_includes_head=True)
        ax.arrow(mu[0], mu[1], n_norm[0]*4, n_norm[1]*4,
                  width=0.07, color="tab:orange", length_includes_head=True)
        ax.text(mu[0] + d_norm[0]*6.5, mu[1] + d_norm[1]*6.5,
                 "d_long", color="tab:blue", fontsize=10)
        ax.text(mu[0] + n_norm[0]*4.5, mu[1] + n_norm[1]*4.5,
                 "d_trans", color="tab:orange", fontsize=10)
        ax.set_xlim(-1, grid.Nx); ax.set_ylim(-1, grid.Ny)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Accident geometry"); ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    st.subheader("Cell environment")
    st.markdown(
        "The grid carries two physical fields per cell: **depth** and **roughness**, "
        "both normalised to $[0, 1]$. They feed the detection model in tab 3."
    )
    fig2, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, vals, title, cmap in [(axes[0], grid.depth, "depth", "Blues"),
                                    (axes[1], grid.roughness, "roughness", "Oranges")]:
        im = ax.imshow(grid.reshape_2d(vals), origin="lower",
                        extent=[0, grid.Nx, 0, grid.Ny], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig2)


# --------------------------------------------------------------------------- #
# TAB 2 — Priors
# --------------------------------------------------------------------------- #
def tab_priors(prior_name: str):
    grid = _load_grid()
    st.header("2 · Priors over the unknown cell Z")
    st.markdown(
        "Each prior assigns a probability to every cell. The operational priors "
        "**P11–P14** are derived from the sweep (see notebook 00 §5); the others "
        "are the candidates explored during model selection."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Selected prior")
        spec = PRIORS[prior_name]
        st.markdown(f"**`{prior_name}`**")
        st.json({
            "form": spec.form,
            "witnesses": spec.witnesses,
            "info": spec.info,
            "sigma_long_ref": spec.sigma_long_ref,
            "sigma_trans_ref": spec.sigma_trans_ref,
            "uniform_mix": spec.uniform_mix,
            "likelihood_temperature": spec.likelihood_temperature,
            "b1_mean": spec.b1_mean,
            "b2_mean": spec.b2_mean,
        }, expanded=False)
        pi = _prior_pi_cached(prior_name)
        st.markdown(
            f"- max $\\pi_j$ = {pi.max():.4f}\n"
            f"- min $\\pi_j$ = {pi.min():.5f}\n"
            f"- entropy = {entropy_nats(pi):.2f} nats"
        )

    with c2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        heatmap(ax, pi, grid, cmap="magma",
                 title=f"Prior predictive π_j for {prior_name}")
        st.pyplot(fig)

    st.subheader("All registered priors (12)")
    fig2, axes = plt.subplots(3, 4, figsize=(17, 10))
    for ax, name in zip(axes.flat, PRIORS.keys()):
        pi_i = _prior_pi_cached(name)
        ax.imshow(grid.reshape_2d(pi_i), origin="lower",
                   extent=[0, grid.Nx, 0, grid.Ny], cmap="magma")
        ax.scatter(*expected_landing(1, 1), s=30, facecolors="none",
                   edgecolors="cyan", linewidths=1.2)
        marker = " [selected]" if name == prior_name else ""
        ax.set_title(name.replace("_", " ") + marker, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("All 12 priors (the current selection is tagged)", y=1.005)
    plt.tight_layout()
    st.pyplot(fig2)


# --------------------------------------------------------------------------- #
# TAB 3 — Detectors
# --------------------------------------------------------------------------- #
def tab_detectors(detector_name: str):
    grid = _load_grid()
    st.header("3 · Detection models ρ_j(depth, roughness, effort)")
    st.markdown(
        "Given that a rectangle covers cell $j$, the detector returns the "
        "probability of finding the aircraft. Four families are implemented; "
        "**D1 (saturating exponential)** is the operational default because it "
        "calibrates the qualitative pattern *high effort + easy cell → near-certain "
        "detection* from `previous_missions_reports.pdf`."
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        effort = st.slider("Search effort e", 1, 3, 2,
                            help="Higher effort → more cost but higher detection probability.")
        rho = _rho_cached(detector_name, effort)
        st.markdown(
            f"**Selected:** `{detector_name}`, effort = {effort}\n\n"
            f"- mean ρ = {rho.mean():.3f}\n"
            f"- max ρ = {rho.max():.3f}\n"
            f"- min ρ = {rho.min():.3f}"
        )
        st.markdown("Try the slider to see how effort changes the map.")
    with c2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        heatmap(ax, rho, grid, cmap="viridis", vmin=0, vmax=1,
                 title=f"ρ_j for {detector_name}, effort = {effort}", mark_mu=False)
        st.pyplot(fig)

    st.subheader("All four detector families at the selected effort")
    fig2, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, name in zip(axes, DETECTORS.keys()):
        r = _rho_cached(name, effort)
        ax.imshow(grid.reshape_2d(r), origin="lower",
                   extent=[0, grid.Nx, 0, grid.Ny], cmap="viridis",
                   vmin=0, vmax=1)
        marker = " [selected]" if name == detector_name else ""
        ax.set_title(name + marker, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Detector family comparison at effort = {effort}", y=1.02)
    plt.tight_layout()
    st.pyplot(fig2)


# --------------------------------------------------------------------------- #
# TAB 4 — Strategy preview
# --------------------------------------------------------------------------- #
def tab_strategy_preview(prior_name: str, detector_name: str, budget: float):
    grid = _load_grid()
    st.header("4 · Strategy preview")
    st.markdown(
        "Given a prior over Z and the detector, the strategy returns the next "
        "rectangle and effort. Compare the four strategies on the **same prior** "
        "with no observed history yet."
    )

    pi = _prior_pi_cached(prior_name)
    det = DETECTORS[detector_name]

    c = st.columns(4)
    config = {
        "hdr_mass": c[0].slider("HDR α (FIXED/INFO_GAIN)", 0.50, 0.99, 0.80, 0.05),
        "top_k": c[1].slider("top_k (FIXED/INFO_GAIN)", 3, 30, 10, 1),
        "exploration_mix": c[2].slider("exploration_mix", 0.0, 0.30, 0.05, 0.01),
        "scenario": c[3].selectbox("scenario (progressive)",
                                       ["mixture", "center", "right", "left", "strong_wind"]),
    }

    proposals = {}
    try:
        proposals["FIXED"] = propose_via_hdr_topk(
            pi, grid, det, budget,
            hdr_mass=config["hdr_mass"], top_k=config["top_k"],
            cost_exponent=0.7, exploration_mix=config["exploration_mix"],
        )
    except Exception as e:
        proposals["FIXED"] = e
    try:
        proposals["ADAPTIVE"] = propose_adaptive(pi, grid, det, budget)
    except Exception as e:
        proposals["ADAPTIVE"] = e
    try:
        proposals["INFO_GAIN"] = propose_info_gain(
            pi, grid, det, budget,
            hdr_mass=config["hdr_mass"], top_k=config["top_k"],
            exploration_mix=config["exploration_mix"], n_candidates=20,
        )
    except Exception as e:
        proposals["INFO_GAIN"] = e
    try:
        proposals["PROGRESSIVE"] = propose_progressive_scenario_search(
            pi, grid, det, budget, history=[], scenario=config["scenario"],
        )
    except Exception as e:
        proposals["PROGRESSIVE"] = e

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (name, prop) in zip(axes, proposals.items()):
        if isinstance(prop, Exception):
            ax.text(0.5, 0.5, f"{name}\nfailed:\n{type(prop).__name__}",
                     ha="center", va="center", fontsize=9, transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.imshow(grid.reshape_2d(pi), origin="lower",
                   extent=[0, grid.Nx, 0, grid.Ny], cmap="magma")
        ax.add_patch(plt.Rectangle((prop.x_min, prop.y_min),
                                     prop.x_max - prop.x_min,
                                     prop.y_max - prop.y_min,
                                     edgecolor="white", facecolor="none", lw=2))
        ax.scatter(*expected_landing(1, 1), s=30, facecolors="none",
                    edgecolors="cyan", linewidths=1.2)
        ax.set_title(
            f"{name}\nrect = ({prop.x_min:.1f}, {prop.x_max:.1f}) × "
            f"({prop.y_min:.1f}, {prop.y_max:.1f})\n"
            f"effort = {prop.effort}, cost = {prop.cost:.0f}",
            fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Mission proposed by each strategy (prior = {prior_name})",
                  y=1.05)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Proposal details")
    rows = []
    for name, prop in proposals.items():
        if isinstance(prop, Exception):
            rows.append({"strategy": name, "status": "FAILED"})
            continue
        rows.append({
            "strategy": name,
            "x_min": prop.x_min, "x_max": prop.x_max,
            "y_min": prop.y_min, "y_max": prop.y_max,
            "effort": prop.effort, "n_cells": prop.n_cells, "cost": prop.cost,
            "expected_detection": round(prop.expected_detection, 4),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")


# --------------------------------------------------------------------------- #
# TAB 5 — Interactive simulator
# --------------------------------------------------------------------------- #
def tab_simulator(prior_name: str, detector_name: str,
                   update_temperature: float, budget: float):
    grid = _load_grid()
    st.header("5 · Interactive simulator")
    st.markdown(
        "Plant a bomb, then run missions one by one. After every mission the "
        "posterior is updated via the direct cell-level rule with the chosen "
        "temperature. Use this tab to *feel* how the Bayesian update behaves."
    )

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = {}
    state = st.session_state.sim_state

    cfg_cols = st.columns([1, 1, 1, 1])
    plant_mode = cfg_cols[0].radio(
        "Plant bomb", ["from prior", "from progressive scenario", "manual cell"],
        horizontal=False,
    )
    scenario = cfg_cols[1].selectbox(
        "Search scenario (progressive)",
        ["mixture", "center", "right", "left", "strong_wind"], index=0,
    )
    plant_scenario = cfg_cols[2].selectbox(
        "Plant scenario (if 'from progressive scenario')",
        ["center", "right", "left", "strong_wind"], index=0,
    )
    manual_cell = cfg_cols[3].number_input(
        "Manual cell id (if 'manual cell')",
        min_value=0, max_value=grid.n_cells - 1, value=663, step=1,
    )
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=7,
                            step=1)

    action_cols = st.columns([1, 1, 1, 4])
    if action_cols[0].button("Plant / reset", width="stretch"):
        rng = np.random.default_rng(int(seed))
        if plant_mode == "from prior":
            pi0 = _prior_pi_cached(prior_name)
            true_cell = int(rng.choice(grid.n_cells, p=pi0))
        elif plant_mode == "from progressive scenario":
            fake_hist = [type("M", (), {"s_t": 0})() for _ in range(3)]
            pi_sc, *_ = progressive_scenario_prior(grid, fake_hist,
                                                     scenario=plant_scenario)
            true_cell = int(rng.choice(grid.n_cells, p=pi_sc))
        else:
            true_cell = int(manual_cell)

        env = SearchEnvironment(
            grid=grid, detector=TrueDetector(),
            budget_total=float(budget),
            rng=np.random.default_rng(int(seed)),
        )
        env.plant_object(cell_id=true_cell)

        state["env"] = env
        state["true_cell"] = true_cell
        state["posterior"] = _prior_pi_cached(prior_name).copy()
        state["history_rows"] = []
        state["trace"] = [state["posterior"].copy()]
        state["scenario"] = scenario
        st.success(
            f"Bomb planted at cell {true_cell}  "
            f"(x = {grid.x[true_cell]}, y = {grid.y[true_cell]})"
        )

    if "env" not in state:
        st.info("Click **Plant / reset** to start a simulation.")
        return

    env = state["env"]
    posterior = state["posterior"]

    step_btn = action_cols[1].button("Run next mission", width="stretch")
    run_to_end = action_cols[2].button("Run to detection / budget end",
                                          width="stretch")

    det = DETECTORS[detector_name]

    def do_one_mission():
        try:
            prop = propose_progressive_scenario_search(
                state["posterior"], grid, det, env.budget_remaining,
                history=env.history, scenario=state["scenario"],
            )
        except RuntimeError as e:
            st.warning(str(e))
            return False
        rec = env.run_mission(**prop.as_kwargs())
        new_post, covered, q, L = direct_update(
            state["posterior"], rec, det, grid, update_temperature,
        )
        if rec.s_t == 0:
            new_post, mix, c_long, c_xy, _ = apply_progressive_scenario_escape(
                new_post, grid, env.history, scenario=state["scenario"],
            )
        else:
            _, c_long, c_xy, _, _ = progressive_scenario_prior(
                grid, env.history, scenario=state["scenario"],
            )
        state["posterior"] = new_post
        state["trace"].append(new_post.copy())
        state["history_rows"].append({
            "mission_id": rec.mission_id,
            "x_min": rec.x_min, "x_max": rec.x_max,
            "y_min": rec.y_min, "y_max": rec.y_max,
            "effort": rec.effort, "cost": rec.cost, "s_t": rec.s_t,
            "budget_remaining": rec.budget_remaining,
            "contains_true": bool(covered[state["true_cell"]]),
            "center_x": float(c_xy[0]), "center_y": float(c_xy[1]),
        })
        return rec.s_t == 1

    if step_btn:
        if env.budget_remaining < 4:
            st.warning("Budget exhausted.")
        else:
            detected = do_one_mission()
            if detected:
                st.balloons()
                st.success("Detected!")

    if run_to_end:
        ndet = False
        while not ndet and env.budget_remaining >= 4:
            ndet = do_one_mission()
        if ndet:
            st.balloons(); st.success("Detected!")
        else:
            st.info("Budget exhausted without detection.")

    plot_cols = st.columns([3, 2])
    with plot_cols[0]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        last_rect = None
        if state["history_rows"]:
            r = state["history_rows"][-1]
            last_rect = (r["x_min"], r["x_max"], r["y_min"], r["y_max"])
        heatmap(ax, state["posterior"], grid, cmap="magma",
                 title=f"P(Z) after {len(state['history_rows'])} mission(s) "
                       f"— budget left {env.budget_remaining:.0f}/{env.budget_total:.0f}",
                 true_cell=state["true_cell"], rect=last_rect)
        for r in state["history_rows"]:
            col = "lime" if r["s_t"] == 1 else "white"
            ax.add_patch(plt.Rectangle((r["x_min"], r["y_min"]),
                                         r["x_max"] - r["x_min"],
                                         r["y_max"] - r["y_min"],
                                         edgecolor=col, facecolor="none",
                                         lw=1.2, alpha=0.6))
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig)
    with plot_cols[1]:
        st.subheader("Mission history")
        if state["history_rows"]:
            st.dataframe(pd.DataFrame(state["history_rows"]), width="stretch")
        else:
            st.caption("No missions run yet.")
        st.markdown(
            f"**Posterior entropy** = {entropy_nats(state['posterior']):.2f} nats"
        )
        st.markdown(
            f"**P(Z = true_cell)** = "
            f"{state['posterior'][state['true_cell']]:.5f}"
        )


# --------------------------------------------------------------------------- #
# TAB 6 — Final operational model
# --------------------------------------------------------------------------- #
@dataclass
class _RecLike:
    mission_id: int
    x_min: float; x_max: float
    y_min: float; y_max: float
    effort: int
    cost: float
    s_t: int
    budget_remaining: float


def _run_operational(prior_name: str, scenario: str, detector_name: str,
                      temperature: float, budget: float,
                      assume_no_detection: bool, plant_scenario: str,
                      seed: int, max_missions: int = 12):
    """Run the operational pipeline. Mirrors notebook 04 logic.

    If ``assume_no_detection`` is True, every mission outcome is forced to
    ``s_t = 0`` (regardless of where the bomb is) — answering 'what would the
    full sequence look like if we never detect?'. Otherwise the TrueDetector
    sampled outcome is used.
    """
    grid = _load_grid()
    det = DETECTORS[detector_name]

    # Plant a true cell from the chosen scenario (only used for "true sim" mode).
    rng = np.random.default_rng(seed)
    fake_hist = [type("M", (), {"s_t": 0})() for _ in range(3)]
    pi_plant, *_ = progressive_scenario_prior(grid, fake_hist,
                                                scenario=plant_scenario)
    true_cell = int(rng.choice(grid.n_cells, p=pi_plant))

    env = SearchEnvironment(
        grid=grid, detector=TrueDetector(),
        budget_total=float(budget), rng=np.random.default_rng(seed),
    )
    env.plant_object(cell_id=true_cell)

    posterior = _prior_pi_cached(prior_name).copy()
    trace = [posterior.copy()]
    rows = []
    mid = 0
    for _ in range(max_missions):
        if env.budget_remaining < 4:
            break
        try:
            prop = propose_progressive_scenario_search(
                posterior, grid, det, env.budget_remaining,
                history=env.history, scenario=scenario,
            )
        except RuntimeError:
            break
        if assume_no_detection:
            mid += 1
            rec = _RecLike(
                mission_id=mid,
                x_min=prop.x_min, x_max=prop.x_max,
                y_min=prop.y_min, y_max=prop.y_max,
                effort=int(prop.effort), cost=float(prop.cost),
                s_t=0,
                budget_remaining=float(env.budget_remaining - prop.cost),
            )
            # We still bookkeep cost & history via env (mark as failed).
            covered_true = grid.coverage_mask(prop.x_min, prop.x_max,
                                                prop.y_min, prop.y_max)[true_cell]
            env.history.append(rec)
            env._next_mission_id = mid + 1  # keep env counter in sync
        else:
            rec = env.run_mission(**prop.as_kwargs())
            covered_true = grid.coverage_mask(rec.x_min, rec.x_max,
                                                rec.y_min, rec.y_max)[true_cell]
        new_post, _, _, _ = direct_update(posterior, rec, det, grid, temperature)
        if rec.s_t == 0:
            new_post, mix, c_long, c_xy, _ = apply_progressive_scenario_escape(
                new_post, grid, env.history, scenario=scenario,
            )
        else:
            _, c_long, c_xy, _, _ = progressive_scenario_prior(
                grid, env.history, scenario=scenario,
            )
        posterior = new_post
        trace.append(posterior.copy())
        rows.append({
            "mission_id": rec.mission_id,
            "x_min": rec.x_min, "x_max": rec.x_max,
            "y_min": rec.y_min, "y_max": rec.y_max,
            "effort": rec.effort, "cost": rec.cost, "s_t": rec.s_t,
            "budget_remaining": rec.budget_remaining,
            "contains_true": bool(covered_true),
            "center_x": float(c_xy[0]), "center_y": float(c_xy[1]),
        })
        if rec.s_t == 1 and not assume_no_detection:
            break

    return {
        "history": pd.DataFrame(rows), "trace": trace,
        "true_cell": true_cell, "env": env,
    }


def tab_operational():
    grid = _load_grid()
    st.header("6 · Final operational model")
    st.markdown(
        "This tab runs the **recommended configuration** end-to-end, exactly as "
        "documented in [notebook 04](../notebooks/04_operational_summary.ipynb):"
    )
    st.markdown(
        "- Prior: **`P13_impact_robust_uniform10`** (derived from the EB winner, "
        "with 10% uniform mix for robustness)\n"
        "- Detector: **`D1_saturating_exponential`** (best calibrated against "
        "`previous_missions_reports.pdf`)\n"
        "- Strategy: **progressive scenario search** with the `mixture` hypothesis "
        "(centre + right + left + strong-wind)\n"
        "- Update temperature: **T = 1.5**\n"
        "- Budget cap: **B = 530**"
    )

    cfg_cols = st.columns([1, 1, 1, 1])
    scenario = cfg_cols[0].selectbox(
        "Search scenario", ["mixture", "center", "right", "left", "strong_wind"],
        index=0, key="op_scenario",
    )
    plant_scenario = cfg_cols[1].selectbox(
        "Plant scenario", ["center", "right", "left", "strong_wind"],
        index=0, key="op_plant_scenario",
        help="Only used when 'simulate true outcomes' is on. Otherwise the "
              "bomb is invisible to the run.",
    )
    assume_no_detection = cfg_cols[2].checkbox(
        "Assume non-detection at every step", value=True,
        help="If checked, every s_t is forced to 0 — gives you the full "
              "sequence of rectangles you would search before exhausting the "
              "budget. This is the most informative 'what if I never detect?' view.",
    )
    seed = cfg_cols[3].number_input("Seed", min_value=0, max_value=10_000,
                                       value=7, step=1, key="op_seed")
    max_missions = st.slider("Maximum missions", 4, 20, 12, 1, key="op_maxm")

    if st.button("Run operational pipeline", type="primary"):
        with st.spinner("Running operational pipeline..."):
            result = _run_operational(
                prior_name="P13_impact_robust_uniform10",
                scenario=scenario,
                detector_name="D1_saturating_exponential",
                temperature=1.5,
                budget=BUDGET_CAP,
                assume_no_detection=assume_no_detection,
                plant_scenario=plant_scenario,
                seed=int(seed),
                max_missions=int(max_missions),
            )
        st.session_state["op_result"] = result

    if "op_result" not in st.session_state:
        st.info("Click **Run operational pipeline** to generate the mission sequence.")
        return

    result = st.session_state["op_result"]
    df = result["history"]
    trace = result["trace"]
    true_cell = result["true_cell"]
    env = result["env"]

    st.subheader("Recommended mission sequence")
    st.dataframe(df, width="stretch")

    csv_text = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download missions.csv", csv_text,
                        file_name="missions.csv", mime="text/csv")

    detected = bool((df["s_t"] == 1).any())
    n_missions = len(df)
    cost_used = float(df["cost"].sum()) if n_missions else 0.0
    ever_covered = bool(df["contains_true"].any())
    st.markdown(
        f"**Detected:** {'yes' if detected else 'no (or forced non-detection)'}  ·  "
        f"**Missions:** {n_missions}  ·  "
        f"**Budget used:** {cost_used:.0f} / {BUDGET_CAP}  ·  "
        f"**True cell ever covered:** {'yes' if ever_covered else 'no'}  ·  "
        f"**True cell:** {true_cell}  (x = {grid.x[true_cell]}, y = {grid.y[true_cell]})"
    )

    st.subheader("Posterior at the end of the sequence")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    heatmap(axes[0], trace[0], grid, cmap="magma", title="Initial prior",
             true_cell=true_cell)
    heatmap(axes[1], trace[-1], grid, cmap="magma",
             title=f"Posterior after {n_missions} mission(s)",
             true_cell=true_cell)
    for _, r in df.iterrows():
        col = "lime" if r["s_t"] == 1 else "white"
        axes[1].add_patch(plt.Rectangle((r["x_min"], r["y_min"]),
                                          r["x_max"] - r["x_min"],
                                          r["y_max"] - r["y_min"],
                                          edgecolor=col, facecolor="none",
                                          lw=1.2, alpha=0.7))
        axes[1].text(r["x_min"], r["y_max"] + 0.2,
                      str(int(r["mission_id"])), color=col, fontsize=8)
    st.pyplot(fig)

    st.subheader("Per-mission posterior evolution")
    st.caption(
        "Each panel shows the posterior right after a mission. The white "
        "rectangle is the mission just performed; previous missions are shown "
        "in faint cyan."
    )
    ncols = 4
    nrows = int(np.ceil((len(trace) - 1) / ncols)) or 1
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(17, 3.6 * nrows))
    axes2 = np.atleast_2d(axes2).flatten()
    for k in range(1, len(trace)):
        ax = axes2[k - 1]
        ax.imshow(grid.reshape_2d(trace[k]), origin="lower",
                   extent=[0, grid.Nx, 0, grid.Ny], cmap="magma")
        ax.scatter(grid.x[true_cell], grid.y[true_cell], color="lime",
                    marker="X", s=60, edgecolor="black")
        for jj in range(k - 1):
            r = df.iloc[jj]
            ax.add_patch(plt.Rectangle((r["x_min"], r["y_min"]),
                                         r["x_max"] - r["x_min"],
                                         r["y_max"] - r["y_min"],
                                         edgecolor="cyan", facecolor="none",
                                         lw=0.9, alpha=0.4))
        r = df.iloc[k - 1]
        ax.add_patch(plt.Rectangle((r["x_min"], r["y_min"]),
                                     r["x_max"] - r["x_min"],
                                     r["y_max"] - r["y_min"],
                                     edgecolor="white", facecolor="none", lw=1.6))
        ax.set_title(f"Mission {int(r['mission_id'])}  effort={int(r['effort'])}  "
                      f"cost={int(r['cost'])}  s_t={int(r['s_t'])}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes2[len(trace) - 1:]:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Recommended next action")
    if detected:
        st.success(
            f"The aircraft was detected in mission {int(df[df['s_t']==1].iloc[0]['mission_id'])}. "
            "The operational sequence completes here."
        )
    else:
        if n_missions == 0:
            st.info("No mission could be proposed within budget.")
        else:
            next_rect = (df.iloc[-1])
            st.markdown(
                f"After the {n_missions} simulated missions, the next recommended "
                "rectangle the pipeline would propose if the campaign continued is "
                "the proposal at the bottom of the table above. "
                f"Remaining budget: **{env.budget_total - cost_used:.0f}**."
            )


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main():
    prior_name, detector_name, update_temperature, budget = render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Priors", "Detectors",
        "Strategy preview", "Simulator", "Operational model",
    ])
    with tab1:
        tab_overview()
    with tab2:
        tab_priors(prior_name)
    with tab3:
        tab_detectors(detector_name)
    with tab4:
        tab_strategy_preview(prior_name, detector_name, budget)
    with tab5:
        tab_simulator(prior_name, detector_name, update_temperature, budget)
    with tab6:
        tab_operational()


if __name__ == "__main__":
    main()
else:
    main()
