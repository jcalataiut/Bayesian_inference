"""Local Streamlit clone of the professor's search-missions webapp.

Adds knobs the real webapp does not expose:
  * pick a prior (uniform / drift / drift+witnesses / drift+witnesses+deep)
  * tune DetectionModel parameters
  * pick a strategy and auto-fill the next mission proposal
  * plant the hidden truth (random, from prior, or by clicking coordinates)
  * inspect prior, posterior, true cell, and mission history side by side.

Run with:

    cd <repo_root>
    streamlit run gemini_proyect/app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the gemini_proyect package importable when run via `streamlit run`.
THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gemini_proyect.modeling import (  # noqa: E402
    DetectionModel,
    drift_prior,
    drift_prior_with_witnesses,
    mission_likelihood,
    physics_prior,
    posterior_update,
    sinking_adjusted_prior,
    step_by_step_posterior,
    uniform_prior,
)
from gemini_proyect.simulator import SearchEnvironment, TrueDetector  # noqa: E402
from gemini_proyect.strategies import (  # noqa: E402
    CommitAndVerifyStrategy,
    StochasticPosteriorSampler,
    propose_cooldown_aware,
    propose_info_gain,
    propose_max_expected_detection,
    propose_max_posterior_rect,
)
from gemini_proyect.experiments.multi_campaign import (  # noqa: E402
    run_multi_campaign,
    summarize_campaigns,
)


GRID_CSV = THIS.parents[1] / "data" / "grid_dataset.csv"

st.set_page_config(page_title="Bayesian search sandbox", layout="wide")


# ---------------------------------------------------------------------- #
#  Session helpers
# ---------------------------------------------------------------------- #
def _build_prior(name: str, grid, depth_bias: float = 1.0, mc_seed: int = 0):
    if name == "uniform":
        return uniform_prior(grid)
    if name == "drift":
        return drift_prior(grid)
    if name == "drift+witnesses":
        return drift_prior_with_witnesses(grid)
    if name == "drift+witnesses+deep":
        return sinking_adjusted_prior(drift_prior_with_witnesses(grid), grid, depth_bias=depth_bias)
    if name == "physics_mc":
        return physics_prior(grid, n_samples=2000, seed=mc_seed)
    raise ValueError(name)


def _init_state(force: bool = False):
    if force or "env" not in st.session_state:
        seed = int(st.session_state.get("seed", 0))
        budget = float(st.session_state.get("budget_total", 230.0))
        true_det = TrueDetector(
            base=float(st.session_state.get("td_base", 1.0)),
            p_unit=float(st.session_state.get("td_p_unit", 0.6)),
            alpha_d=float(st.session_state.get("td_alpha_d", 0.4)),
            alpha_r=float(st.session_state.get("td_alpha_r", 0.3)),
        )
        env = SearchEnvironment.from_csv(
            GRID_CSV, seed=seed, budget_total=budget, detector=true_det
        )
        st.session_state.env = env
        st.session_state.prior = None  # rebuilt on demand
        st.session_state.posterior = None
        st.session_state.true_planted = False


def _plant_truth(mode: str, prior_for_truth: np.ndarray):
    env: SearchEnvironment = st.session_state.env
    if mode == "From the displayed prior":
        env.plant_object(prior=prior_for_truth)
    elif mode == "Uniform random":
        env.plant_object()
    else:
        # manual coordinates
        x = float(st.session_state.get("truth_x", 7.0))
        y = float(st.session_state.get("truth_y", 20.0))
        mask = (env.grid.x == np.floor(x) + 0.5) & (env.grid.y == np.floor(y) + 0.5)
        idx = int(np.argmax(mask))
        if not mask.any():
            idx = int(np.argmin((env.grid.x - x) ** 2 + (env.grid.y - y) ** 2))
        env.plant_object(cell_id=idx)
    st.session_state.true_planted = True


# ---------------------------------------------------------------------- #
#  Sidebar — global controls
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("Simulator")
    st.number_input("Seed", value=0, key="seed", step=1)
    st.number_input("Budget total", value=230.0, key="budget_total", step=10.0)

    st.divider()
    st.subheader("True detector (hidden)")
    st.slider("base", 0.5, 1.0, 1.0, 0.05, key="td_base")
    st.slider("p_unit (per-pass)", 0.1, 0.95, 0.6, 0.05, key="td_p_unit")
    st.slider("alpha_d (depth penalty)", 0.0, 2.0, 0.4, 0.1, key="td_alpha_d")
    st.slider("alpha_r (rough penalty)", 0.0, 2.0, 0.3, 0.1, key="td_alpha_r")

    st.divider()
    if st.button("Reset campaign", type="primary"):
        _init_state(force=True)
        st.rerun()

# Initialise environment once parameters are stable.
_init_state(force=False)
env: SearchEnvironment = st.session_state.env
grid = env.grid


# ---------------------------------------------------------------------- #
#  Top bar — team status, mirroring the professor's webapp
# ---------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Completed missions", len(env.history))
c2.metric("Budget total", f"{env.budget_total:.0f}")
c3.metric("Budget used", f"{env.budget_used:.0f}")
c4.metric("Budget remaining", f"{env.budget_remaining:.0f}")

st.divider()


# ---------------------------------------------------------------------- #
#  Modeling controls
# ---------------------------------------------------------------------- #
mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.subheader("Prior")
    prior_name = st.selectbox(
        "Choose a prior",
        ["uniform", "drift", "drift+witnesses", "drift+witnesses+deep", "physics_mc"],
        index=4,
        help=(
            "physics_mc marginalises over uncertain physics constants "
            "(τ_fall, w_wind, τ_drift, witness shifts) via Monte Carlo, "
            "adds depth/roughness factors and a small uniform component."
        ),
    )
    depth_bias = st.slider(
        "depth_bias (only for +deep and physics_mc)", -3.0, 3.0, 0.5, 0.1
    )
    mc_seed = st.number_input("Monte Carlo seed (physics_mc only)", value=0, step=1)

with mc2:
    st.subheader("Detection model (ours)")
    lambda_0 = st.slider("lambda_0", 0.1, 3.0, 1.2, 0.1)
    a_d = st.slider("a_d", 0.0, 2.0, 1.0, 0.1)
    a_r = st.slider("a_r", 0.0, 2.0, 0.6, 0.1)
    model = DetectionModel(lambda_0=lambda_0, a_d=a_d, a_r=a_r)

with mc3:
    st.subheader("Truth")
    truth_mode = st.radio(
        "How to plant the hidden cell",
        ["From the displayed prior", "Uniform random", "Manual coordinates"],
        index=0,
    )
    if truth_mode == "Manual coordinates":
        st.number_input("truth x", 0.5, float(grid.Nx) - 0.5, 7.0, 1.0, key="truth_x")
        st.number_input("truth y", 0.5, float(grid.Ny) - 0.5, 20.0, 1.0, key="truth_y")
    if st.button("Plant / re-plant truth"):
        prior_tmp = _build_prior(prior_name, grid, depth_bias=depth_bias, mc_seed=int(mc_seed))
        _plant_truth(truth_mode, prior_tmp)


# Build current prior + posterior on every render.
prior = _build_prior(prior_name, grid, depth_bias=depth_bias, mc_seed=int(mc_seed))
st.session_state.prior = prior
posterior = posterior_update(prior, grid, model, env.history)
st.session_state.posterior = posterior


# ---------------------------------------------------------------------- #
#  Mission preview + new mission form
# ---------------------------------------------------------------------- #
st.divider()
preview_col, form_col = st.columns([3, 2])

with form_col:
    st.subheader("New mission")
    auto_propose = st.checkbox("Auto-fill from a strategy", value=False)
    if auto_propose:
        strategy_name = st.selectbox(
            "Strategy",
            [
                "cooldown_aware",
                "commit_and_verify",
                "thompson",
                "max_posterior_rect",
                "max_expected_detection",
                "info_gain",
            ],
            index=0,
            help=(
                "Top entries are designed for the 12 h cooldown regime "
                "(few big missions). max_expected_detection / info_gain "
                "need 50+ missions and are kept for benchmarking only."
            ),
        )

        if strategy_name == "cooldown_aware":
            st.session_state.pop("cav", None)
            st.session_state.pop("cav_key", None)
            st.session_state.pop("samp", None)
            st.session_state.pop("samp_seed", None)
            n_missions_target = st.slider(
                "Mission budget plan (target # missions)",
                min_value=2,
                max_value=15,
                value=5,
                step=1,
                help=(
                    "Plan to spread the remaining budget across this many "
                    "missions. Each mission's cost is soft-capped at "
                    "budget_remaining / n_missions_target so the strategy "
                    "doesn't blow the plan on a single sweep."
                ),
            )
            try:
                prop = propose_cooldown_aware(
                    posterior, grid, model, max(env.budget_remaining, 1.0),
                    n_missions_target=int(n_missions_target),
                )
                x_min_default, x_max_default = prop.x_min, prop.x_max
                y_min_default, y_max_default = prop.y_min, prop.y_max
                effort_default = prop.effort
                st.info(
                    f"cooldown_aware (plan={n_missions_target} missions) → "
                    f"x=[{prop.x_min:.1f}, {prop.x_max:.1f}], "
                    f"y=[{prop.y_min:.1f}, {prop.y_max:.1f}], "
                    f"effort={prop.effort}, cost={prop.cost:.0f}. "
                    "Score = expected P(detect) for this single mission."
                )
            except RuntimeError as e:
                st.warning(f"Strategy could not propose: {e}")
                x_min_default, x_max_default = 0.5, 3.5
                y_min_default, y_max_default = 0.5, 3.5
                effort_default = 1
        elif strategy_name == "commit_and_verify":
            confidence = st.slider(
                "Confidence threshold (commit zone until ruled out)",
                min_value=0.50,
                max_value=0.99,
                value=0.80,
                step=0.01,
                help=(
                    "Keep probing the same rectangle until P(Z ∈ R | data) "
                    "drops below 1 − confidence."
                ),
            )
            base_proposer_name = st.selectbox(
                "Base proposer for new commitments",
                ["max_expected_detection", "info_gain", "max_posterior_rect"],
                index=0,
            )
            # Persist (and reset when settings change) the strategy state.
            key = ("cav", base_proposer_name)
            cav = st.session_state.get("cav")
            if (
                cav is None
                or st.session_state.get("cav_key") != key
                or cav.confidence != confidence
            ):
                cav = CommitAndVerifyStrategy(
                    confidence=confidence, base_proposer_name=base_proposer_name
                )
                st.session_state.cav = cav
                st.session_state.cav_key = key
            else:
                cav.confidence = confidence  # allow live slider changes
            if st.button("Reset commitment"):
                cav.reset()

            try:
                prop = cav.propose(posterior, grid, model, max(env.budget_remaining, 1.0))
                x_min_default, x_max_default = prop.x_min, prop.x_max
                y_min_default, y_max_default = prop.y_min, prop.y_max
                effort_default = prop.effort
                mass = cav.mass_in_commitment(posterior)
                conf_against = 1.0 - mass  # how sure we are the object is not in commit
                st.info(
                    f"commit_and_verify (confidence={confidence:.2f}) "
                    f"→ x=[{prop.x_min:.1f}, {prop.x_max:.1f}], "
                    f"y=[{prop.y_min:.1f}, {prop.y_max:.1f}], "
                    f"effort={prop.effort}, cost={prop.cost:.0f}. "
                    f"P(Z ∈ committed rect | data) = {mass:.3f}; "
                    f"confidence that object is NOT here = {conf_against:.3f}."
                )
                st.caption(
                    "While `P(Z ∈ R) > 1 − confidence` ({:.3f} > {:.3f}) the "
                    "strategy re-probes this rectangle. It only releases the "
                    "commitment after the zone is ruled out or detection "
                    "occurs.".format(mass, 1 - confidence)
                )
            except RuntimeError as e:
                st.warning(f"Strategy could not propose: {e}")
                x_min_default, x_max_default = 0.5, 3.5
                y_min_default, y_max_default = 0.5, 3.5
                effort_default = 1
        elif strategy_name == "thompson":
            st.session_state.pop("cav", None)
            st.session_state.pop("cav_key", None)
            sampler_seed = st.number_input("Thompson RNG seed", value=42, step=1)
            samp = st.session_state.get("samp")
            if samp is None or st.session_state.get("samp_seed") != sampler_seed:
                samp = StochasticPosteriorSampler(seed=int(sampler_seed))
                st.session_state.samp = samp
                st.session_state.samp_seed = sampler_seed
            try:
                prop = samp.propose(posterior, grid, model, max(env.budget_remaining, 1.0))
                x_min_default, x_max_default = prop.x_min, prop.x_max
                y_min_default, y_max_default = prop.y_min, prop.y_max
                effort_default = prop.effort
                st.info(
                    f"thompson (seed={sampler_seed}) → "
                    f"x=[{prop.x_min:.1f}, {prop.x_max:.1f}], "
                    f"y=[{prop.y_min:.1f}, {prop.y_max:.1f}], "
                    f"effort={prop.effort}, cost={prop.cost:.0f}. "
                    "Different runs propose different rectangles by design — "
                    "the sampler draws j* ~ posterior each call."
                )
            except RuntimeError as e:
                st.warning(f"Strategy could not propose: {e}")
                x_min_default, x_max_default = 0.5, 3.5
                y_min_default, y_max_default = 0.5, 3.5
                effort_default = 1
        else:
            # Reset any commit / Thompson state when leaving those strategies.
            st.session_state.pop("cav", None)
            st.session_state.pop("cav_key", None)
            st.session_state.pop("samp", None)
            st.session_state.pop("samp_seed", None)
            propose_fn = {
                "max_expected_detection": propose_max_expected_detection,
                "info_gain": propose_info_gain,
                "max_posterior_rect": propose_max_posterior_rect,
            }[strategy_name]
            try:
                prop = propose_fn(posterior, grid, model, max(env.budget_remaining, 1.0))
                x_min_default, x_max_default = prop.x_min, prop.x_max
                y_min_default, y_max_default = prop.y_min, prop.y_max
                effort_default = prop.effort
                st.info(
                    f"{strategy_name} proposes "
                    f"x=[{prop.x_min:.1f}, {prop.x_max:.1f}], "
                    f"y=[{prop.y_min:.1f}, {prop.y_max:.1f}], "
                    f"effort={prop.effort}, cost={prop.cost:.0f}, score={prop.score:.4f}"
                )
            except RuntimeError as e:
                st.warning(f"Strategy could not propose: {e}")
                x_min_default, x_max_default = 0.5, 3.5
                y_min_default, y_max_default = 0.5, 3.5
                effort_default = 1
    else:
        x_min_default, x_max_default = 0.5, 3.5
        y_min_default, y_max_default = 0.5, 3.5
        effort_default = 1

    x_min = st.number_input("x_min", 0.5, float(grid.Nx) - 0.5, x_min_default, 1.0)
    x_max = st.number_input("x_max", 0.5, float(grid.Nx) - 0.5, x_max_default, 1.0)
    y_min = st.number_input("y_min", 0.5, float(grid.Ny) - 0.5, y_min_default, 1.0)
    y_max = st.number_input("y_max", 0.5, float(grid.Ny) - 0.5, y_max_default, 1.0)
    effort = st.selectbox("Effort level", [1, 2, 3], index=[1, 2, 3].index(effort_default))

    covered = grid.cells_in_rectangle(x_min, x_max, y_min, y_max)
    n_cells = int(covered.size)
    cost = effort * n_cells

    st.markdown(
        f"**Mission ID:** {env._next_mission_id}  \n"
        f"**Covered cells:** {n_cells}  \n"
        f"**Cost:** {cost}  \n"
        f"**Remaining budget after submission:** {env.budget_remaining - cost:.0f}"
    )

    can_submit = (
        st.session_state.true_planted
        and cost <= env.budget_remaining
        and n_cells > 0
        and x_min <= x_max
        and y_min <= y_max
    )
    if not st.session_state.true_planted:
        st.warning("Plant a truth before running missions.")
    if cost > env.budget_remaining:
        st.error("Mission cost exceeds remaining budget.")
    if x_min > x_max or y_min > y_max:
        st.error("Invalid rectangle bounds.")

    if st.button("Run mission", type="primary", disabled=not can_submit):
        rec = env.run_mission(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, effort=effort)
        st.success(f"Mission {rec.mission_id} → s_t = {rec.s_t}")
        st.rerun()


with preview_col:
    st.subheader("Mission preview")

    surface_choice = st.radio(
        "Underlay",
        ["depth", "prior", "posterior"],
        index=2,
        horizontal=True,
    )
    if surface_choice == "depth":
        surface = grid.reshape_2d(grid.depth)
        cmap = "viridis"
        cbar_label = "depth"
    elif surface_choice == "prior":
        surface = grid.reshape_2d(prior)
        cmap = "magma"
        cbar_label = "prior π_j"
    else:
        surface = grid.reshape_2d(posterior)
        cmap = "magma"
        cbar_label = "posterior π_j | s_{1:T}"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(
        surface,
        origin="lower",
        cmap=cmap,
        extent=[0, grid.Nx, 0, grid.Ny],
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)

    # Mission history rectangles.
    for m in env.history:
        ax.add_patch(
            plt.Rectangle(
                (m.x_min - 0.5, m.y_min - 0.5),
                m.x_max - m.x_min + 1,
                m.y_max - m.y_min + 1,
                fill=False,
                edgecolor="red" if m.s_t == 1 else "white",
                linewidth=2 if m.s_t == 1 else 0.8,
                alpha=0.9,
            )
        )

    # Pending mission rectangle.
    ax.add_patch(
        plt.Rectangle(
            (x_min - 0.5, y_min - 0.5),
            x_max - x_min + 1,
            y_max - y_min + 1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
            linestyle="--",
            label="pending mission",
        )
    )

    # Accident point.
    ax.scatter([7], [20], color="cyan", s=80, marker="*", label="accident")

    # Truth (only if planted and user wants to see it).
    show_truth = st.checkbox("Reveal hidden truth on the map", value=False)
    if st.session_state.true_planted and show_truth:
        tc = env.true_cell
        ax.scatter([grid.x[tc]], [grid.y[tc]], color="red", s=120, marker="x", label="truth")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Grid and missions")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------------- #
#  Detectability ρ and likelihood explorer (uses the pending rect / effort)
# ---------------------------------------------------------------------- #
st.divider()
st.subheader("Detectability ρ and likelihood — how observations turn into evidence")
st.markdown(
    r"""
The likelihood is **not** something that exists before data; it is born with
each observation. The full chain that the model wires together (Tasks 2–3 of
the deliverable) is:

$$
\rho_{t,j} = 1 - \exp\!\bigl(-\lambda_0\,(1-d_j)^{a_d}\,(1-r_j)^{a_r}\,e_t\bigr)
\;\;\longrightarrow\;\;
q_{t,j} = c_{t,j}\,\rho_{t,j}
\;\;\longrightarrow\;\;
L_j(s_t) = q_{t,j}^{\,s_t}\,(1-q_{t,j})^{1-s_t}
$$

* $\rho_{t,j}$ is high in **shallow, smooth** cells and low in **deep, rough**
  ones — exactly the qualitative pattern of the previous mission reports.
* $c_{t,j}\in\{0,1\}$ is the coverage indicator of the rectangle, so
  $q_{t,j}$ is zero outside the searched rectangle.
* If a cell is **not covered**, $q=0$; then $L=1$ when $s_t=0$ (nothing
  learnt about that cell) and $L=0$ when $s_t=1$ (any detection rules out
  every uncovered cell).
* The observation $s_t\in\{0,1\}$ enters **only through the exponent**:
  if you detect, the likelihood is $q$; if you don't, it is $1-q$.

The panels below use the rectangle and effort of the *pending* mission so
you can analyse the evidence each possible outcome would carry, before you
actually click **Run mission**.
"""
)

inspect_effort = st.select_slider(
    "Effort level for the detectability map (independent of the pending mission's effort)",
    options=[1, 2, 3],
    value=int(effort),
)
rho_inspect = model.rho(grid.depth, grid.roughness, inspect_effort)

covered_mask_pending = (
    (grid.x >= x_min) & (grid.x <= x_max) & (grid.y >= y_min) & (grid.y <= y_max)
)
rho_pending = model.rho(grid.depth, grid.roughness, effort)
q_pending = np.where(covered_mask_pending, rho_pending, 0.0)
L_if_detect = q_pending                        # if s_t = 1
L_if_no_detect = 1.0 - q_pending               # if s_t = 0

fig_lk, axes_lk = plt.subplots(1, 4, figsize=(20, 4.2), constrained_layout=True)
panels_lk = [
    (rho_inspect, f"ρ_t,j (effort = {inspect_effort})\nany cell, before coverage", "magma"),
    (
        q_pending,
        f"q_t,j = c_t,j · ρ_t,j\npending rect, effort = {effort}",
        "magma",
    ),
    (L_if_no_detect, "Likelihood L_j if s_t = 0\n(no detection)", "cividis"),
    (L_if_detect, "Likelihood L_j if s_t = 1\n(detection)", "cividis"),
]
for ax, (arr, title, cmap) in zip(axes_lk, panels_lk):
    vmin = float(arr.min())
    vmax = float(arr.max()) if arr.max() > vmin else vmin + 1e-12
    im = ax.imshow(
        grid.reshape_2d(arr),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, grid.Nx, 0, grid.Ny],
        aspect="auto",
    )
    if title.startswith("q_") or title.startswith("Likelihood"):
        ax.add_patch(
            plt.Rectangle(
                (x_min - 0.5, y_min - 0.5),
                x_max - x_min + 1,
                y_max - y_min + 1,
                fill=False,
                edgecolor="lime",
                linewidth=2,
                linestyle="--",
            )
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig_lk.colorbar(im, ax=ax, shrink=0.85)
st.pyplot(fig_lk, clear_figure=True)

# Counterfactual posteriors for the pending mission, conditional on each s_t.
st.markdown("**Counterfactual posterior for the pending mission**")
st.caption(
    "If you submitted the pending rectangle and got s_t = 0 vs s_t = 1, "
    "this is the posterior you would obtain on top of the current one. "
    "Note that current posterior = prior when no mission has been run yet."
)
current_posterior = posterior  # already computed for the existing history

post_if_no_detect = current_posterior * L_if_no_detect
sn = post_if_no_detect.sum()
post_if_no_detect = post_if_no_detect / sn if sn > 0 else current_posterior

post_if_detect = current_posterior * L_if_detect
sd = post_if_detect.sum()
post_if_detect = post_if_detect / sd if sd > 0 else np.zeros_like(current_posterior)

fig_cf, axes_cf = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for ax, arr, title in zip(
    axes_cf,
    [post_if_no_detect, post_if_detect],
    ["Counterfactual posterior if s_t = 0", "Counterfactual posterior if s_t = 1"],
):
    vmin = float(arr.min())
    vmax = float(arr.max()) if arr.max() > vmin else vmin + 1e-12
    im = ax.imshow(
        grid.reshape_2d(arr),
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=[0, grid.Nx, 0, grid.Ny],
        aspect="auto",
    )
    ax.add_patch(
        plt.Rectangle(
            (x_min - 0.5, y_min - 0.5),
            x_max - x_min + 1,
            y_max - y_min + 1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
            linestyle="--",
        )
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig_cf.colorbar(im, ax=ax, shrink=0.85)
st.pyplot(fig_cf, clear_figure=True)

p_s1 = float(np.sum(current_posterior * q_pending))
p_s0 = 1.0 - p_s1
st.markdown(
    f"**Predicted outcome probabilities under current posterior:** "
    f"P(s_t = 1) = {p_s1:.3f}, P(s_t = 0) = {p_s0:.3f}"
)


# ---------------------------------------------------------------------- #
#  Bayesian update breakdown
# ---------------------------------------------------------------------- #
st.divider()
st.subheader("Bayesian update — prior × likelihood → posterior")

if len(env.history) == 0:
    st.info(
        "No missions have been run yet, so there is **no likelihood**: the "
        "posterior equals the prior. Submit a mission above and this section "
        "will let you step through how each observation reshapes the posterior."
    )
else:
    states = step_by_step_posterior(prior, grid, model, env.history)
    # states[0] = initial prior, states[k] = posterior after k missions.

    t = st.slider(
        "Inspect mission number",
        min_value=1,
        max_value=len(env.history),
        value=len(env.history),
        step=1,
    )
    mission = env.history[t - 1]
    prior_before = states[t - 1]                   # posterior after t-1 missions
    posterior_after = states[t]                    # posterior after t missions
    likelihood = mission_likelihood(grid, model, mission)

    st.markdown(
        f"**Mission {mission.mission_id}** "
        f"— rect x=[{mission.x_min:.1f}, {mission.x_max:.1f}], "
        f"y=[{mission.y_min:.1f}, {mission.y_max:.1f}], "
        f"effort = {mission.effort}, "
        f"observed **s_t = {mission.s_t}** "
        f"({'detected' if mission.s_t == 1 else 'no detection'})."
    )
    st.markdown(
        r"$$ \pi_j^{(t)} \;\propto\; \pi_j^{(t-1)} \cdot L_j(s_t) "
        r"\qquad L_j(s_t)=\begin{cases} q_{t,j} & s_t = 1 \\ 1 - q_{t,j} & s_t = 0 \end{cases} "
        r"\qquad q_{t,j} = c_{t,j}\,\rho_{t,j}. $$"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    panels = [
        (prior_before, f"Prior before mission {t}\n(= posterior after {t-1})", "magma"),
        (likelihood, f"Likelihood L_j(s_t={mission.s_t})", "cividis"),
        (posterior_after, f"Posterior after mission {t}", "magma"),
    ]
    for ax, (arr, title, cmap) in zip(axes, panels):
        vmin = float(arr.min())
        vmax = float(arr.max()) if arr.max() > vmin else vmin + 1e-12
        im = ax.imshow(
            grid.reshape_2d(arr),
            origin="lower",
            cmap=cmap,
            extent=[0, grid.Nx, 0, grid.Ny],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        # Mark this mission's rectangle on every panel.
        ax.add_patch(
            plt.Rectangle(
                (mission.x_min - 0.5, mission.y_min - 0.5),
                mission.x_max - mission.x_min + 1,
                mission.y_max - mission.y_min + 1,
                fill=False,
                edgecolor="red" if mission.s_t == 1 else "white",
                linewidth=2,
            )
        )
        if st.session_state.true_planted:
            tc = env.true_cell
            ax.scatter(
                [grid.x[tc]], [grid.y[tc]], color="red", s=60, marker="x", alpha=0.8
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    st.pyplot(fig, clear_figure=True)

    # Top-5 cells most strengthened / weakened by this mission.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prior_before > 0, posterior_after / prior_before, 0.0)
    top_up = np.argsort(ratio)[-5:][::-1]
    top_down = np.argsort(ratio)[:5]
    import pandas as pd
    rows_up = pd.DataFrame(
        {
            "cell_id": top_up,
            "x": grid.x[top_up],
            "y": grid.y[top_up],
            "prior_before": prior_before[top_up],
            "likelihood": likelihood[top_up],
            "posterior_after": posterior_after[top_up],
            "ratio (post/prior)": ratio[top_up],
        }
    )
    rows_down = pd.DataFrame(
        {
            "cell_id": top_down,
            "x": grid.x[top_down],
            "y": grid.y[top_down],
            "prior_before": prior_before[top_down],
            "likelihood": likelihood[top_down],
            "posterior_after": posterior_after[top_down],
            "ratio (post/prior)": ratio[top_down],
        }
    )
    cu, cd = st.columns(2)
    cu.markdown("**Top-5 most reinforced cells**")
    cu.dataframe(rows_up, use_container_width=True, hide_index=True)
    cd.markdown("**Top-5 most weakened cells**")
    cd.dataframe(rows_down, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------- #
#  Multi-campaign simulator
# ---------------------------------------------------------------------- #
st.divider()
st.subheader("Multi-campaign benchmark")
st.markdown(
    """
Simulate many full campaigns to compare strategies. For each campaign we
**sample one truth location** from the chosen truth prior and **replay
every selected strategy on the same truth** (paired design → much lower
variance). Between campaigns the truth changes, so the success rate you
see at the bottom is a Monte Carlo estimate of how often each strategy
finds the object across the truth distribution.

Each campaign for each strategy runs until detection or the strategy
exhausts the budget — so the number of missions per campaign is determined
by the strategy itself, not capped externally.
"""
)

sim_c1, sim_c2, sim_c3 = st.columns(3)
with sim_c1:
    n_campaigns = st.number_input("Number of campaigns", 5, 500, 50, 5)
    sim_budget = st.number_input("Budget per campaign", 50, 1000, 230, 10)
with sim_c2:
    sim_prior = st.selectbox(
        "Truth & inference prior",
        ["uniform", "drift", "drift+witnesses", "drift+witnesses+deep", "physics_mc"],
        index=4,
        key="sim_prior",
    )
    sim_confidence = st.slider(
        "Confidence for commit_and_verify", 0.50, 0.99, 0.80, 0.01, key="sim_confidence"
    )
    sim_n_missions = st.slider(
        "n_missions_target (cooldown_aware)", 2, 15, 5, 1, key="sim_n_missions_target"
    )
with sim_c3:
    sim_strategies = st.multiselect(
        "Strategies to compare",
        [
            "cooldown_aware",
            "commit_and_verify",
            "thompson",
            "max_posterior_rect",
            "max_expected_detection",
            "info_gain",
        ],
        default=["cooldown_aware", "commit_and_verify", "thompson", "max_posterior_rect"],
    )

run_sim = st.button("Run multi-campaign simulation", type="primary")

if run_sim:
    if not sim_strategies:
        st.error("Pick at least one strategy.")
    else:
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(done: int, total: int):
            progress.progress(done / max(total, 1))
            status.text(f"Progress: {done} / {total} (campaign × strategy) runs")

        df_sim = run_multi_campaign(
            strategies=sim_strategies,
            prior_name=sim_prior,
            n_campaigns=int(n_campaigns),
            budget_total=float(sim_budget),
            confidence=float(sim_confidence),
            n_missions_target=int(sim_n_missions),
            mc_seed=int(mc_seed),
            depth_bias=float(depth_bias),
            detector_kwargs=dict(
                base=float(st.session_state.get("td_base", 1.0)),
                p_unit=float(st.session_state.get("td_p_unit", 0.6)),
                alpha_d=float(st.session_state.get("td_alpha_d", 0.4)),
                alpha_r=float(st.session_state.get("td_alpha_r", 0.3)),
            ),
            detection_model_kwargs=dict(lambda_0=lambda_0, a_d=a_d, a_r=a_r),
            progress_cb=_cb,
        )
        progress.empty()
        status.empty()
        st.session_state.sim_df = df_sim

if "sim_df" in st.session_state:
    df_sim = st.session_state.sim_df
    summary = summarize_campaigns(df_sim)
    st.markdown("**Summary (per strategy):**")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Side-by-side bars: detection rate, mean missions, mean budget.
    fig_sum, axes_sum = plt.subplots(1, 3, figsize=(15, 3.5), constrained_layout=True)
    axes_sum[0].bar(summary["strategy"], summary["detection_rate"], color="tab:green")
    axes_sum[0].set_title("Detection rate")
    axes_sum[0].set_ylim(0, 1)
    axes_sum[0].tick_params(axis="x", rotation=30)
    axes_sum[1].bar(summary["strategy"], summary["mean_missions"], color="tab:blue")
    axes_sum[1].set_title("Mean missions to outcome")
    axes_sum[1].tick_params(axis="x", rotation=30)
    axes_sum[2].bar(summary["strategy"], summary["mean_budget_used"], color="tab:orange")
    axes_sum[2].set_title("Mean budget used")
    axes_sum[2].tick_params(axis="x", rotation=30)
    st.pyplot(fig_sum, clear_figure=True)

    st.markdown("**Per-campaign detail (paired):**")
    pivot = df_sim.pivot(index="campaign", columns="strategy", values="found").astype(int)
    st.dataframe(pivot, use_container_width=True)
    st.download_button(
        "Download per-campaign trials",
        df_sim.to_csv(index=False).encode("utf-8"),
        file_name="multi_campaign_trials.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------- #
#  Mission history table + download
# ---------------------------------------------------------------------- #
st.divider()
st.subheader("Mission history")
df = env.history_to_dataframe()
if len(df):
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download missions.csv (deliverable format)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="missions.csv",
        mime="text/csv",
    )
else:
    st.caption("No missions yet.")
