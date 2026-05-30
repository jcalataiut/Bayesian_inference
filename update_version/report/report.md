# Deliverable 2 — Bayesian Spatial Search

University of Barcelona — Bayesian Inference — Team `TEAM_XX`

## 1. Prior construction

We model the unknown cell $Z$ via a **logistic-softmax prior with hierarchical coefficients**, rather than a fixed Gaussian. For each cell $j$:
$$\pi_j(\beta, \tau) = \mathrm{softmax}_j\bigl(\eta_j\bigr),\quad \eta_j = \beta_0 + \beta_1 d_\text{long}(j;\tau) + \beta_2 d_\text{trans}(j;\tau) - \beta_3 d_\text{long}^2(j;\tau) - \beta_4 d_\text{trans}^2(j;\tau).$$

Here $d_\text{long}$ and $d_\text{trans}$ are signed projections of the cell center onto, respectively, the falling-trajectory axis $d_\text{norm} = \mathrm{normalize}(v_\text{plane} + 0.5\,v_\text{wind})$ and its perpendicular $n_\text{norm}$, relative to the expected landing point
$$\mu(\tau) = x_E + \tau_\text{fall}(v_\text{plane} + 0.5\,v_\text{wind}) + \tau_\text{drift} v_\text{drift}.$$

The **mixed (linear + quadratic) form** is intentional: the quadratic terms give a Gaussian-like peak around $\mu$ (HalfNormal priors on $\beta_3, \beta_4$ ensure positivity), while the linear terms allow witness-driven displacement (witness 1 forward bias, witness 2 lateral offset). Both $\tau_\text{fall}$ and $\tau_\text{drift}$ have LogNormal$(0, 0.3)$ priors centered on 1, propagating uncertainty about the fall/drift timescales.

The chosen baseline is `P6_mixed_wit_informative`: mixed form, witness-aware priors $\beta_1 \sim \mathcal{N}(0.10, 0.10)$, $\beta_2 \sim \mathcal{N}(0.07, 0.10)$, and quadratic coefficients $\beta_3 \sim \mathrm{HalfNormal}(1/(2\cdot 6^2))$, $\beta_4 \sim \mathrm{HalfNormal}(1/(2\cdot 4^2))$, consistent with longitudinal spread $\sim 6$ cells and transverse spread $\sim 4$ cells. Seven additional variants (forms `linear`/`quadratic`/`mixed` $\times$ witnesses on/off $\times$ informative/weak/vague) are evaluated by Monte Carlo in `experiments/compare_priors.py`. **The prior heatmap** is in notebook `deliverable_summary.ipynb`, Task 1.

## 2. Detection model

We use the **saturating-exponential family** $\rho_j = 1 - \exp(-\lambda_0(1-d_j)^{a_d}(1-r_j)^{a_r}\,e)$ with $\lambda_0 = 1.2$, $a_d = 1.0$, $a_r = 0.6$ — the classical search-theory form (Koopman, Stone). It is monotone in effort, in $(1-d)$ and in $(1-r)$, takes values in $[0, 1)$, and effort acts as the number of independent sensor passes. Parameters are calibrated qualitatively against `previous_missions_reports.pdf`: report 2 (shallow + smooth + high effort) gives $\rho \approx 0.96$ in our model, report 3 (deep + rough + high effort) gives $\rho \approx 0.05$, report 4 (intermediate, repeated inspection) gives $\rho \approx 0.45$. Three alternative families (independent passes, logistic GLM, multiplicative + soft threshold) are implemented in `modeling/detection.py` for sensitivity analysis.

## 3. Posterior update

Given history $s_{1:T}$, we **marginalize** the latent cell $Z$ analytically:
$$P(s_t \mid \beta, \tau) = \sum_{j \in R_t} \pi_j(\beta, \tau)\,\rho_j^{s_t}(1 - \rho_j)^{1-s_t}.$$
Cells outside $R_t$ contribute zero to the $s_t=1$ branch and the unmasked prior mass to the $s_t=0$ branch, giving the compact form
$\log P(s_t\mid\beta,\tau) = \log s_R$ if $s_t = 1$, or $\log(1 - s_R)$ if $s_t = 0$, with $s_R = \sum_{j\in R_t} \pi_j \rho_j$.

The joint log-posterior is
$$\log P(\beta, \tau \mid s_{1:T}) = \log P(\beta) + \log P(\tau) + \sum_t \log P(s_t \mid \beta, \tau)$$
sampled with **PyMC NUTS** (4 chains × 1000 draws + 1000 tune, $\hat R < 1.01$). Per-cell posterior is then $P(Z=j \mid s_{1:T}) = \mathbb{E}_{\beta,\tau\sim\text{post}}\!\left[\pi_j(\beta,\tau)\,L_j / \sum_k \pi_k L_k\right]$, computed by averaging the per-sample normalized distribution. Empty-history MCMC reproduces the prior to KL $< 0.02$ (sanity 1); a synthetic detection in a small rectangle concentrates $P(Z\in R) > 0.99$ (sanity 2). Heatmap and trace plots in notebook 03.

## 4. Next mission — HDR funnel

We follow a **three-stage funnel** instead of a flat global rectangle search, so that decisions are explicitly tied to credible beliefs about $Z$ rather than to per-cost arithmetic over the whole grid:

1. **Highest-Density Region.** Sort cells by current $P(Z=j)$ descending and accumulate until reaching mass $\alpha$ (default $\alpha=0.80$); we visualise three nested levels $\{0.50, 0.80, 0.95\}$ to expose the shape of the belief.
2. **Score restricted to HDR.** Inside $\mathcal{H}_\alpha$ score each cell by $P(Z=j)\,\rho_j(e)$ for $e\in\{1,2,3\}$ — combining "where it could be" with "where the detector is informative".
3. **Top-$K$ → minimum rectangle.** Take the $K$ highest-scoring cells inside the HDR and propose the *minimum axis-aligned rectangle* that contains them. The effort is the one maximising $\dfrac{\sum_{j\in R} P(Z=j)\,\rho_j(e)}{e\,|R|}$ subject to the remaining budget.

Implementation in `strategies/next_mission.py` (`hdr_cells`, `hdr_mask`, `propose_via_hdr_topk`). The concrete proposal $(x_\text{min}, x_\text{max}, y_\text{min}, y_\text{max}, e)$ appears in `notebooks/deliverable_summary.ipynb` Task 4 and updates after every new observation.

## 5. Discussion — Flexibility and the structural ceiling

**Model flexibility (what we built).** The implementation exposes three independent "dials" that let the posterior move further per observation: (i) `uniform_mix` in the prior spec (e.g. P9, P10), which guarantees minimum cell mass everywhere; (ii) `likelihood_temperature` in the PyMC Potential, which amplifies updates; (iii) `exploration_mix` in the strategy itself, which adds an $\varepsilon$ uniform component at decision time. On top of those, three interchangeable strategies are evaluated — `FIXED`, `ADAPTIVE` (parameters mutate with the failure streak), and `INFO_GAIN` (entropy-reduction-per-cost à la Bayesian optimal experiment design).

**The structural ceiling.** A Monte Carlo over 200 plausible bomb positions (sampled from P8's prior predictive) reveals an empirical detection-rate vs. budget curve that saturates well below 100%. With the simple `FIXED` strategy: 30% at B=230, **~42% at the operational cap B=530**, 51% at B=800, 66% at B=2000. The more "sophisticated" strategies do not improve on this — the bottleneck is not decision sophistication but **grid coverage**: rectangular missions cost 2–3 per cell, so a budget of 530 visits ~150–200 cells out of 1750 (≈ 8–11 %). No Bayesian decision rule can detect a target it never covers. To exceed 80 % detection one would need B ≥ 1000–1200 — beyond the problem's constraint.

**Recommendation for the deliverable (B ≤ 530).** The operational version now starts with an impact-centered prior (`P12_impact_balanced`) and the `Drift-tail rescue` strategy. This prior is conservative: it places mass around the expected first water-contact point $\mu$, not in the final stress-test zone. If the first missions return $s_t=0$, the update penalizes searched cells through $(1-q_j)^T$ and the strategy moves an auxiliary physical center progressively along $d_\text{long}$. This is the search logic we want to defend: if the plane was not near the conservative impact point, the next coherent hypothesis is that it travelled farther along the drift/down-track direction.

**Empirical Bayes iteration.** The notebook now includes a second-round calibration step: take the Monte Carlo campaigns where the plane was found, build a smoothed KDE prior over those successful cells (`EmpiricalPriorSpec`), mix it with a small uniform component, and rerun the benchmark. This turns the simulation output into a better prior for the next round without breaking Bayesian logic.

**Ten-round calibration result.** We ran 10 Empirical Bayes rounds with a hard budget cap of $B=530$, using separate train and validation bomb scenarios. The empirical prior improves on the training set (best train ≈ 53.3 % detection), but it does not beat the best original model on validation. The validated winner is `P3_quadratic_nowit_informative + explore_x1_5`, with detection rate ≈ 47.5 %, mean cost ≈ 364, and mean 8.2 missions. The best update form is therefore `explore_x1_5`: failed searches are weighted more strongly (`update_temperature=1.5`) while maintaining decision-time exploration (`exploration_mix=0.10`) and stronger avoidance of repeated failed regions (`repetition_penalty=0.10`).

**Drift-tail rescue stress test.** We also evaluate a deliberately hard case where the hidden target is planted in cells with $x\in[18,21]$ and $y\in[9,12]$. This is not passed to the model as evidence. Instead, after failed missions around the nominal peak, the strategy moves an auxiliary physical center step by step along the drift axis: $d_{\text{long, center}} = 2.5 \times \#\text{failed missions}$, capped at 8.0. The posterior is mixed with a local Gaussian around this moving center. Thus the model first searches conservatively near $\mu$; if that fails, it progressively follows the down-track continuation until it reaches the southeast tail. The Streamlit app exposes this as `Drift-tail rescue`: the green rectangle is only a visual stress-test marker, while the update is driven by failed-search likelihoods plus this sequential drift hypothesis.

**Future improvements.** Three concrete extensions: (1) hierarchical detector (priors on $\lambda_0, a_d, a_r$) instead of fixed parameters; (2) full Bayesian model averaging — weight the 10 prior variants by their marginal likelihood and use the averaged $P(Z)$ for proposals; (3) replace the myopic strategy by a finite-horizon POMDP solver to plan a SEQUENCE of missions jointly (would substantially improve the asymptotic detection rate by avoiding redundant coverage).

---

*Code, notebooks (01–04 plus this summary) and an interactive Streamlit visor/simulator are in [`update_version/`](../). The strategies benchmark and detection-rate curve are in [`notebooks/04_pipeline_and_strategy.ipynb`](../notebooks/04_pipeline_and_strategy.ipynb). Reproduce the Streamlit app with:* `streamlit run update_version/app/streamlit_app.py`.
