# gemini_proyect — Bayesian aircraft-debris search sandbox

A local simulation environment for **Deliverable 2 — Bayesian Inference** at
UB. It lets you build, run, and compare prior distributions and search
strategies against a faithful replica of the professor's search-missions
Streamlit webapp before spending real budget on the submission.

## Layout

```
gemini_proyect/
├── data/                       # copy of grid_dataset.csv
├── simulator/                  # local replica of the webapp
│   ├── grid.py                 #   load grid, indexing helpers
│   ├── true_detection.py       #   ground-truth detection model
│   └── environment.py          #   SearchEnvironment.run_mission(...)
├── modeling/                   # the inference layer (the "student" code)
│   ├── priors.py               #   uniform / drift / drift+witnesses / sinking
│   ├── detection.py            #   our rho_{t,j} model
│   └── bayes.py                #   posterior_update from a mission history
├── strategies/
│   └── strategies.py           #   max-expected-detection, info-gain, max-posterior
├── experiments/
│   ├── single_run.py           #   one full simulated campaign
│   └── compare_strategies.py   #   Monte Carlo over (strategy, prior) pairs
├── notebooks/
│   └── exploration.ipynb       #   end-to-end walkthrough
└── results/                    #   CSVs from compare_strategies runs
```

## How the simulator mirrors the real webapp

| Webapp concept                | Local equivalent                                       |
|-------------------------------|--------------------------------------------------------|
| `Team status` panel           | `SearchEnvironment.budget_total`, `budget_used`, `history` |
| `New mission` form            | `env.run_mission(x_min, x_max, y_min, y_max, effort)`  |
| `s_t` returned by server      | `MissionRecord.s_t` (sampled from the true detector)   |
| Cost rule `cost = e * |R|`    | enforced inside `run_mission`                          |
| Hidden "bomb" location        | planted via `env.plant_object(prior=...)`              |

The crucial difference: locally we *control* the hidden cell and the data-
generating process (`TrueDetector`). That lets us run thousands of trials
to rank strategies without touching the shared budget on the real server.

## Modeling decisions (Tasks 1–2 of the deliverable)

### Prior

Built from the accident information in `deliverable2.ipynb`:

```
mu        = x_E + tau_fall  * (v_plane + w_wind * v_wind)   # impact point
mu_seabed = mu  + tau_drift * v_drift                       # post-impact drift
```

Around `mu_seabed` we place an anisotropic Gaussian whose along-trajectory
standard deviation is larger than the cross-trajectory one (we know the
direction of motion well, less so how far). The witness statements shift
the centre slightly forward and laterally; the magnitude of the shift is
small relative to the spread, consistent with the statements being
qualitative.

An optional environmental factor `exp(depth_bias * depth)` lets us encode
a soft prior that a sinking object ends up in deeper cells.

### Detection model

For a cell with normalised depth `d` and roughness `r`, at effort `e`:

```
rho(d, r, e) = 1 - exp( -lambda_0 (1-d)^a_d (1-r)^a_r * e )
```

Properties:

* always in `[0, 1)`,
* monotone in effort, in `(1-d)`, and in `(1-r)`,
* effort acts as the number of independent sensor passes,
* qualitatively matches every previous mission report (report 2: shallow +
  smooth + high effort → near-1; report 3: deep + rough + high effort →
  near-0; report 4: intermediate, needed repetition → moderate).

### Posterior

With conditional independence across missions given the (unknown) true
cell `Z`,

```
L_j   = prod_t  q_{tj}^{s_t} * (1 - q_{tj})^{1 - s_t}
pi_j  *=  L_j   (then renormalise)
```

implemented in log space for numerical stability.

### Strategies

* `propose_max_expected_detection` — argmax of `sum_j pi_j q_{tj} / cost`.
* `propose_info_gain` — argmax of expected entropy reduction per unit cost
  (one-step Bayesian optimal-experiment).
* `propose_max_posterior_rect` — argmax of `sum_j pi_j q_{tj}` ignoring
  cost; tends to prefer larger rectangles.

All strategies enumerate axis-aligned rectangles of allowed sizes and
score them in vectorised form via 2-D integral images, so each call is
O(rectangles) with constant work per rectangle.

## How to use

```bash
# 1. one simulated campaign
python -m gemini_proyect.experiments.single_run

# 2. Monte Carlo strategy/prior comparison (writes results/summary.csv)
python -m gemini_proyect.experiments.compare_strategies

# 3. exploration notebook
jupyter notebook gemini_proyect/notebooks/exploration.ipynb
```

## Going from simulator to the real submission

The `SearchEnvironment` mirrors the webapp API exactly, so the workflow
for the real deliverable is:

1. Pick a prior and a strategy after Monte Carlo validation here.
2. Compute the proposal rectangle from the current posterior.
3. Enter it into the real Streamlit form and read back `s_t`.
4. Append a `MissionRecord` to `history` and call `posterior_update`.
5. Repeat until the budget is exhausted or `s_t = 1`.

The notebook `notebooks/exploration.ipynb` ends with a section showing
exactly this mapping.
