# update_version — Bayesian Spatial Search (Deliverable 2)

Reimplementación del Deliverable 2 con **prior logístico jerárquico** sobre celdas y posterior MCMC sobre los coeficientes del predictor lineal.

## Estructura

```
update_version/
├── data/                              # grid_dataset.csv
├── simulator/                         # réplica local del webapp del profesor
│   ├── grid.py                        #   GridInfo, cells_in_rectangle
│   ├── true_detection.py              #   TrueDetector (binomial oculto)
│   └── environment.py                 #   SearchEnvironment.run_mission(...)
├── modeling/                          # modelo Bayesiano
│   ├── features.py                    #   d_long, d_trans, expected_landing
│   ├── priors_logistic.py             #   P1-P14 + EmpiricalPriorSpec
│   ├── detection.py                   #   4 modelos rho_{t,j}
│   ├── pymc_model.py                  #   build PyMC + run_mcmc + cache
│   └── posterior.py                   #   beta-samples -> P(Z=j | data)
├── strategies/
│   └── next_mission.py                # HDR, drift progresivo, escenarios laterales
├── experiments/
│   ├── compare_priors.py              # Monte Carlo sobre (prior, detector)
│   └── fast_monte_carlo.py            # evaluación rápida + Empirical Bayes
├── notebooks/
│   ├── 00_model_selection_history.ipynb # P1-P14, MCMC, EB y selección
│   ├── 01_physics_and_priors.ipynb      # física, d_long/d_trans y priors
│   ├── 02_detection_and_bayes_update.ipynb # detectores + posterior update
│   ├── 03_progressive_drift_strategy.ipynb # drift/left/right strategies
│   └── 04_operational_summary.ipynb     # resumen operativo final
├── app/
│   └── streamlit_app.py               # visor + simulador interactivo
├── report/
│   └── report.md                      # reporte 2 páginas
├── results/                           # outputs (cache MCMC, CSV, plots)
└── requirements.txt
```

## Instalación

```bash
pip install -r update_version/requirements.txt
```

(Si pymc/pytensor da problemas: instalar con `--no-deps` y luego añadir cons, etuples, logical-unification, minikanren, cachetools, filelock, xarray, cloudpickle, rich.)

## Uso

### 1. Notebooks explicativos

```bash
jupyter notebook update_version/notebooks/01_features_and_priors.ipynb
```

Recorre los notebooks en orden (00 → 04) para entender el pipeline completo. El notebook `00_model_selection_history.ipynb` recupera el proceso de P1-P14, MCMC, Monte Carlo y Empirical Bayes; `04_operational_summary.ipynb` resume la versión operativa final.

### 2. App Streamlit

```bash
streamlit run update_version/app/streamlit_app.py
```

La app muestra el proceso interno de búsqueda: prior, posterior actualizada, misión propuesta, drift progresivo y escenarios centro/derecha/izquierda.

### 3. Experimentos Monte Carlo

```bash
python -m update_version.experiments.compare_priors --trials 5
```

Compara las 32 combinaciones (8 priors x 4 detectores) por tasa de detección y misiones-hasta-detectar.

Evaluación rápida sobre muchos escenarios:

```bash
python -m update_version.experiments.fast_monte_carlo --n-trials 1000 --budget 530
```

## Modelo en una frase

`pi_j = softmax_j(beta_0 + beta_1*d_long + beta_2*d_trans - beta_3*d_long^2 - beta_4*d_trans^2)`
con `(beta, tau_fall, tau_drift)` Bayesianos, actualizados por NUTS marginalizando `Z`. El detector `rho_{t,j}(depth, roughness, effort)` es fijo, calibrado cualitativamente con `previous_missions_reports.pdf`.

Detalle en [`report/report.md`](report/report.md).
