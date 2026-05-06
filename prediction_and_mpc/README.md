# Prediction and MPC

This folder is the prediction and control layer for the LEO network simulator in
`graph_constructor_and_link_model/`.

Plainly:
- the simulator builds the network graph over time
- this package predicts future graphs
- this package picks routes using those predictions
- this package measures outage, latency, and handovers
- this package can train a simple learned link predictor from simulator telemetry

## What is in here

- `pred_mpc/`
  The Python package.
- `scripts/train_predictor.py`
  Generates telemetry and trains the learned predictor.
- `scripts/evaluate.py`
  Runs the controller/predictor experiments and writes raw results.
- `scripts/plot_results.py`
  Turns raw results into plots and one summary data file.
- `tests/`
  Unit tests for predictors, controllers, runner, and metrics.
- `outputs/`
  Saved evaluation results and plots.
- `checkpoints/`
  Saved trained model files.

## Main idea

There are two predictors and two controllers.

Predictors:
- `geometric_mean`
  Uses the simulator's own deterministic mean forecast directly.
- `learned_sysid`
  Uses the same forecasted topology, but replaces the link SNR/weight estimate
  with a learned model trained from telemetry.

Controllers:
- `rolling_replan`
  Main MPC-like method. At each timestep it scores a small set of candidate
  routes over a future horizon and picks the best one for now.
- `time_expanded`
  Simpler comparison method. It aggregates predicted costs across the horizon
  and routes on that aggregate.

So the package is comparing:
- different ways to predict the future network
- different ways to use those predictions for routing

## Current experiment setup

The current training and evaluation scripts use:
- 3 ground stations: `SF`, `LON`, `SIN`
- 3 flows: `SF-LON`, `SF-SIN`, `LON-SIN`
- a 6-plane, 12-satellites-per-plane constellation
- `isl_mode="neighbor"`
- relaxed access settings compared with the original sparse setup

This setup was chosen because the earlier sparse version caused nearly constant
outage, which made the controller comparison meaningless.

## How the pipeline works

### 1. Train a model

Run:

```powershell
python prediction_and_mpc/scripts/train_predictor.py
```

This will:
- build a simulator configuration
- collect per-edge telemetry from `mode="sample"`
- train a regression model to predict observed `snr_db`
- save the model to `prediction_and_mpc/checkpoints/learned_sysid.joblib`

Important note:
- the training script has a train/test split
- it does not only report training error
- the learned model is used only for link-quality prediction
- topology still comes from the simulator forecast

### 2. Run evaluation

Quick smoke test:

```powershell
python prediction_and_mpc/scripts/evaluate.py --quick
```

Full sweep:

```powershell
python prediction_and_mpc/scripts/evaluate.py
```

`--quick` uses:
- 1 seed
- 30 timesteps
- 3 error rates

Full run uses:
- 5 seeds
- 180 timesteps
- 5 error rates

The script prints progress and ETA while it runs.

If the checkpoint file does not exist yet, `evaluate.py` will train one
automatically before running the sweep.

Raw results are written to:
- `prediction_and_mpc/outputs/eval_records.jsonl`

Each result row corresponds to one combination of:
- seed
- flow
- controller
- predictor
- forecast error rate

## What the metrics mean

The main saved metrics are:
- `outage_prob`
  Fraction of timesteps where the chosen path is invalid in the realized graph.
- `p50_latency_s`, `p95_latency_s`, `p99_latency_s`
  End-to-end propagation-delay statistics over successful deliveries.
- `handover_rate_per_min`
  How often access satellites change.
- `regret_rate`
  Fraction of handovers that look unnecessary under the dwell-window rule.

## Plotting results

Run:

```powershell
python prediction_and_mpc/scripts/plot_results.py
```

This reads `eval_records.jsonl` and writes:
- `prediction_and_mpc/outputs/eval_overview.png`
- `prediction_and_mpc/outputs/robustness_sweep.png`
- `prediction_and_mpc/outputs/plot_results_data.json`

`plot_results_data.json` stores the aggregated numbers used in the plots.

## Files your teammate will most likely touch

- `scripts/evaluate.py`
  Change experiment settings, constellation density, or sweep size.
- `scripts/train_predictor.py`
  Change training size or checkpoint path.
- `pred_mpc/controllers.py`
  Change routing logic.
- `pred_mpc/predictors.py`
  Change forecast logic or forecast-error injection.
- `pred_mpc/metrics.py`
  Change how results are summarized.

## Tests

Run:

```powershell
python -m pytest prediction_and_mpc/tests -q -p no:cacheprovider
```

These tests cover:
- predictor behavior
- controller invariants
- simulation runner behavior
- metric calculations

## Setup

From the repository root:

```powershell
pip install -r graph_constructor_and_link_model/requirements.txt
pip install -r prediction_and_mpc/requirements.txt
```

This package imports the simulator from the sibling folder automatically, so
there is no need to install `graph_constructor_and_link_model` as a separate
editable package first.

## Short summary for a teammate

If you only need the basic workflow, do this:

```powershell
python -m pytest prediction_and_mpc/tests -q -p no:cacheprovider
python prediction_and_mpc/scripts/train_predictor.py
python prediction_and_mpc/scripts/evaluate.py --quick
python prediction_and_mpc/scripts/plot_results.py
```

Then check:
- `prediction_and_mpc/outputs/eval_records.jsonl`
- `prediction_and_mpc/outputs/plot_results_data.json`
- `prediction_and_mpc/outputs/eval_overview.png`
- `prediction_and_mpc/outputs/robustness_sweep.png`
