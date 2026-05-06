# Prediction + MPC

This package adds the prediction and control layer on top of the existing
snapshot-graph simulator in `graph_constructor_and_link_model/`.

It provides:
- flow definitions and controller interfaces
- predictors built on `GraphBuilder.forecast_at(...)`
- horizon-aware controllers
- a simulation runner and metrics
- telemetry generation and learned system-identification training

## Layout

- `pred_mpc/`: importable package
- `scripts/`: training, evaluation, and plotting entry points
- `tests/`: unit and smoke tests against the real simulator

## Quickstart

From the repository root:

```powershell
pip install -r graph_constructor_and_link_model/requirements.txt
pip install -r prediction_and_mpc/requirements.txt
python -m pytest prediction_and_mpc/tests -q -p no:cacheprovider
python prediction_and_mpc/scripts/train_predictor.py
python prediction_and_mpc/scripts/evaluate.py --quick
python prediction_and_mpc/scripts/plot_results.py
```

The package bootstraps the sibling `graph_constructor_and_link_model/`
directory onto `sys.path` so it can import `sim` without requiring an
editable install.
