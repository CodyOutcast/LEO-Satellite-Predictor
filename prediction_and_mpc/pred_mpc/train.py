from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._bootstrap import ensure_sim_import_path

ensure_sim_import_path()

from sim import GraphBuilder  # type: ignore


def generate_telemetry(builder: GraphBuilder, t_start: int, t_end: int) -> pd.DataFrame:
    if t_end <= t_start:
        raise ValueError("t_end must be > t_start")

    rows: list[dict] = []
    for t_idx in range(int(t_start), int(t_end)):
        G = builder.graph_at(t_idx, mode="sample")
        for u, v, data in G.edges(data=True):
            kind = str(data.get("kind", ""))
            rows.append(
                {
                    "t_idx": int(t_idx),
                    "u": u,
                    "v": v,
                    "range_m": float(data.get("range_m", 0.0)),
                    "kind": kind,
                    "is_access": 1.0 if kind == "access" else 0.0,
                    "is_isl": 1.0 if kind == "isl" else 0.0,
                    "snr_db_observed": float(data.get("snr_db", np.nan)),
                    "snr_db_nominal": float(data.get("snr_db_nominal", np.nan)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No telemetry edges collected; check the builder config")
    return df


def _feature_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"range_m", "is_access", "is_isl", "snr_db_observed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing telemetry columns: {sorted(missing)}")

    clean = df.dropna(subset=["range_m", "is_access", "is_isl", "snr_db_observed"]).copy()
    X = clean[["range_m", "is_access", "is_isl"]].to_numpy(dtype=float)
    y = clean["snr_db_observed"].to_numpy(dtype=float)
    return X, y


def train_learned_sysid(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    model_kind: Literal["mlp", "linear"] = "mlp",
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[Pipeline, dict[str, float]]:
    X, y = _feature_target(df)
    if len(y) < 50:
        raise ValueError("Need at least 50 telemetry samples for a stable fit")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    if model_kind == "mlp":
        regressor = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=5e-3,
            max_iter=1200,
            random_state=int(random_state),
        )
    elif model_kind == "linear":
        regressor = LinearRegression()
    else:
        raise ValueError("model_kind must be 'mlp' or 'linear'")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    metrics = {
        "mae_train_db": float(mean_absolute_error(y_train, yhat_train)),
        "mae_test_db": float(mean_absolute_error(y_test, yhat_test)),
        "r2_train": float(r2_score(y_train, yhat_train)),
        "r2_test": float(r2_score(y_test, yhat_test)),
        "n_train": float(len(y_train)),
        "n_test": float(len(y_test)),
    }

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    return model, metrics


def load_learned_sysid(path: str | Path) -> Pipeline:
    return joblib.load(Path(path))
