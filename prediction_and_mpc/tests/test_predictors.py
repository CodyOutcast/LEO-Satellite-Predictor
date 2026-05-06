from __future__ import annotations

import networkx as nx
import numpy as np

from sim import (  # type: ignore
    ConstellationConfig,
    GraphBuilder,
    GraphBuilderConfig,
    GroundStation,
    LinkConstraints,
    LinkModelConfig,
    SimConfig,
)

from pred_mpc.predictors import (
    GeometricMeanPredictor,
    LearnedSysIDPredictor,
    make_pred_error,
)
from pred_mpc.train import generate_telemetry, train_learned_sysid


def _make_builder(seed: int = 0, sigma_db: float = 0.0) -> GraphBuilder:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=2.0, t_end_s=400.0, seed=seed),
        constellation=ConstellationConfig(
            num_planes=2,
            sats_per_plane=6,
            altitude_m=550_000.0,
            inclination_deg=53.0,
            phase_offset_deg=5.0,
        ),
        ground_stations=[
            GroundStation("SF", 37.7749, -122.4194, 0.0),
            GroundStation("LON", 51.5072, -0.1276, 0.0),
        ],
        links=LinkConstraints(theta_min_deg=5.0, isl_range_max_m=6_000_000.0, isl_mode="neighbor"),
        link_model=LinkModelConfig(sigma_db=sigma_db, p_edge_fail=0.0, snr_down_threshold_db=None),
    )
    return GraphBuilder(cfg)


def _edge_keys(G: nx.Graph) -> list[tuple[str, str]]:
    return sorted((u, v) if u <= v else (v, u) for u, v in G.edges())


def _edge_signature(G: nx.Graph) -> list[tuple[tuple[str, str], float, float, float]]:
    out = []
    for u, v, d in G.edges(data=True):
        key = (u, v) if u <= v else (v, u)
        out.append((key, float(d.get("snr_db", 0.0)), float(d.get("p_succ", 0.0)), float(d.get("weight", 0.0))))
    return sorted(out, key=lambda x: (x[0][0], x[0][1]))


def test_geometric_mean_predictor_horizon() -> None:
    builder = _make_builder(seed=1)
    predictor = GeometricMeanPredictor(builder)
    graphs = predictor.predict(5, 4)

    assert len(graphs) == 4
    assert graphs[0].graph["t_idx"] == 5
    assert graphs[-1].graph["t_idx"] == 8


def test_learned_sysid_predictor_recovers_snr_curve() -> None:
    builder = _make_builder(seed=2, sigma_db=0.0)
    df = generate_telemetry(builder, 0, 120)
    model, metrics = train_learned_sysid(df, save_path=None, model_kind="linear", random_state=0)

    assert metrics["mae_test_db"] < 1.5

    predictor = LearnedSysIDPredictor(builder, model)
    graphs = predictor.predict(20, 2)

    assert len(graphs) == 2
    for G in graphs:
        if G.number_of_edges() == 0:
            continue
        _, _, data = next(iter(G.edges(data=True)))
        assert "snr_db" in data
        assert "weight" in data


def test_make_pred_error_zero_is_identity() -> None:
    builder = _make_builder(seed=3)
    G = builder.graph_at(10, mode="mean")

    pred_error = make_pred_error(0.0, kind="snr_jitter", link_model_cfg=builder.cfg.link_model)
    H = pred_error(G, 10, np.random.default_rng(0))

    assert _edge_signature(G) == _edge_signature(H)


def test_make_pred_error_edge_flip_full_rate_independent_of_input_edges() -> None:
    builder = _make_builder(seed=4)
    G1 = builder.graph_at(10, mode="mean")
    G2 = builder.graph_at(11, mode="mean")

    pred_error = make_pred_error(1.0, kind="edge_flip", link_model_cfg=builder.cfg.link_model)
    H1 = pred_error(G1, 10, np.random.default_rng(123))
    H2 = pred_error(G2, 10, np.random.default_rng(123))

    assert _edge_keys(H1) == _edge_keys(H2)
