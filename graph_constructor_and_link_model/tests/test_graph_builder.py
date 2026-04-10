from __future__ import annotations

from typing import Any

import networkx as nx

from sim import (
    ConstellationConfig,
    GraphBuilder,
    GraphBuilderConfig,
    GroundStation,
    LinkConstraints,
    LinkModelConfig,
    NodeFailureWindow,
    SimConfig,
)


def _edge_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _edge_signature(G: nx.Graph) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    items = []
    for u, v, data in G.edges(data=True):
        key = _edge_key(u, v)
        # Only compare a stable subset of attrs that should match exactly
        attrs = {
            "kind": data.get("kind"),
            "range_m": float(data.get("range_m")),
            "delay_s": float(data.get("delay_s")),
            "snr_db": float(data.get("snr_db")),
            "p_succ": float(data.get("p_succ")),
            "penalty": float(data.get("penalty")),
            "weight": float(data.get("weight")),
        }
        items.append((key, attrs))
    return sorted(items, key=lambda x: (x[0][0], x[0][1], x[1]["kind"]))


def test_graph_builder_reproducible_sample_mode() -> None:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=1.0, t_end_s=60.0, seed=123),
        constellation=ConstellationConfig(
            num_planes=2,
            sats_per_plane=6,
            altitude_m=550_000.0,
            inclination_deg=53.0,
            phase_offset_deg=15.0,
        ),
        ground_stations=[
            GroundStation("A", 0.0, 0.0, 0.0),
            GroundStation("B", 0.0, 90.0, 0.0),
        ],
        links=LinkConstraints(theta_min_deg=10.0, isl_range_max_m=6_000_000.0, isl_mode="neighbor"),
        link_model=LinkModelConfig(sigma_db=3.0, p_edge_fail=0.0, snr_down_threshold_db=None),
    )

    b = GraphBuilder(cfg)
    G1 = b.graph_at(10, mode="sample")
    G2 = b.graph_at(10, mode="sample")

    assert _edge_signature(G1) == _edge_signature(G2)


def test_mean_vs_sample_same_geometry_when_no_down_or_fail() -> None:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=1.0, t_end_s=60.0, seed=7),
        constellation=ConstellationConfig(num_planes=2, sats_per_plane=4, altitude_m=550_000.0, inclination_deg=53.0),
        ground_stations=[GroundStation("A", 0.0, 0.0, 0.0)],
        links=LinkConstraints(theta_min_deg=10.0, isl_range_max_m=6_000_000.0, isl_mode="neighbor"),
        link_model=LinkModelConfig(sigma_db=4.0, p_edge_fail=0.0, snr_down_threshold_db=None),
    )

    b = GraphBuilder(cfg)
    G_mean = b.graph_at(3, mode="mean")
    G_samp = b.graph_at(3, mode="sample")

    mean_edges = sorted(_edge_key(u, v) for u, v in G_mean.edges())
    samp_edges = sorted(_edge_key(u, v) for u, v in G_samp.edges())

    assert mean_edges == samp_edges


def test_forecast_at_returns_h_graphs() -> None:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=2.0, t_end_s=60.0, seed=42),
        constellation=ConstellationConfig(num_planes=1, sats_per_plane=6, altitude_m=550_000.0, inclination_deg=53.0),
        ground_stations=[GroundStation("A", 0.0, 0.0, 0.0)],
    )

    b = GraphBuilder(cfg)
    graphs = b.forecast_at(5, 4, mode="mean")

    assert len(graphs) == 4
    assert graphs[0].graph["t_idx"] == 5
    assert graphs[-1].graph["t_idx"] == 8


def test_edge_fail_reproducible_when_enabled() -> None:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=1.0, t_end_s=10.0, seed=99),
        constellation=ConstellationConfig(num_planes=2, sats_per_plane=4, altitude_m=550_000.0, inclination_deg=53.0),
        ground_stations=[GroundStation("A", 0.0, 0.0, 0.0)],
        links=LinkConstraints(theta_min_deg=0.0, isl_range_max_m=1e9, earth_occlusion=False, isl_mode="neighbor"),
        link_model=LinkModelConfig(sigma_db=0.0, p_edge_fail=0.35, snr_down_threshold_db=None),
    )

    b = GraphBuilder(cfg)
    G1 = b.graph_at(0, mode="sample")
    G2 = b.graph_at(0, mode="sample")
    assert _edge_signature(G1) == _edge_signature(G2)


def test_snr_down_threshold_drops_edges_in_sample_only() -> None:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=1.0, t_end_s=10.0, seed=1),
        constellation=ConstellationConfig(num_planes=2, sats_per_plane=4, altitude_m=550_000.0, inclination_deg=53.0),
        ground_stations=[GroundStation("A", 0.0, 0.0, 0.0)],
        links=LinkConstraints(theta_min_deg=0.0, isl_range_max_m=1e9, earth_occlusion=False, isl_mode="neighbor"),
        link_model=LinkModelConfig(sigma_db=0.0, p_edge_fail=0.0, snr_down_threshold_db=1_000.0),
    )

    b = GraphBuilder(cfg)
    G_mean = b.graph_at(0, mode="mean")
    G_sample = b.graph_at(0, mode="sample")

    assert G_mean.number_of_edges() > 0
    assert G_sample.number_of_edges() == 0


def test_node_failure_schedule_isolates_node_in_sample_only() -> None:
    failed_sat = "SAT-P0-S0"
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=1.0, t_end_s=10.0, seed=5),
        constellation=ConstellationConfig(num_planes=1, sats_per_plane=6, altitude_m=550_000.0, inclination_deg=53.0),
        ground_stations=[GroundStation("A", 0.0, 0.0, 0.0)],
        links=LinkConstraints(theta_min_deg=0.0, isl_range_max_m=1e9, earth_occlusion=False, isl_mode="neighbor"),
        link_model=LinkModelConfig(
            sigma_db=0.0,
            p_edge_fail=0.0,
            snr_down_threshold_db=None,
            node_failure_schedule=(NodeFailureWindow(0, 0, failed_sat),),
        ),
    )

    b = GraphBuilder(cfg)
    G_mean = b.graph_at(0, mode="mean")
    G_sample = b.graph_at(0, mode="sample")

    assert G_mean.nodes[failed_sat]["failed"] is False
    assert G_mean.degree[failed_sat] > 0

    assert G_sample.nodes[failed_sat]["failed"] is True
    assert G_sample.degree[failed_sat] == 0


def test_access_edges_decrease_with_theta_min() -> None:
    # Construct a deterministic case at t=0 where SAT-P0-S0 is overhead of GS-EQ,
    # and SAT-P0-S1/S23 are near-horizon (visible for theta_min=0, but not theta_min=20).
    base = dict(
        sim=SimConfig(dt_s=1.0, t_end_s=10.0, seed=0),
        constellation=ConstellationConfig(
            num_planes=1,
            sats_per_plane=24,
            altitude_m=550_000.0,
            inclination_deg=0.0,
            raan_offset_deg=0.0,
            phase_offset_deg=0.0,
        ),
        ground_stations=[GroundStation("EQ", 0.0, 0.0, 0.0)],
        link_model=LinkModelConfig(sigma_db=0.0, p_edge_fail=0.0, snr_down_threshold_db=None),
    )

    cfg0 = GraphBuilderConfig(
        **base,
        links=LinkConstraints(theta_min_deg=0.0, isl_range_max_m=1.0, earth_occlusion=False, isl_mode="neighbor"),
    )
    cfg20 = GraphBuilderConfig(
        **base,
        links=LinkConstraints(theta_min_deg=20.0, isl_range_max_m=1.0, earth_occlusion=False, isl_mode="neighbor"),
    )

    b0 = GraphBuilder(cfg0)
    b20 = GraphBuilder(cfg20)
    G0 = b0.graph_at(0, mode="mean")
    G20 = b20.graph_at(0, mode="mean")

    access0 = sum(1 for _, _, d in G0.edges(data=True) if d.get("kind") == "access")
    access20 = sum(1 for _, _, d in G20.edges(data=True) if d.get("kind") == "access")

    assert access0 > access20
    assert access20 >= 1
