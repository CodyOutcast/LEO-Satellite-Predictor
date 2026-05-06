from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol

import networkx as nx
import numpy as np

from ._bootstrap import ensure_sim_import_path

ensure_sim_import_path()

from sim import GraphBuilder, LinkModelConfig  # type: ignore

PredictionErrorFn = Callable[[nx.Graph, int, np.random.Generator], nx.Graph]


class Regressor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def _ordered_pair(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _clone_graph(G: nx.Graph) -> nx.Graph:
    H = nx.Graph(**dict(G.graph))
    for n, data in G.nodes(data=True):
        H.add_node(n, **dict(data))
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, **dict(data))
    return H


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _snr_nominal_db(distance_m: float, cfg: LinkModelConfig) -> float:
    d = float(max(distance_m, 1e-9))
    return float(cfg.snr_ref_db - 20.0 * np.log10(d / float(cfg.d_ref_m)))


def _update_weight_fields(attrs: dict, *, snr_db: float, cfg: LinkModelConfig) -> None:
    x = (float(snr_db) - float(cfg.snr_threshold_db)) / float(cfg.snr_softness_db)
    p_succ = _sigmoid(x)
    penalty = float(-np.log(max(p_succ, 1e-12)))
    delay_s = float(attrs.get("delay_s", 0.0))
    weight_rel_s = float(cfg.w_rel_s * penalty)

    attrs["snr_db"] = float(snr_db)
    attrs["p_succ"] = float(p_succ)
    attrs["penalty"] = float(penalty)
    attrs["weight_delay_s"] = float(delay_s)
    attrs["weight_rel_s"] = float(weight_rel_s)
    attrs["weight"] = float(delay_s + weight_rel_s)


def _build_synthetic_edge_attrs(nodes: dict, u: str, v: str, cfg: LinkModelConfig) -> dict:
    node_u = nodes[u]
    node_v = nodes[v]
    kind_u = str(node_u.get("kind"))
    kind_v = str(node_v.get("kind"))

    kind = "access" if {kind_u, kind_v} == {"gs", "sat"} else "isl"

    p_u = np.asarray(node_u.get("pos_ecef_m"), dtype=float).reshape(3)
    p_v = np.asarray(node_v.get("pos_ecef_m"), dtype=float).reshape(3)
    distance_m = float(np.linalg.norm(p_u - p_v))
    delay_s = float(distance_m / 299_792_458.0)

    snr_db = _snr_nominal_db(distance_m, cfg)
    attrs = {
        "kind": kind,
        "range_m": float(distance_m),
        "delay_s": float(delay_s),
        "snr_db_nominal": float(snr_db),
    }
    _update_weight_fields(attrs, snr_db=snr_db, cfg=cfg)
    return attrs


def _apply_pred_error(
    builder: GraphBuilder,
    graphs: list[nx.Graph],
    *,
    t_idx_start: int,
    pred_error: Optional[PredictionErrorFn],
) -> list[nx.Graph]:
    if pred_error is None:
        return graphs

    out: list[nx.Graph] = []
    for k, G in enumerate(graphs):
        idx = int(t_idx_start) + int(k)
        seed_sequence = np.random.SeedSequence([int(builder.cfg.sim.seed), idx, 2])
        rng = np.random.default_rng(seed_sequence)
        out.append(pred_error(_clone_graph(G), idx, rng))
    return out


@dataclass
class GeometricMeanPredictor:
    builder: GraphBuilder
    pred_error: Optional[PredictionErrorFn] = None

    def predict(self, t_idx: int, H: int) -> list[nx.Graph]:
        graphs = self.builder.forecast_at(t_idx, H, mode="mean")
        return _apply_pred_error(self.builder, graphs, t_idx_start=t_idx, pred_error=self.pred_error)


@dataclass
class OracleSamplePredictor:
    builder: GraphBuilder
    pred_error: Optional[PredictionErrorFn] = None

    def predict(self, t_idx: int, H: int) -> list[nx.Graph]:
        graphs = self.builder.forecast_at(t_idx, H, mode="sample")
        return _apply_pred_error(self.builder, graphs, t_idx_start=t_idx, pred_error=self.pred_error)


@dataclass
class LearnedSysIDPredictor:
    builder: GraphBuilder
    model: Regressor
    link_model_cfg: Optional[LinkModelConfig] = None
    pred_error: Optional[PredictionErrorFn] = None

    def __post_init__(self) -> None:
        if self.link_model_cfg is None:
            self.link_model_cfg = self.builder.cfg.link_model

    def predict(self, t_idx: int, H: int) -> list[nx.Graph]:
        if H <= 0:
            raise ValueError("H must be > 0")

        graphs = self.builder.forecast_at(t_idx, H, mode="mean")
        assert self.link_model_cfg is not None

        for G in graphs:
            edges = list(G.edges(data=True))
            if not edges:
                continue

            X = np.asarray(
                [
                    [
                        float(data.get("range_m", 0.0)),
                        1.0 if data.get("kind") == "access" else 0.0,
                        1.0 if data.get("kind") == "isl" else 0.0,
                    ]
                    for _, _, data in edges
                ],
                dtype=float,
            )
            snr_pred = np.asarray(self.model.predict(X), dtype=float).reshape(-1)

            for (u, v, data), snr_db in zip(edges, snr_pred, strict=False):
                data["snr_db_nominal"] = float(snr_db)
                _update_weight_fields(data, snr_db=float(snr_db), cfg=self.link_model_cfg)
                G[u][v].update(data)

        return _apply_pred_error(self.builder, graphs, t_idx_start=t_idx, pred_error=self.pred_error)


def make_pred_error(
    error_rate: float,
    *,
    kind: Literal["snr_jitter", "edge_flip"] = "snr_jitter",
    sigma_db: float = 6.0,
    link_model_cfg: Optional[LinkModelConfig] = None,
) -> PredictionErrorFn:
    if not (0.0 <= float(error_rate) <= 1.0):
        raise ValueError("error_rate must be in [0, 1]")
    if sigma_db < 0:
        raise ValueError("sigma_db must be >= 0")
    if kind not in {"snr_jitter", "edge_flip"}:
        raise ValueError("kind must be 'snr_jitter' or 'edge_flip'")

    cfg = link_model_cfg or LinkModelConfig()
    rate = float(error_rate)

    def _identity(G: nx.Graph, _: int, __: np.random.Generator) -> nx.Graph:
        return _clone_graph(G)

    if rate == 0.0:
        return _identity

    if kind == "snr_jitter":

        def _snr_jitter(G: nx.Graph, _: int, rng: np.random.Generator) -> nx.Graph:
            H = _clone_graph(G)
            for u, v, data in H.edges(data=True):
                base_snr = float(
                    data.get(
                        "snr_db",
                        data.get(
                            "snr_db_nominal",
                            _snr_nominal_db(float(data.get("range_m", 1_000_000.0)), cfg),
                        ),
                    )
                )
                perturbed = float(base_snr + rng.normal(0.0, sigma_db * rate))
                if "snr_db_nominal" not in data:
                    data["snr_db_nominal"] = float(base_snr)
                _update_weight_fields(data, snr_db=perturbed, cfg=cfg)
                H[u][v].update(data)
            return H

        return _snr_jitter

    def _edge_flip(G: nx.Graph, _: int, rng: np.random.Generator) -> nx.Graph:
        H = nx.Graph(**dict(G.graph))
        for n, data in G.nodes(data=True):
            H.add_node(n, **dict(data))

        nodes = list(H.nodes())
        node_data = dict(H.nodes(data=True))

        eligible_pairs: list[tuple[str, str]] = []
        for i, u in enumerate(nodes):
            kind_u = str(node_data[u].get("kind"))
            for v in nodes[i + 1 :]:
                kind_v = str(node_data[v].get("kind"))
                if kind_u == "gs" and kind_v == "gs":
                    continue
                eligible_pairs.append(_ordered_pair(u, v))

        existing = {_ordered_pair(u, v) for u, v in G.edges()}
        if rate >= 1.0 - 1e-12:
            base_prob = 0.25
        else:
            base_prob = (float(len(existing)) / float(len(eligible_pairs))) if eligible_pairs else 0.0

        for u, v in eligible_pairs:
            keep_original = (u, v) in existing
            if rng.random() < rate:
                keep = bool(rng.random() < base_prob)
            else:
                keep = keep_original

            if not keep:
                continue

            if G.has_edge(u, v):
                attrs = dict(G[u][v])
            else:
                attrs = _build_synthetic_edge_attrs(node_data, u, v, cfg)
            H.add_edge(u, v, **attrs)

        return H

    return _edge_flip

