from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx

from ._bootstrap import ensure_sim_import_path
from .flows import Flow
from .interfaces import Predictor

ensure_sim_import_path()

from sim import GraphBuilder  # type: ignore


def _path_edges(path: list[str]) -> list[tuple[str, str]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _path_exists_in_graph(G: nx.Graph, path: list[str]) -> bool:
    if len(path) < 2:
        return False
    for u, v in _path_edges(path):
        if not G.has_edge(u, v):
            return False
    return True


def _access_satellites(path: list[str], flow: Flow) -> tuple[str | None, str | None]:
    src_access = None
    dst_access = None

    if len(path) >= 2 and path[0] == flow.src and path[1].startswith("SAT-"):
        src_access = path[1]
    if len(path) >= 2 and path[-1] == flow.dst and path[-2].startswith("SAT-"):
        dst_access = path[-2]

    return src_access, dst_access


def _handover_penalty(
    prev_access: tuple[str | None, str | None] | None,
    new_access: tuple[str | None, str | None],
    lambda_handover: float,
) -> float:
    if prev_access is None:
        return 0.0

    p_src, p_dst = prev_access
    n_src, n_dst = new_access
    cost = 0.0

    if p_src is not None and n_src is not None and p_src != n_src:
        cost += float(lambda_handover)
    if p_dst is not None and n_dst is not None and p_dst != n_dst:
        cost += float(lambda_handover)

    return cost


@dataclass
class RollingReplanMPC:
    builder: GraphBuilder
    predictor: Predictor
    flows: Sequence[Flow]
    H: int
    lambda_handover: float
    K_candidates: int = 5
    outage_penalty_s: float = 60.0

    _prev_access: dict[str, tuple[str | None, str | None]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.H <= 0:
            raise ValueError("H must be > 0")
        if self.K_candidates <= 0:
            raise ValueError("K_candidates must be > 0")

    def _k_shortest_paths(self, G: nx.Graph, src: str, dst: str) -> list[list[str]]:
        if src not in G or dst not in G:
            return []

        try:
            if not nx.has_path(G, src, dst):
                return []
            generator = nx.shortest_simple_paths(G, src, dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        out: list[list[str]] = []
        try:
            for path in generator:
                out.append(list(path))
                if len(out) >= self.K_candidates:
                    break
        except nx.NetworkXNoPath:
            return []
        return out

    def _score_candidate(
        self,
        flow: Flow,
        path: list[str],
        pred_graphs: list[nx.Graph],
        prev_access: tuple[str | None, str | None] | None,
    ) -> float:
        score = 0.0

        for G in pred_graphs:
            step_cost = 0.0
            feasible = True
            for u, v in _path_edges(path):
                if not G.has_edge(u, v):
                    feasible = False
                    break
                edge = G[u][v]
                step_cost += float(edge.get("weight", edge.get("delay_s", self.outage_penalty_s)))

            if feasible:
                score += step_cost
            else:
                score += float(self.outage_penalty_s)

        score += _handover_penalty(prev_access, _access_satellites(path, flow), self.lambda_handover)
        return score

    def step(self, t_idx: int) -> dict[str, list[str]]:
        G_now = self.builder.graph_at(t_idx, mode="sample")
        pred_graphs = self.predictor.predict(t_idx, self.H)
        if len(pred_graphs) != self.H:
            raise ValueError("predictor returned wrong horizon length")

        results: dict[str, list[str]] = {}

        for flow in self.flows:
            prev_access = self._prev_access.get(flow.name)

            # Invariant anchor: with H=1, lambda=0 we should match predicted Dijkstra,
            # as long as that path is currently feasible.
            if self.H == 1 and self.lambda_handover == 0.0:
                try:
                    dijkstra_path = list(nx.dijkstra_path(pred_graphs[0], flow.src, flow.dst, weight="weight"))
                    if _path_exists_in_graph(G_now, dijkstra_path):
                        results[flow.name] = dijkstra_path
                        self._prev_access[flow.name] = _access_satellites(dijkstra_path, flow)
                        continue
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            cand_now = self._k_shortest_paths(G_now, flow.src, flow.dst)
            cand_pred = self._k_shortest_paths(pred_graphs[0], flow.src, flow.dst)

            seen: set[tuple[str, ...]] = set()
            candidates: list[list[str]] = []
            for path in cand_now + cand_pred:
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(path)

            if not candidates:
                results[flow.name] = []
                continue

            best_path: list[str] = []
            best_score = float("inf")
            for path in candidates:
                if not _path_exists_in_graph(G_now, path):
                    continue
                score = self._score_candidate(flow, path, pred_graphs, prev_access)
                if score < best_score:
                    best_score = score
                    best_path = path

            if best_path:
                results[flow.name] = best_path
                self._prev_access[flow.name] = _access_satellites(best_path, flow)
            else:
                results[flow.name] = []

        return results


@dataclass
class TimeExpandedMPC:
    builder: GraphBuilder
    predictor: Predictor
    flows: Sequence[Flow]
    H: int
    lambda_handover: float
    K_candidates: int = 10
    outage_penalty_s: float = 60.0

    _prev_access: dict[str, tuple[str | None, str | None]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.H <= 0:
            raise ValueError("H must be > 0")
        if self.K_candidates <= 0:
            raise ValueError("K_candidates must be > 0")

    def _aggregate_graph(self, pred_graphs: list[nx.Graph]) -> nx.Graph:
        agg = nx.Graph()

        for G in pred_graphs:
            for n, data in G.nodes(data=True):
                if n not in agg:
                    agg.add_node(n, **dict(data))

        edge_union: set[tuple[str, str]] = set()
        for G in pred_graphs:
            for u, v in G.edges():
                edge_union.add((u, v) if u <= v else (v, u))

        for u, v in sorted(edge_union):
            total_weight = 0.0
            base_attrs: dict = {}
            for G in pred_graphs:
                if G.has_edge(u, v):
                    edge = dict(G[u][v])
                    total_weight += float(edge.get("weight", edge.get("delay_s", self.outage_penalty_s)))
                    if not base_attrs:
                        base_attrs = edge
                else:
                    total_weight += float(self.outage_penalty_s)

            attrs = dict(base_attrs)
            attrs["weight"] = float(total_weight)
            agg.add_edge(u, v, **attrs)

        return agg

    def _k_shortest_paths(self, G: nx.Graph, src: str, dst: str) -> list[list[str]]:
        if src not in G or dst not in G:
            return []

        try:
            if not nx.has_path(G, src, dst):
                return []
            generator = nx.shortest_simple_paths(G, src, dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        out: list[list[str]] = []
        try:
            for path in generator:
                out.append(list(path))
                if len(out) >= self.K_candidates:
                    break
        except nx.NetworkXNoPath:
            return []
        return out

    def step(self, t_idx: int) -> dict[str, list[str]]:
        G_now = self.builder.graph_at(t_idx, mode="sample")
        pred_graphs = self.predictor.predict(t_idx, self.H)
        if len(pred_graphs) != self.H:
            raise ValueError("predictor returned wrong horizon length")

        agg = self._aggregate_graph(pred_graphs)
        results: dict[str, list[str]] = {}

        for flow in self.flows:
            prev_access = self._prev_access.get(flow.name)
            candidates = self._k_shortest_paths(agg, flow.src, flow.dst)
            if not candidates:
                results[flow.name] = []
                continue

            best_path: list[str] = []
            best_score = float("inf")

            for path in candidates:
                if not _path_exists_in_graph(G_now, path):
                    continue

                path_weight = 0.0
                for u, v in _path_edges(path):
                    if not agg.has_edge(u, v):
                        path_weight += float(self.outage_penalty_s)
                        continue
                    path_weight += float(agg[u][v].get("weight", self.outage_penalty_s))

                handover_cost = _handover_penalty(prev_access, _access_satellites(path, flow), self.lambda_handover)
                score = float(path_weight + handover_cost)
                if score < best_score:
                    best_score = score
                    best_path = path

            if best_path:
                results[flow.name] = best_path
                self._prev_access[flow.name] = _access_satellites(best_path, flow)
            else:
                results[flow.name] = []

        return results

