import networkx as nx
from typing import Dict, List, Optional, Sequence
from .flows import Flow


class ReactiveBaselineController:

    def __init__(self, builder, predictor, flows: Sequence[Flow]):
        self.builder = builder
        self.flows = flows
        # predictor

    def step(self, t_idx: int) -> Dict[str, List[str]]:
        G_now = self.builder.graph_at(t_idx, mode="sample")

        results = {}
        for flow in self.flows:
            try:
                if G_now.has_node(flow.src) and G_now.has_node(flow.dst):
                    path = nx.shortest_path(G_now, flow.src, flow.dst, weight='weight')
                    results[flow.name] = path
                else:
                    results[flow.name] = []
            except nx.NetworkXNoPath:
                results[flow.name] = []
        return results


class GreedyHandoverController:


    def __init__(self, builder, predictor, flows: Sequence[Flow]):
        self.builder = builder
        self.flows = flows
        self._prev_paths: Dict[str, List[str]] = {f.name: [] for f in flows}

    def _is_path_valid(self, G: nx.Graph, path: List[str]) -> bool:

        if len(path) < 2: return False
        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i + 1]):
                return False
        return True

    def step(self, t_idx: int) -> Dict[str, List[str]]:
        G_now = self.builder.graph_at(t_idx, mode="sample")
        results = {}

        for flow in self.flows:
            prev_path = self._prev_paths[flow.name]
            if prev_path and self._is_path_valid(G_now, prev_path):
                results[flow.name] = prev_path
            else:
                try:
                    if G_now.has_node(flow.src) and G_now.has_node(flow.dst):
                        path = nx.shortest_path(G_now, flow.src, flow.dst, weight='weight')
                        results[flow.name] = path
                        self._prev_paths[flow.name] = path
                    else:
                        results[flow.name] = []
                        self._prev_paths[flow.name] = []
                except nx.NetworkXNoPath:
                    results[flow.name] = []
                    self._prev_paths[flow.name] = []
        return results