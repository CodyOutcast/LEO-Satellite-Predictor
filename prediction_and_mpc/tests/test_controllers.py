from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from pred_mpc.controllers import RollingReplanMPC, TimeExpandedMPC
from pred_mpc.flows import Flow


class SequenceBuilder:
    def __init__(self, graphs: list[nx.Graph]):
        self.graphs = graphs
        self.cfg = SimpleNamespace(sim=SimpleNamespace(seed=0, dt_s=1.0, num_steps=len(graphs)))

    def graph_at(self, t_idx: int, *, mode: str = "sample") -> nx.Graph:
        return self.graphs[t_idx].copy()


class SequencePredictor:
    def __init__(self, graphs: list[nx.Graph]):
        self.graphs = graphs

    def predict(self, t_idx: int, H: int) -> list[nx.Graph]:
        out = []
        for k in range(H):
            idx = min(t_idx + k, len(self.graphs) - 1)
            out.append(self.graphs[idx].copy())
        return out


def _make_graph(*, w_sat1: float, w_sat2: float) -> nx.Graph:
    G = nx.Graph()
    G.add_node("GS-A", kind="gs", pos_ecef_m=[0.0, 0.0, 0.0])
    G.add_node("GS-B", kind="gs", pos_ecef_m=[1.0, 0.0, 0.0])
    G.add_node("SAT-1", kind="sat", pos_ecef_m=[0.0, 1.0, 0.0])
    G.add_node("SAT-2", kind="sat", pos_ecef_m=[1.0, 1.0, 0.0])

    G.add_edge("GS-A", "SAT-1", kind="access", delay_s=0.01, weight=w_sat1)
    G.add_edge("SAT-1", "GS-B", kind="access", delay_s=0.01, weight=w_sat1)
    G.add_edge("GS-A", "SAT-2", kind="access", delay_s=0.01, weight=w_sat2)
    G.add_edge("SAT-2", "GS-B", kind="access", delay_s=0.01, weight=w_sat2)
    G.add_edge("SAT-1", "SAT-2", kind="isl", delay_s=0.05, weight=10.0)
    return G


def test_controllers_match_dijkstra_when_h1_lambda0() -> None:
    G = _make_graph(w_sat1=1.0, w_sat2=2.0)
    flow = Flow(name="A-B", src="GS-A", dst="GS-B")

    builder = SequenceBuilder([G])
    predictor = SequencePredictor([G])
    expected = nx.dijkstra_path(G, flow.src, flow.dst, weight="weight")

    rolling = RollingReplanMPC(builder, predictor, [flow], H=1, lambda_handover=0.0)
    teg = TimeExpandedMPC(builder, predictor, [flow], H=1, lambda_handover=0.0)

    assert rolling.step(0)[flow.name] == expected
    assert teg.step(0)[flow.name] == expected


def test_rolling_replan_prefers_stability_with_large_lambda() -> None:
    G0 = _make_graph(w_sat1=1.0, w_sat2=2.0)
    G1 = _make_graph(w_sat1=2.0, w_sat2=1.7)
    flow = Flow(name="A-B", src="GS-A", dst="GS-B")

    builder = SequenceBuilder([G0, G1])
    predictor = SequencePredictor([G0, G1])

    controller = RollingReplanMPC(builder, predictor, [flow], H=1, lambda_handover=10.0)

    path0 = controller.step(0)[flow.name]
    path1 = controller.step(1)[flow.name]

    assert path0[1] == "SAT-1"
    assert path1[1] == "SAT-1"


def test_controller_returns_outage_path_when_disconnected() -> None:
    G = nx.Graph()
    G.add_node("GS-A", kind="gs", pos_ecef_m=[0.0, 0.0, 0.0])
    G.add_node("GS-B", kind="gs", pos_ecef_m=[1.0, 0.0, 0.0])

    flow = Flow(name="A-B", src="GS-A", dst="GS-B")
    builder = SequenceBuilder([G])
    predictor = SequencePredictor([G])
    controller = RollingReplanMPC(builder, predictor, [flow], H=1, lambda_handover=0.0)

    assert controller.step(0)[flow.name] == []
