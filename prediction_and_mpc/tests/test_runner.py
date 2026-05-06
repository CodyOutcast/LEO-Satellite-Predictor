from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from pred_mpc.flows import Flow
from pred_mpc.runner import SimulationRunner


class SequenceBuilder:
    def __init__(self, graphs: list[nx.Graph]):
        self.graphs = graphs
        self.cfg = SimpleNamespace(sim=SimpleNamespace(seed=0, dt_s=1.0, num_steps=len(graphs)))

    def graph_at(self, t_idx: int, *, mode: str = "sample") -> nx.Graph:
        return self.graphs[t_idx].copy()


class FixedController:
    def __init__(self, paths_by_t: dict[int, dict[str, list[str]]]):
        self.paths_by_t = paths_by_t

    def step(self, t_idx: int) -> dict[str, list[str]]:
        return self.paths_by_t[t_idx]


def _truth_graph(include_terminal_edge: bool) -> nx.Graph:
    G = nx.Graph()
    G.add_node("GS-A", kind="gs", pos_ecef_m=[0.0, 0.0, 0.0])
    G.add_node("SAT-1", kind="sat", pos_ecef_m=[0.0, 1.0, 0.0])
    G.add_node("GS-B", kind="gs", pos_ecef_m=[1.0, 0.0, 0.0])

    G.add_edge("GS-A", "SAT-1", kind="access", delay_s=0.01, weight=0.01)
    if include_terminal_edge:
        G.add_edge("SAT-1", "GS-B", kind="access", delay_s=0.02, weight=0.02)
    return G


def test_runner_produces_records_and_flags_invalid_path() -> None:
    flow = Flow(name="A-B", src="GS-A", dst="GS-B")

    G0 = _truth_graph(include_terminal_edge=True)
    G1 = _truth_graph(include_terminal_edge=False)

    builder = SequenceBuilder([G0, G1])
    controller = FixedController(
        {
            0: {flow.name: ["GS-A", "SAT-1", "GS-B"]},
            1: {flow.name: ["GS-A", "SAT-1", "GS-B"]},
        }
    )

    runner = SimulationRunner(builder, controller, [flow])
    records = runner.run(t_start=0, t_end=2)

    assert len(records) == 2
    assert records[0].outage is False
    assert records[0].latency_s == 0.03
    assert records[1].outage is True
    assert records[1].latency_s is None
