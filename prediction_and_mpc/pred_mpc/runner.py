from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from ._bootstrap import ensure_sim_import_path
from .flows import Flow
from .interfaces import Controller

ensure_sim_import_path()

from sim import GraphBuilder  # type: ignore


@dataclass
class StepRecord:
    t_idx: int
    flow_name: str
    outage: bool
    latency_s: float | None
    path: list[str]
    access_sat_src: str | None
    access_sat_dst: str | None
    handover_at_src: bool
    handover_at_dst: bool


def _path_edges(path: list[str]) -> list[tuple[str, str]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _extract_access(path: list[str], flow: Flow) -> tuple[str | None, str | None]:
    src_access = None
    dst_access = None

    if len(path) >= 2 and path[0] == flow.src and path[1].startswith("SAT-"):
        src_access = path[1]
    if len(path) >= 2 and path[-1] == flow.dst and path[-2].startswith("SAT-"):
        dst_access = path[-2]

    return src_access, dst_access


class SimulationRunner:
    def __init__(self, builder: GraphBuilder, controller: Controller, flows: Sequence[Flow]):
        self.builder = builder
        self.controller = controller
        self.flows = list(flows)

    def _validate_and_latency(self, path: list[str], flow: Flow, G_truth: nx.Graph) -> tuple[bool, float | None]:
        if len(path) < 2:
            return False, None
        if path[0] != flow.src or path[-1] != flow.dst:
            return False, None

        latency_s = 0.0
        for u, v in _path_edges(path):
            if not G_truth.has_edge(u, v):
                return False, None
            edge = G_truth[u][v]
            latency_s += float(edge.get("delay_s", edge.get("weight", 0.0)))

        return True, float(latency_s)

    def run(self, *, t_start: int = 0, t_end: int | None = None) -> list[StepRecord]:
        if t_end is None:
            t_end = int(self.builder.cfg.sim.num_steps)
        if t_end < t_start:
            raise ValueError("t_end must be >= t_start")

        records: list[StepRecord] = []
        prev_access: dict[str, tuple[str | None, str | None]] = {}

        for t_idx in range(int(t_start), int(t_end)):
            paths = self.controller.step(t_idx)
            G_truth = self.builder.graph_at(t_idx, mode="sample")

            for flow in self.flows:
                path = list(paths.get(flow.name, []))
                valid, latency_s = self._validate_and_latency(path, flow, G_truth)

                if valid:
                    src_access, dst_access = _extract_access(path, flow)
                else:
                    src_access, dst_access = (None, None)

                prev_src, prev_dst = prev_access.get(flow.name, (None, None))
                handover_src = bool(prev_src is not None and src_access is not None and prev_src != src_access)
                handover_dst = bool(prev_dst is not None and dst_access is not None and prev_dst != dst_access)

                if valid:
                    prev_access[flow.name] = (src_access, dst_access)

                records.append(
                    StepRecord(
                        t_idx=t_idx,
                        flow_name=flow.name,
                        outage=(not valid),
                        latency_s=latency_s,
                        path=path,
                        access_sat_src=src_access,
                        access_sat_dst=dst_access,
                        handover_at_src=handover_src,
                        handover_at_dst=handover_dst,
                    )
                )

        return records
