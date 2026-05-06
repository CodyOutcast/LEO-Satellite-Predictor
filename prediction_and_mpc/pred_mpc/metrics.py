from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .runner import StepRecord


def _group_by_flow(records: Iterable[StepRecord]) -> dict[str, list[StepRecord]]:
    grouped: dict[str, list[StepRecord]] = defaultdict(list)
    for r in records:
        grouped[r.flow_name].append(r)
    return grouped


def outage_probability(records: list[StepRecord], by_flow: bool = True) -> dict[str, float]:
    if not records:
        return {} if by_flow else {"overall": float("nan")}

    if not by_flow:
        outage = sum(1 for r in records if r.outage)
        return {"overall": float(outage / len(records))}

    out: dict[str, float] = {}
    for flow_name, flow_records in _group_by_flow(records).items():
        outage = sum(1 for r in flow_records if r.outage)
        out[flow_name] = float(outage / len(flow_records))
    return out


def latency_cdf(
    records: list[StepRecord],
    percentiles: tuple[int, ...] = (50, 95, 99),
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for flow_name, flow_records in _group_by_flow(records).items():
        latencies = [float(r.latency_s) for r in flow_records if (not r.outage and r.latency_s is not None)]
        if not latencies:
            out[flow_name] = {p: float("nan") for p in percentiles}
            continue

        arr = np.asarray(latencies, dtype=float)
        out[flow_name] = {p: float(np.percentile(arr, p)) for p in percentiles}
    return out


def _endpoint_regret(records: list[StepRecord], *, endpoint: str, dwell_window_tau: int) -> int:
    if len(records) < 3:
        return 0

    regret = 0
    for i in range(1, len(records)):
        curr = records[i]
        if endpoint == "src":
            handover = curr.handover_at_src
            sat_now = curr.access_sat_src
            sat_prev = records[i - 1].access_sat_src
        else:
            handover = curr.handover_at_dst
            sat_now = curr.access_sat_dst
            sat_prev = records[i - 1].access_sat_dst

        if not handover or sat_now is None or sat_prev is None:
            continue

        t0 = curr.t_idx
        for j in range(i + 1, len(records)):
            future = records[j]
            if future.t_idx - t0 > dwell_window_tau:
                break

            if endpoint == "src":
                future_handover = future.handover_at_src
                future_sat = future.access_sat_src
            else:
                future_handover = future.handover_at_dst
                future_sat = future.access_sat_dst

            if future_handover and future_sat == sat_prev:
                regret += 1
                break

    return regret


def handover_stats(
    records: list[StepRecord],
    dwell_window_tau: int = 5,
    dt_s: float = 1.0,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for flow_name, flow_records in _group_by_flow(records).items():
        flow_records_sorted = sorted(flow_records, key=lambda r: r.t_idx)

        src_count = sum(1 for r in flow_records_sorted if r.handover_at_src)
        dst_count = sum(1 for r in flow_records_sorted if r.handover_at_dst)
        total_count = src_count + dst_count

        duration_steps = max(1, flow_records_sorted[-1].t_idx - flow_records_sorted[0].t_idx + 1)
        duration_min = max(1e-9, (duration_steps * float(dt_s)) / 60.0)

        src_regret = _endpoint_regret(flow_records_sorted, endpoint="src", dwell_window_tau=dwell_window_tau)
        dst_regret = _endpoint_regret(flow_records_sorted, endpoint="dst", dwell_window_tau=dwell_window_tau)
        total_regret = src_regret + dst_regret

        out[flow_name] = {
            "handover_count": float(total_count),
            "handover_count_src": float(src_count),
            "handover_count_dst": float(dst_count),
            "handover_rate_per_min": float(total_count / duration_min),
            "regret_count": float(total_regret),
            "regret_rate": float(total_regret / total_count) if total_count > 0 else 0.0,
        }

    return out
