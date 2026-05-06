from __future__ import annotations

from pred_mpc.metrics import handover_stats, latency_cdf, outage_probability
from pred_mpc.runner import StepRecord


def test_metrics_match_hand_computed_values() -> None:
    records = [
        StepRecord(
            t_idx=0,
            flow_name="F",
            outage=False,
            latency_s=1.0,
            path=["GS-A", "SAT-1", "GS-B"],
            access_sat_src="SAT-1",
            access_sat_dst="SAT-1",
            handover_at_src=False,
            handover_at_dst=False,
        ),
        StepRecord(
            t_idx=1,
            flow_name="F",
            outage=False,
            latency_s=1.5,
            path=["GS-A", "SAT-1", "GS-B"],
            access_sat_src="SAT-1",
            access_sat_dst="SAT-1",
            handover_at_src=False,
            handover_at_dst=False,
        ),
        StepRecord(
            t_idx=2,
            flow_name="F",
            outage=False,
            latency_s=2.0,
            path=["GS-A", "SAT-2", "GS-B"],
            access_sat_src="SAT-2",
            access_sat_dst="SAT-1",
            handover_at_src=True,
            handover_at_dst=False,
        ),
        StepRecord(
            t_idx=3,
            flow_name="F",
            outage=False,
            latency_s=3.0,
            path=["GS-A", "SAT-1", "GS-B"],
            access_sat_src="SAT-1",
            access_sat_dst="SAT-1",
            handover_at_src=True,
            handover_at_dst=False,
        ),
        StepRecord(
            t_idx=4,
            flow_name="F",
            outage=True,
            latency_s=None,
            path=[],
            access_sat_src=None,
            access_sat_dst=None,
            handover_at_src=False,
            handover_at_dst=False,
        ),
    ]

    outage = outage_probability(records, by_flow=True)
    assert outage["F"] == 0.2

    lat = latency_cdf(records, percentiles=(50, 95))
    assert lat["F"][50] == 1.75
    assert lat["F"][95] > 2.8

    hand = handover_stats(records, dwell_window_tau=2, dt_s=1.0)
    assert int(hand["F"]["handover_count"]) == 2
    assert int(hand["F"]["regret_count"]) == 1
