import sys
import json
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from scripts.evaluate import make_builder
from pred_mpc.controllers import RollingReplanMPC
from pred_mpc.my_baselines import ReactiveBaselineController, GreedyHandoverController
from pred_mpc.flows import DEFAULT_FLOWS
from pred_mpc.predictors import GeometricMeanPredictor
from pred_mpc.runner import SimulationRunner
from pred_mpc.metrics import latency_cdf, outage_probability, handover_stats


def run_and_save():
    t_end_s = 600.0
    steps = int(t_end_s / 2.0)  # dt_s = 2.0

    builder = make_builder(seed=42, t_end_s=t_end_s)
    predictor = GeometricMeanPredictor(builder)

    test_suite = {
        "MPC_Rolling": RollingReplanMPC(builder, predictor, DEFAULT_FLOWS, H=5, lambda_handover=0.5),
        "Baseline_Reactive": ReactiveBaselineController(builder, predictor, DEFAULT_FLOWS),
        "Baseline_Greedy": GreedyHandoverController(builder, predictor, DEFAULT_FLOWS)
    }

    rows = []
    for name, controller in test_suite.items():
        print(f"running: {name}...")
        runner = SimulationRunner(builder, controller, DEFAULT_FLOWS)
        records = runner.run(t_start=0, t_end=steps)
        outage_dict = outage_probability(records, by_flow=True)
        latency_dict = latency_cdf(records)
        handover_dict = handover_stats(records, dwell_window_tau=5, dt_s=builder.cfg.sim.dt_s)
        flow_name = DEFAULT_FLOWS[0].name
        val_outage = outage_dict.get(flow_name, 0.0)
        val_p50 = latency_dict.get(flow_name, {}).get(50, 0.0)
        val_h_rate = handover_dict.get(flow_name, {}).get("handover_rate_per_min", 0.0)
        print(f"{name:<25} | {val_outage:>8.2%} | {val_p50:>10.4f} | {val_h_rate:>12.2f}")

        for flow in DEFAULT_FLOWS:
            f_name = flow.name
            flow_lat = latency_dict.get(f_name, {})
            flow_handover = handover_dict.get(f_name, {})

            rows.append({
                "seed": 42,
                "error_rate": 0.0,
                "predictor": "GeometricMean",
                "controller": name,
                "flow": f_name,
                "outage_prob": float(outage_dict.get(f_name, 0.0)),
                "p50_latency_s": float(flow_lat.get(50, 0.0)),
                "p95_latency_s": float(flow_lat.get(95, 0.0)),
                "handover_rate_per_min": float(flow_handover.get("handover_rate_per_min", 0.0)),
                "regret_rate": float(flow_handover.get("regret_rate", 0.0))
            })

    output_path = PKG_ROOT / "outputs" / "my_eval_records.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nResults saved to: {output_path}")
    print("use python scripts/plot_results.py --input outputs/my_eval_records.jsonl to see the visulaized result")


if __name__ == "__main__":
    run_and_save()