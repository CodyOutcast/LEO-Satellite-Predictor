from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pred_mpc._bootstrap import ensure_sim_import_path

ensure_sim_import_path()

from sim import (  # type: ignore
    ConstellationConfig,
    GraphBuilder,
    GraphBuilderConfig,
    GroundStation,
    LinkConstraints,
    LinkModelConfig,
    SimConfig,
)

from pred_mpc.controllers import RollingReplanMPC, TimeExpandedMPC
from pred_mpc.flows import DEFAULT_FLOWS
from pred_mpc.metrics import handover_stats, latency_cdf, outage_probability
from pred_mpc.predictors import GeometricMeanPredictor, LearnedSysIDPredictor, make_pred_error
from pred_mpc.runner import SimulationRunner
from pred_mpc.train import generate_telemetry, load_learned_sysid, train_learned_sysid


def make_builder(seed: int, *, t_end_s: float) -> GraphBuilder:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=2.0, t_end_s=t_end_s, seed=seed),
        constellation=ConstellationConfig(
            num_planes=3,
            sats_per_plane=8,
            altitude_m=550_000.0,
            inclination_deg=53.0,
            phase_offset_deg=10.0,
        ),
        ground_stations=[
            GroundStation("SF", 37.7749, -122.4194, 0.0),
            GroundStation("SIN", 1.3521, 103.8198, 0.0),
            GroundStation("LON", 51.5072, -0.1276, 0.0),
        ],
        links=LinkConstraints(
            theta_min_deg=10.0,
            isl_range_max_m=6_000_000.0,
            earth_occlusion=True,
            occlusion_margin_m=1.0,
            isl_mode="neighbor",
        ),
        link_model=LinkModelConfig(
            snr_ref_db=22.0,
            d_ref_m=1_000_000.0,
            sigma_db=3.0,
            snr_threshold_db=10.0,
            snr_softness_db=2.0,
            w_rel_s=0.01,
            snr_down_threshold_db=None,
            p_edge_fail=0.0,
        ),
    )
    return GraphBuilder(cfg)


def _ensure_model(path: Path, *, seed: int) -> Path:
    if path.exists():
        return path

    trainer = make_builder(seed, t_end_s=2_000.0)
    telemetry = generate_telemetry(trainer, 0, 800)
    _, metrics = train_learned_sysid(telemetry, save_path=path, model_kind="mlp", random_state=seed)
    print(f"trained fallback model at {path} (mae_test_db={metrics['mae_test_db']:.4f})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controller/predictor evaluation sweeps")
    parser.add_argument("--quick", action="store_true", help="Run a short single-seed sweep")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lambda-handover", type=float, default=0.05)
    parser.add_argument("--error-kind", choices=["snr_jitter", "edge_flip"], default="snr_jitter")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PKG_ROOT / "checkpoints" / "learned_sysid.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PKG_ROOT / "outputs" / "eval_records.jsonl",
    )
    args = parser.parse_args()

    if args.quick:
        seeds = [0]
        steps = 30
        error_rates = [0.0, 0.5, 1.0]
    else:
        seeds = [0, 1, 2, 3, 4]
        steps = 180
        error_rates = [0.0, 0.1, 0.3, 0.5, 1.0]

    model_path = _ensure_model(args.model_path, seed=seeds[0])
    learned_model = load_learned_sysid(model_path)

    rows: list[dict] = []

    for seed in seeds:
        builder = make_builder(seed, t_end_s=float(steps * 2 + 2))
        flows = list(DEFAULT_FLOWS)

        for error_rate in error_rates:
            pred_error = make_pred_error(
                error_rate,
                kind=args.error_kind,
                link_model_cfg=builder.cfg.link_model,
            )

            predictors = {
                "geometric_mean": GeometricMeanPredictor(builder, pred_error=pred_error),
                "learned_sysid": LearnedSysIDPredictor(
                    builder,
                    learned_model,
                    pred_error=pred_error,
                ),
            }

            for predictor_name, predictor in predictors.items():
                controllers = {
                    "rolling_replan": RollingReplanMPC(
                        builder,
                        predictor,
                        flows,
                        H=args.horizon,
                        lambda_handover=args.lambda_handover,
                    ),
                    "time_expanded": TimeExpandedMPC(
                        builder,
                        predictor,
                        flows,
                        H=args.horizon,
                        lambda_handover=args.lambda_handover,
                    ),
                }

                for controller_name, controller in controllers.items():
                    runner = SimulationRunner(builder, controller, flows)
                    records = runner.run(t_start=0, t_end=steps)

                    outage = outage_probability(records, by_flow=True)
                    latency = latency_cdf(records)
                    handover = handover_stats(records, dwell_window_tau=5, dt_s=builder.cfg.sim.dt_s)

                    for flow in flows:
                        flow_latency = latency.get(flow.name, {})
                        flow_handover = handover.get(flow.name, {})

                        rows.append(
                            {
                                "seed": int(seed),
                                "flow": flow.name,
                                "controller": controller_name,
                                "predictor": predictor_name,
                                "error_rate": float(error_rate),
                                "horizon": int(args.horizon),
                                "lambda_handover": float(args.lambda_handover),
                                "outage_prob": float(outage.get(flow.name, float("nan"))),
                                "p50_latency_s": float(flow_latency.get(50, float("nan"))),
                                "p95_latency_s": float(flow_latency.get(95, float("nan"))),
                                "p99_latency_s": float(flow_latency.get(99, float("nan"))),
                                "handover_rate_per_min": float(flow_handover.get("handover_rate_per_min", 0.0)),
                                "regret_rate": float(flow_handover.get("regret_rate", 0.0)),
                                "num_records": int(len(records)),
                            }
                        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
