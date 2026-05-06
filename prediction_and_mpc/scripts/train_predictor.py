from __future__ import annotations

import argparse
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

from pred_mpc.train import generate_telemetry, train_learned_sysid


def make_builder(seed: int, t_end_s: float) -> GraphBuilder:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate telemetry and train LearnedSysID predictor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-start", type=int, default=0)
    parser.add_argument("--t-end", type=int, default=800)
    parser.add_argument("--model-kind", choices=["mlp", "linear"], default="mlp")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PKG_ROOT / "checkpoints" / "learned_sysid.joblib",
    )
    args = parser.parse_args()

    builder = make_builder(seed=args.seed, t_end_s=max(2.0, float(args.t_end) * 2.0 + 2.0))
    telemetry = generate_telemetry(builder, args.t_start, args.t_end)

    _, metrics = train_learned_sysid(
        telemetry,
        save_path=args.checkpoint,
        model_kind=args.model_kind,
        random_state=args.seed,
    )

    print(f"wrote model: {args.checkpoint}")
    print(f"telemetry_rows={len(telemetry)}")
    print(f"mae_train_db={metrics['mae_train_db']:.4f}")
    print(f"mae_test_db={metrics['mae_test_db']:.4f}")
    print(f"r2_test={metrics['r2_test']:.4f}")


if __name__ == "__main__":
    main()
