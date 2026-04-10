"""Quick sanity script: build graphs over time and plot edge counts.

Run:
- `python scripts/sanity_snapshot_counts.py`

Outputs:
- `outputs/edge_counts.png`
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import (
    ConstellationConfig,
    GraphBuilder,
    GraphBuilderConfig,
    GroundStation,
    LinkConstraints,
    LinkModelConfig,
    SimConfig,
)


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=2.0, t_end_s=300.0, seed=123),
        constellation=ConstellationConfig(
            num_planes=3,
            sats_per_plane=8,
            altitude_m=550_000.0,
            inclination_deg=53.0,
            phase_offset_deg=10.0,
        ),
        ground_stations=[
            GroundStation("A", 0.0, 0.0, 0.0),
            GroundStation("B", 30.0, 90.0, 0.0),
        ],
        links=LinkConstraints(theta_min_deg=10.0, isl_range_max_m=6_000_000.0, isl_mode="neighbor"),
        link_model=LinkModelConfig(sigma_db=0.0, p_edge_fail=0.0, snr_down_threshold_db=None),
    )

    b = GraphBuilder(cfg)

    t_idxs = list(range(cfg.sim.num_steps))
    access_counts = []
    isl_counts = []

    for t_idx in t_idxs:
        G = b.graph_at(t_idx, mode="mean")
        access_counts.append(sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "access"))
        isl_counts.append(sum(1 for _, _, d in G.edges(data=True) if d.get("kind") == "isl"))

    plt.figure(figsize=(9, 4))
    plt.plot(t_idxs, access_counts, label="access")
    plt.plot(t_idxs, isl_counts, label="isl")
    plt.xlabel("t_idx")
    plt.ylabel("# edges")
    plt.title("Edge counts over time (mean mode)")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "edge_counts.png"
    plt.savefig(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
