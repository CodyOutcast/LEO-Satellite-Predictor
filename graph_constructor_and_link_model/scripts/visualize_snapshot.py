"""Simple visualizer for verifying the snapshot graph + link model.

What it produces
- A 3D view of Earth, ground stations, satellites, and edges in one snapshot G(t)
- Scatter plots showing that link-model attributes behave sensibly:
  - range_km vs snr_db
  - range_km vs weight_ms

Run
- `python scripts/visualize_snapshot.py`
- `python scripts/visualize_snapshot.py --t-idx 10 --mode sample`

Output
- Writes an image to `outputs/visualize_t{t_idx}_{mode}.png`

This uses a synthetic configuration (small constellation + a few ground stations)
so it can be used for demos and regression sanity checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from sim import (
    ConstellationConfig,
    GraphBuilder,
    GraphBuilderConfig,
    GroundStation,
    LinkConstraints,
    LinkModelConfig,
    SimConfig,
)


def _set_axes_equal_3d(ax) -> None:
    """Set 3D axes to equal scale so spheres look like spheres."""

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def make_synthetic_builder(seed: int) -> GraphBuilder:
    cfg = GraphBuilderConfig(
        sim=SimConfig(dt_s=2.0, t_end_s=600.0, seed=seed),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-idx", type=int, default=10)
    parser.add_argument("--mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--show", action="store_true", help="Display window (in addition to saving)")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    builder = make_synthetic_builder(args.seed)
    G = builder.graph_at(args.t_idx, mode=args.mode)

    # Collect node positions (ECEF meters)
    node_ids = list(G.nodes())
    pos = np.stack([np.asarray(G.nodes[n]["pos_ecef_m"], dtype=float).reshape(3) for n in node_ids], axis=0)

    is_sat = np.array([G.nodes[n]["kind"] == "sat" for n in node_ids], dtype=bool)
    is_gs = ~is_sat

    # Edge metrics
    ranges_km = []
    snr_db = []
    weight_ms = []
    kinds = []

    for u, v, d in G.edges(data=True):
        ranges_km.append(float(d["range_m"]) / 1e3)
        snr_db.append(float(d.get("snr_db", np.nan)))
        weight_ms.append(float(d.get("weight", np.nan)) * 1e3)
        kinds.append(str(d.get("kind")))

    ranges_km = np.asarray(ranges_km, dtype=float)
    snr_db = np.asarray(snr_db, dtype=float)
    weight_ms = np.asarray(weight_ms, dtype=float)
    kinds = np.asarray(kinds, dtype=str)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_snr = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1])

    # Draw Earth sphere (wireframe)
    R = builder.cfg.earth.R_earth_m
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.4, alpha=0.25)

    # Nodes
    ax3d.scatter(pos[is_sat, 0], pos[is_sat, 1], pos[is_sat, 2], s=18, c="#1f77b4", label="sat")
    ax3d.scatter(
        pos[is_gs, 0], pos[is_gs, 1], pos[is_gs, 2], s=40, c="#d62728", marker="^", label="gs"
    )

    # Edges (straight in ECEF)
    for u_id, v_id, d in G.edges(data=True):
        p1 = np.asarray(G.nodes[u_id]["pos_ecef_m"], dtype=float)
        p2 = np.asarray(G.nodes[v_id]["pos_ecef_m"], dtype=float)

        kind = str(d.get("kind"))
        if kind == "access":
            color = "#2ca02c"
            alpha = 0.35
        else:
            color = "#7f7f7f"
            alpha = 0.22

        ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, linewidth=0.8)

    ax3d.set_title(
        f"Snapshot G(t): t_idx={args.t_idx}, t_s={G.graph['t_s']:.1f}, mode={args.mode}\n"
        f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}"
    )
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(loc="upper left")

    # Scale/limits
    lim = R + builder.cfg.constellation.altitude_m + 200_000.0
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    _set_axes_equal_3d(ax3d)

    # Edge scatter plots
    for kind, color, label in [
        ("access", "#2ca02c", "access"),
        ("isl", "#7f7f7f", "isl"),
    ]:
        m = kinds == kind
        if np.any(m):
            ax_snr.scatter(ranges_km[m], snr_db[m], s=14, alpha=0.7, c=color, label=label)
            ax_w.scatter(ranges_km[m], weight_ms[m], s=14, alpha=0.7, c=color, label=label)

    ax_snr.set_title("Link model: range → SNR proxy")
    ax_snr.set_xlabel("range (km)")
    ax_snr.set_ylabel("snr_db")
    ax_snr.grid(True, alpha=0.3)
    ax_snr.legend()

    ax_w.set_title("Routing weight: range → weight")
    ax_w.set_xlabel("range (km)")
    ax_w.set_ylabel("weight (ms)")
    ax_w.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = out_dir / f"visualize_t{args.t_idx}_{args.mode}.png"
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
