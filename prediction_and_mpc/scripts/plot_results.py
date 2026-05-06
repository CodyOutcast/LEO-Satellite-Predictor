from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--input",
        type=Path,
        default=PKG_ROOT / "outputs" / "eval_records.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PKG_ROOT / "outputs",
    )
    args = parser.parse_args()

    df = _load_jsonl(args.input)
    for col in [
        "outage_prob",
        "p95_latency_s",
        "handover_rate_per_min",
        "regret_rate",
        "error_rate",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(["controller", "predictor", "error_rate"], as_index=False)[
            ["outage_prob", "p95_latency_s", "handover_rate_per_min", "regret_rate"]
        ]
        .mean(numeric_only=True)
    )

    baseline = grouped[grouped["error_rate"] == grouped["error_rate"].min()].copy()
    baseline["label"] = baseline["controller"] + " | " + baseline["predictor"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.reshape(-1)

    metrics = [
        ("outage_prob", "Outage Probability"),
        ("p95_latency_s", "P95 Latency (s)"),
        ("handover_rate_per_min", "Handover Rate (/min)"),
        ("regret_rate", "Handover Regret Rate"),
    ]

    for ax, (col, title) in zip(axes, metrics, strict=False):
        ax.bar(baseline["label"], baseline[col])
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    overview_path = args.output_dir / "eval_overview.png"
    fig.savefig(overview_path, dpi=180)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for (controller, predictor), sub in grouped.groupby(["controller", "predictor"]):
        sub_sorted = sub.sort_values("error_rate")
        ax2.plot(
            sub_sorted["error_rate"],
            sub_sorted["outage_prob"],
            marker="o",
            label=f"{controller} | {predictor}",
        )

    ax2.set_title("Robustness Sweep: Outage vs Forecast Error")
    ax2.set_xlabel("Forecast Error Rate")
    ax2.set_ylabel("Outage Probability")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()

    sweep_path = args.output_dir / "robustness_sweep.png"
    fig2.savefig(sweep_path, dpi=180)

    print(f"wrote {overview_path}")
    print(f"wrote {sweep_path}")


if __name__ == "__main__":
    main()
