from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = REPO_ROOT / "prediction_and_mpc"
SIM_ROOT = REPO_ROOT / "graph_constructor_and_link_model"

for p in [PRED_ROOT, SIM_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
