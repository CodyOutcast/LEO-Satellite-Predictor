from __future__ import annotations

from pathlib import Path
import sys


def ensure_sim_import_path() -> Path:
    """Add the simulator package directory to sys.path if needed."""

    sim_root = Path(__file__).resolve().parents[2] / "graph_constructor_and_link_model"
    sim_root_str = str(sim_root)
    if sim_root_str not in sys.path:
        sys.path.insert(0, sim_root_str)
    return sim_root
