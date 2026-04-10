# Graph Constructor + Link Model (LEO–Ground Snapshot Graphs)

This folder is a focused, stand-alone component that builds **time-varying snapshot graphs** $G(t)=(V,E(t))$ for a LEO–ground network.

It is intended to be imported by downstream modules (routing, handover, prediction) that need:
- a deterministic **mean** graph for forecasts/inputs
- a seeded stochastic **sample** graph for realized simulation outcomes

## What’s included
- `sim/graph_builder.py`: `GraphBuilder` that emits `nx.Graph` snapshots
- `sim/link_model.py`: delay + reliability penalty (optional fading / failures)
- `sim/geometry.py`, `sim/orbits.py`: MVP geometry + circular Walker-like constellation
- `tests/`: unit tests for geometry/link model/graph builder
- `scripts/`: sanity plot + single-snapshot visualization

## Quickstart (Python)
1. (Recommended) Create a virtual environment:
    - `python -m venv .venv && source .venv/bin/activate`
2. Install deps:
    - `pip install -r requirements.txt`
3. Run tests:
    - `python -m pytest -q`

## Core API
- `GraphBuilder.graph_at(t_idx, *, mode="sample"|"mean") -> nx.Graph`
- `GraphBuilder.forecast_at(t_idx, H, *, mode="mean"|"sample", pred_error=None) -> list[nx.Graph]`
- `GraphBuilder.truth_tables_at(t_idx, *, mode=...) -> (nodes_df, edges_df)` (debugging)

Semantics:
- `mode="mean"`: deterministic (no fading/failures). Use for prediction inputs.
- `mode="sample"`: realized network with seeded randomness. Use for simulation outcomes.

## Minimal example
```python
from sim import (
    ConstellationConfig,
    GraphBuilder,
    GraphBuilderConfig,
    GroundStation,
    LinkConstraints,
    LinkModelConfig,
    SimConfig,
)

cfg = GraphBuilderConfig(
    sim=SimConfig(dt_s=1.0, t_end_s=60.0, seed=123),
    constellation=ConstellationConfig(
        num_planes=2,
        sats_per_plane=6,
        altitude_m=550_000.0,
        inclination_deg=53.0,
    ),
    ground_stations=[GroundStation("A", 0.0, 0.0, 0.0)],
    links=LinkConstraints(theta_min_deg=10.0, isl_range_max_m=6_000_000.0, isl_mode="neighbor"),
    link_model=LinkModelConfig(sigma_db=3.0),
)

b = GraphBuilder(cfg)
G = b.graph_at(10, mode="sample")
print(G.number_of_nodes(), G.number_of_edges())
```

## Docs
- `docs/GRAPH_BUILDER_CONTRACT.md`: node/edge attributes + units
- `docs/CONFIG_REFERENCE.md`: parameter meanings + recommended ranges
- `docs/SANITY_CHECKS.md`: invariants to validate after model changes
- `docs/IMPLEMENTATION_PLAN.md`: design notes / extension plan
- `docs/PROJECT_CONTEXT.md`: broader project context (optional)

## Scripts
- Edge-count sanity plot: `python scripts/sanity_snapshot_counts.py`
- Snapshot visualizer: `python scripts/visualize_snapshot.py --t-idx 10 --mode sample`

## Reproducibility note
All stochastic components (fading, random failures) are driven by an explicit seed in config and use deterministic iteration order so that:

- same seed + same config ⇒ identical results
