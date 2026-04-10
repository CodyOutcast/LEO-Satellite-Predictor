# GraphBuilder Contract (Nodes, Edges, Units)

This document defines the **shared contract** between:
- graph construction + link model,
- routing baselines,
- handover baselines,
- prediction-aware controller,
- metric logging.

If everyone adheres to this contract, you avoid subtle integration bugs.

## Time indexing
- The simulator advances in discrete timesteps `t_idx = 0,1,2,...`.
- Physical time in seconds is `t_s = t_idx * dt_s`.

## Node IDs
Node IDs must be stable across all timesteps.

Recommended:
- Satellites: `SAT-P{plane}-S{slot}`
- Ground stations: `GS-{name}`

## Node attributes
Every node must include:
- `kind`: `"sat"` or `"gs"`
- `pos_ecef_m`: 3-vector in meters (ECEF), e.g. numpy array or tuple `(x,y,z)`

Optional but useful:
- `plane`, `slot` for satellites
- `lat_deg`, `lon_deg`, `alt_m` for ground stations

## Edge types
Edges can be undirected (`nx.Graph`) unless you intentionally model asymmetry.

Edge attribute `kind`:
- `"access"` for GS–SAT links
- `"isl"` for SAT–SAT links

## Edge attributes (minimum)
Every edge must include:
- `kind`: `"access"|"isl"`
- `range_m`: float (meters)
- `delay_s`: float (seconds), `delay_s = range_m / c`
- `weight`: float (seconds), the value used by routing (`nx.shortest_path(..., weight="weight")`)

## Edge attributes (recommended for evaluation/debugging)
These should be present when the link model includes reliability:
- `snr_db`: float (dB), deterministic in `mode="mean"`, realized in `mode="sample"`
- `p_succ`: float in (0,1]
- `penalty`: float >= 0, typically `-log(max(p_succ, eps))`
- `weight_delay_s`: float (usually same as `delay_s`)
- `weight_rel_s`: float (the reliability contribution in seconds)

## Weight semantics
Recommended:
- `weight = delay_s + w_rel_s * penalty`

Interpretation:
- `delay_s` captures propagation delay.
- `penalty` captures reliability preference (lower is better).

If `penalty = -log(p_succ)`, then summing penalties along a path corresponds to maximizing the product of per-edge success probabilities (a principled reliability objective).

## Mean vs sample graphs
- `mode="mean"`:
  - No stochastic fading or random failures.
  - Use deterministic `snr_db` (or deterministic quality proxy) to compute `p_succ`, `penalty`, and `weight`.
- `mode="sample"`:
  - Apply seeded fading noise and/or random failures.
  - If you support binary link-down: omit edges that are down.

Forecast graphs returned by `forecast_at(...)` should default to `mode="mean"` unless explicitly requested.

If `forecast_at(...)` supports `mode="sample"`, treat it as a debugging/Monte Carlo feature rather than the default prediction input.
