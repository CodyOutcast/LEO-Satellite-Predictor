# Graph Constructor + Link Model — Implementation Plan

This document records the intended design and verification plan for the **snapshot graph generator** (`GraphBuilder`) and the **per-link model** (`link_model`).

Originally drafted by Zhuokai Chen; kept here as onboarding/design notes.

## 0) What we’re building (scope)
This component is responsible for the **time-varying snapshot graph generator** and the **per-link model** used by routing/handover policies.

At each timestep $t_i$ (sampled every $\Delta t$ seconds), generate a NetworkX graph:

- $G(t_i)=(V, E(t_i))$.
- **Nodes:** LEO satellites + ground stations.
- **Edges:** appear/disappear based on **geometry constraints** (and optionally stochastic failures).
- **Edge weights:** combine **propagation delay** and a **reliability penalty** (SNR proxy), with optional stochastic fading/interference.

This must be cleanly callable by:
- reactive shortest-path baseline (Dijkstra on current snapshot),
- greedy handover baseline (pick “best” visible access link),
- prediction-aware controller (needs **forecast** of availability/costs over horizon $H$).

## 1) Design goals (so the rest of the team can move fast)
1. **Correctness first (unit-testable math):** geometry and link equations in pure functions.
2. **Stable, minimal interface:** one class/function to ask for `G(t)` and optional forecast.
3. **Reproducible stochasticity:** fixed seed → same sampled fades/failures.
4. **Scales reasonably:** avoid $O(N^2)$ ISL enumeration by default.
5. **Extensible:** allow swapping “simple orbit” with SGP4 later without rewriting graph logic.

Non-goals (for MVP): high-fidelity link budgets, atmospheric loss models, Doppler, scheduling, MAC/queueing.

## 2) Proposed module/API contract (minimal but future-proof)
### 2.1 Recommended file structure (suggestion)
- `sim/config.py` — dataclasses for parameters
- `sim/geometry.py` — coordinate transforms + visibility
- `sim/orbits.py` — satellite position provider(s)
- `sim/link_model.py` — delay/SNR proxy/reliability/cost
- `sim/graph_builder.py` — builds NetworkX snapshots

(You can start with a single file and split later; the key is keeping “pure math” separable.)

### 2.2 Core public interface (what others call)
Create a class that exposes **truth** graphs and optionally **forecast** graphs:

- `GraphBuilder.graph_at(t_idx, *, mode="sample"|"mean") -> nx.Graph`
- `GraphBuilder.truth_tables_at(t_idx) -> (nodes_df, edges_df)` (optional but very useful for debugging)
- `GraphBuilder.forecast_at(t_idx, H, *, mode="mean"|"sample", pred_error=None) -> list[nx.Graph]`
- `GraphBuilder.time_s(t_idx) -> float` (small helper to standardize $t=t_{\mathrm{idx}}\Delta t$)

Notes:
- `mode="mean"` means deterministic costs (no random fading/failures); use this for prediction inputs.
- `mode="sample"` means apply fading/failures (the realized network).
- `forecast_at(...)` should default to `mode="mean"`; `mode="sample"` is mainly for debugging/Monte Carlo.
- `pred_error` is a hook for the prediction team to inject forecast error later.

### 2.3 Node identifiers (avoid handover bookkeeping pain)
Use stable IDs that never change:
- Satellites: `SAT-P{plane}-S{slot}` or integer indices.
- Ground stations: `GS-{name}`.

Store node attributes:
- `kind: "sat"|"gs"`
- `pos_ecef_m: np.ndarray(3,)` (or separate x/y/z)

## 3) Phase-by-phase implementation (optimal sequencing)
### Phase A — Deterministic geometry + delay-only weights (MVP)
**Deliverable:** `graph_at(t_idx)` produces plausible GS–SAT visibility edges and ISL edges, weighted by propagation delay only.

1. Implement config dataclasses
   - `SimConfig(dt_s, t_end_s, seed)`
   - `ConstellationConfig(num_planes, sats_per_plane, altitude_m, inclination_deg, phase_offset)`
   - `GroundStation(name, lat_deg, lon_deg, alt_m)`
   - `LinkConstraints(theta_min_deg, gs_range_max_m, isl_range_max_m, earth_occlusion=True)`

2. Implement coordinate and visibility utilities (pure functions)
   - `latlon_to_ecef(lat, lon, alt, R_earth)`
   - `range_m(p, q) = ||p-q||`

   Ground-to-sat elevation angle (ground $\mathbf{r}_g$, satellite $\mathbf{r}_s$):
   - Line-of-sight vector: $\boldsymbol{\rho}=\mathbf{r}_s-\mathbf{r}_g$
   - Local zenith unit vector: $\hat{\mathbf{z}}=\mathbf{r}_g/||\mathbf{r}_g||$
   - Elevation (clip for numerical stability):

     $$\mathrm{elev}=\arcsin\left(\mathrm{clip}\left(\frac{\boldsymbol{\rho}\cdot\hat{\mathbf{z}}}{||\boldsymbol{\rho}||},-1,1\right)\right)$$

   ISL Earth-occlusion (line-of-sight) check using distance-to-segment:
   - Satellites at $\mathbf{r}_1,\mathbf{r}_2$, define $\mathbf{d}=\mathbf{r}_2-\mathbf{r}_1$.
   - Closest point on the segment to Earth center:

     $$t^* = \mathrm{clip}\left(-\frac{\mathbf{r}_1\cdot\mathbf{d}}{||\mathbf{d}||^2}, 0, 1\right),\quad \mathbf{p}=\mathbf{r}_1+t^*\mathbf{d}$$

   - LOS is clear iff $||\mathbf{p}||>R_\oplus$ (optionally use $R_\oplus+\delta$ to avoid grazing links due to floating-point noise).

3. Implement a **simple orbit position provider** (fast, good enough for MVP)
  - Use circular Keplerian orbits in ECI and rotate to ECEF so ground stations can remain fixed in ECEF.
  - Keep it simple, but document assumptions and keep a clean interface to swap in SGP4 later.

   Suggested constants:
   - $\mu=3.986004418\times 10^{14}\,\mathrm{m^3/s^2}$
   - $\omega_\oplus=7.2921159\times 10^{-5}\,\mathrm{rad/s}$
   - $R_\oplus\approx 6371\,\mathrm{km}$

   Circular orbit mean motion:
   $$n=\sqrt{\mu/a^3},\quad a=R_\oplus + h$$

    Implementation outline (still MVP-friendly):
    - Compute $\mathbf{r}_{\mathrm{ECI}}(t)$ from circular motion in the orbital plane.
    - Apply rotations for inclination and RAAN.
    - Convert to ECEF: $\mathbf{r}_{\mathrm{ECEF}}(t)=R_3(-\omega_\oplus t)\,\mathbf{r}_{\mathrm{ECI}}(t)$.
    - Deterministically assign each satellite a (plane, slot) → (RAAN, initial phase) mapping (Walker-like spacing is fine).

    Fallback (acceptable if time is tight): keep everything in one frame (treat Earth as non-rotating), but state the approximation and keep the provider swappable for SGP4 later.

4. Build the snapshot graph `G(t)`
   - Add all nodes (sats + GS).
   - Add **GS–SAT edges** where:
     - `elev_deg >= theta_min_deg` and (optional) `range <= gs_range_max_m`.
   - Add **ISL edges** where:
     - `range <= isl_range_max_m` and (optional) Earth not blocking.

   **Important performance choice:** do *not* connect all sat pairs by default.
   - Precompute candidate ISL pairs using a topology (e.g., intra-plane +/-1 neighbor + inter-plane neighbor). Then apply range/LOS constraints.
   - Keep a config switch `isl_mode = "neighbor" | "all_within_range"`.

5. Edge attributes (even in MVP)

  Always include:
  - `range_m`
  - `delay_s = range_m / c` where $c=299\,792\,458\,\mathrm{m/s}$
  - `kind: "access"|"isl"`
  - `weight` (for NetworkX shortest path) — start with `delay_s`

6. MVP sanity checks
   - As `theta_min_deg` increases, number of access edges decreases monotonically.
   - As `isl_range_max_m` increases, number of ISL edges increases.
   - A satellite directly above a GS yields elev ~ 90°.

### Phase B — Link-quality proxy + stochastic fading (core of your “link model”)
**Deliverable:** edge weights combine delay + reliability penalty; optional stochastic outages and/or noisy costs.

1. Choose a simple SNR proxy that is monotone with range
   Two good options:

   **Option 1 (dB-like SNR proxy):**
   $$\mathrm{SNR}_{\mathrm{dB}}(d)=\mathrm{SNR}_{\mathrm{ref}} - 20\log_{10}(d/d_{\mathrm{ref}})$$

   **Option 2 (quality in [0,1]):**
   $$q(d)=\exp(-d/d_0)$$

   Recommendation: Option 1 is easier to interpret and tune.

2. Add stochasticity (fading/interference)
   - Sample per-edge, per-timestep additive noise in dB:

     $$\tilde{\mathrm{SNR}}_{\mathrm{dB}} = \mathrm{SNR}_{\mathrm{dB}} + \epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma_{\mathrm{dB}}^2)$$

    - Use `numpy.random.Generator(seed)` and make sure the sampling order is deterministic (e.g., iterate edges in sorted node-id order).

3. Map SNR proxy to reliability penalty (must stay nonnegative)
   You want a cost term that encourages routes with “healthier” links.

   Suggested approach:
  - Define a soft success probability (logistic), where $\sigma(x)=\frac{1}{1+e^{-x}}$:

     $$p_{\mathrm{succ}}=\sigma\left(\frac{\tilde{\mathrm{SNR}}_{\mathrm{dB}}-\gamma}{k}\right)$$

     where $\gamma$ is an SNR threshold and $k$ is a softness parameter.
   - Reliability penalty:

     $$\mathrm{penalty} = -\log(\max(p_{\mathrm{succ}},\varepsilon))$$

   This is always $\ge 0$ and integrates nicely into shortest-path costs.

    Why this is “optimal” for routing: if you treat path success as the product of per-link success probabilities, then summing $-\log(p_{\mathrm{succ}})$ over edges corresponds to maximizing end-to-end success.

4. Define final edge weight (units: seconds)
  $$\mathrm{weight} = \mathrm{delay}_s + w_{\mathrm{rel},s}\cdot \mathrm{penalty}$$

  Tune `w_rel_s` so that reliability meaningfully competes with delay (start with something like 1–50 ms equivalents depending on your graph sizes).

5. Optional: stochastic *availability* (binary up/down)
   To let “outage probability” reflect more than geometry, you can (optionally) declare a link down if realized SNR is below a hard threshold:
   - If `snr_db < snr_down_threshold_db`: **omit the edge** in the realized graph.

   Keep this optional via config; you’ll still have geometry-only connectivity for simpler experiments.

6. Provide attributes needed by other baselines
   - Greedy handover baseline wants “best visible satellite”. Provide:
     - `snr_db` (or `quality`) on access edges.
   - Routing wants:
     - `weight`, `delay_s`.

   Strong suggestion: also expose `penalty` and the realized `p_succ` (or `snr_db`) as edge attributes so the analysis team can decompose “why a path was chosen.”

### Phase C — Failure/stressor hooks (needed for evaluation scenarios)
**Deliverable:** ability to inject random link/satellite failures without rewriting geometry.

Add optional modifiers applied *after* geometry eligibility:
- Per-edge independent failure: drop edge with probability `p_edge_fail`.
- Per-node failure schedule: a list of `(t_idx_start, t_idx_end, node_id)` to remove.
- Burst failures: Markov on/off per edge (optional; only if time permits).

Keep these in a separate function so the “clean geometry graph” is always available.

### Phase D — Forecast support (so prediction-aware routing is easy)
**Deliverable:** easy access to horizon-$H$ graphs with clear “truth vs prediction” separation.

Implement:
- `forecast_at(t_idx, H, *, mode="mean"|"sample", pred_error=None)`
  - returns `G_hat(t_idx + k)` for `k=0..H-1`.
  - `mode="mean"` should use deterministic SNR (no fading noise) unless explicitly asked.

Do **not** hardcode the forecast-error injection into your builder if the team hasn’t decided on a model yet. Provide a hook:
- `pred_error.apply(edges_df)` or a callable that perturbs `snr_db` or flips availability.

## 4) Performance plan (so it runs fast enough)
1. **Vectorize GS–SAT visibility:** for each GS compute elevation/range to all satellites using NumPy arrays.
2. **Avoid full sat–sat all-pairs:** default ISL candidate set based on neighbor topology; optionally enable all-within-range for small N.
3. **Batch-add edges to NetworkX:** build a list of `(u, v, attrs)` then `G.add_edges_from(...)` once.
4. **Precompute positions for all timesteps (optional):** store `pos[t_idx, sat_idx, 3]` if memory allows. This makes `forecast_at` cheap.

## 5) Verification checklist (what to unit test + sanity plots)
### 5.1 Unit tests (pure functions)
- `latlon_to_ecef` matches known points (equator, poles).
- Elevation:
  - satellite directly above GS → elev ≈ 90°
  - satellite at horizon geometry → elev ≈ 0°
- Earth occlusion:
  - two satellites opposite sides of Earth → blocked
  - nearby satellites with clear LOS → not blocked
- Monotonicity:
  - increasing range → decreasing SNR proxy
  - increasing range → increasing delay and (usually) increasing weight

### 5.2 Integration sanity checks
- Edge counts over time: plot #access edges and #ISL edges vs timestep.
- Connectivity: for a fixed GS pair, plot whether a path exists over time.
- Distribution checks: with fading enabled, plot SNR histogram and ensure variance matches $\sigma_{\mathrm{dB}}$.

### 5.3 Invariants aligned with the report
- As forecast error → 100% (prediction module), performance should trend toward reactive baseline.
- With fading disabled and deterministic orbits, repeated runs match exactly.

## 6) Parameter defaults (good starting points)
These are “reasonable” placeholders; tune later.
- `dt_s`: 1–5 s (start with 1–2 s for handover granularity)
- `theta_min_deg`: 10°–20°
- `isl_range_max_m`: pick based on constellation size; start large enough that neighbor ISLs usually exist
- `snr_ref_db`: pick so that typical access ranges are above threshold
- `sigma_db`: 2–6 dB for mild fading
- `snr_down_threshold_db`: optional, start near the midpoint of your logistic

## 7) Concrete deliverables you can hand to the team
By the end of your work, you should be able to point the team to:
1. A single class/function that produces `G(t)` snapshots with edge attributes:
   - `weight`, `delay_s`, `snr_db` (or `quality`), `kind`.
2. A deterministic `forecast_at(t, H)` that returns predicted graphs (mean model).
3. A small set of tests/sanity scripts that validate geometry and monotonicity.

## 8) Repo/docs files to add (so teammates can use it correctly)
These are lightweight, but they prevent integration bugs and “what does this attribute mean?” confusion.

- `README.md`
  - Project overview, quickstart, and a minimal “how to get `G(t)`” example.
  - Clearly defines the difference between `mode="mean"` and `mode="sample"`.
- `requirements.txt`
  - Minimum Python deps: `numpy`, `networkx`, `pandas` (if using truth tables), `matplotlib`, `pytest`.
- `.gitignore`
  - Python and local environment artifacts.
- `docs/GRAPH_BUILDER_CONTRACT.md`
  - The **API contract**: node IDs, node attributes, edge attributes, units, invariants.
  - Exact semantics of `weight` (what routing should optimize).
- `docs/CONFIG_REFERENCE.md`
  - Parameter meanings and recommended defaults/ranges (e.g., $\theta_{\min}$, ISL mode).
- `docs/SANITY_CHECKS.md`
  - The plots/tests to run after changing geometry or link model.

## 9) Suggested work breakdown (fast path)
1. Geometry utilities + GS coordinate conversion.
2. Simple orbit provider + snapshot builder (delay-only).
3. ISL LOS/occlusion + neighbor topology.
4. SNR proxy + reliability penalty + seeded fading.
5. Optional binary link-down + stress/failure hooks.
6. Forecast helper + docs + sanity scripts.

### One “golden rule” to keep the whole project consistent
Always keep two representations available:
- **Mean/clean graph** (deterministic, for forecasting inputs)
- **Sampled/realized graph** (stochastic, for simulation outcomes)

This makes oracle/noisy/no-forecast ablations straightforward and avoids subtle bugs where prediction accidentally sees the same randomness as truth.
