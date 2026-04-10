# Sanity Checks (Graph + Link Model)

Run these checks whenever you change geometry, ISL topology, or link weight definitions.

## Deterministic checks (no stochasticity)
Use `mode="mean"` and `sigma_db = 0`.

1. **Elevation monotonicity**
   - Increase `theta_min_deg` → #access edges must decrease monotonically.

2. **Range monotonicity**
   - Increase `isl_range_max_m` → #ISL edges must increase (or stay the same).

3. **Overhead satellite**
   - If you place a satellite directly above a ground station, elevation should be ~90°.

4. **ISL occlusion**
   - Two satellites on opposite sides of Earth should fail the LOS check.

## Stochastic checks
Use `mode="sample"` with fixed seed.

1. **Reproducibility**
   - Same seed + same config ⇒ identical edge attributes and edge drops.

2. **Noise variance**
   - With fading enabled, collect `snr_db - snr_db_nominal` over many edges/timesteps and verify the empirical std is close to `sigma_db`.

## Plot-based checks (quick diagnostics)
- Plot `#access edges` and `#ISL edges` vs timestep.
- Plot connectivity indicator for at least one GS–GS pair vs timestep (path exists / not).
- Plot histogram of `snr_db` and the resulting `p_succ`.

Quick visual inspection (single snapshot):
- `python scripts/visualize_snapshot.py --t-idx 10 --mode mean`
- `python scripts/visualize_snapshot.py --t-idx 10 --mode sample`

## Regression invariants aligned with the report
- If you disable prediction (reactive baseline) and then inject 100% forecast error in the prediction module, results should trend back toward reactive.
- Disabling fading should reduce variance in latency and reduce random outages.
