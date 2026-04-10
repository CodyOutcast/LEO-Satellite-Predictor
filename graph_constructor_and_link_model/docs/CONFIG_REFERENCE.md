# Config Reference (Geometry + Link Model)

This is the shared vocabulary for simulator parameters.

## Simulation
- `dt_s`: timestep size in seconds.
- `t_end_s`: simulation horizon in seconds.
- `seed`: RNG seed for any stochastic components.

## Earth/physics constants
- `R_earth_m`: Earth radius (spherical Earth for MVP).
- `mu_m3_s2`: Earth gravitational parameter.
- `omega_earth_rad_s`: Earth rotation rate.
- `c_m_s`: speed of light.

## Constellation (MVP circular orbits)
- `num_planes`: number of orbital planes.
- `sats_per_plane`: satellites per plane.
- `altitude_m`: satellite altitude above Earth surface.
- `inclination_deg`: orbital inclination.
- `raan_spacing`: how RAAN is distributed across planes.
- `phase_offset`: relative phase between planes (controls Walker-like phasing).

## Ground stations
Each ground station:
- `name`
- `lat_deg`, `lon_deg`
- `alt_m`

## Link constraints
- `theta_min_deg`: minimum elevation angle for GS–SAT access links.
- `gs_range_max_m`: optional hard max range for access links.
- `isl_range_max_m`: max range for ISLs.
- `earth_occlusion`: whether to enforce Earth blockage for ISLs.
- `occlusion_margin_m`: optional margin added to `R_earth_m` to avoid grazing links.

## ISL candidate topology
- `isl_mode`: `"neighbor"` or `"all_within_range"`
  - `neighbor`: generate a sparse candidate set (recommended for scale).
  - `all_within_range`: brute force for small constellations.

## Link model / reliability
- `snr_ref_db`: SNR at reference distance.
- `d_ref_m`: reference distance.
- `sigma_db`: standard deviation of fading noise in dB (0 disables fading).
- `snr_threshold_db` (`gamma`): center threshold for logistic mapping to `p_succ`.
- `snr_softness_db` (`k`): slope/softness of logistic.
- `w_rel_s`: how strongly reliability influences `weight` (seconds per penalty unit).

## Optional binary down model
- `snr_down_threshold_db`: if realized `snr_db` falls below this, drop the edge in `mode="sample"`.

## Optional stress/failure hooks
- `p_edge_fail`: independent probability of dropping any eligible edge.
- `node_failure_schedule`: list of `(t_idx_start, t_idx_end, node_id)` to remove.

## Recommended starting ranges (not strict)
- `dt_s`: 1–5
- `theta_min_deg`: 10–20
- `sigma_db`: 2–6 (mild)
- `w_rel_s`: start around 0.005–0.05 (5–50 ms) and tune against typical propagation delays.
