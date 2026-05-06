# Plan: Prediction Module + MPC Integration (Zhangcheng Kang's Part)

## Context

**Project:** ECE 3280 (Spring 2026) Frontier project — *Prediction-Aware Routing and Handover in Time-Varying LEO–Ground Networks*. Final presentation **2026-05-08**, final report **2026-05-11**. Today is **2026-05-06** — ~2 days to demo.

**Why this is being built:** The interim report committed to a *prediction-aware controller* that consumes a horizon-H forecast of edge availability/costs and minimizes
`Σ_{k=0..H-1} Ĉ(t+k) + λ·1[handover at t+k]`,
benchmarked against reactive routing. Zhangcheng owns this controller + the predictor that feeds it.

**Status of upstream work:** Zhuokai's `graph_constructor_and_link_model/` is complete (Phases A–D of its IMPLEMENTATION_PLAN done, 13 unit tests pass, `forecast_at(t_idx, H, mode, pred_error)` is the integration hook). Yiyu's baselines (reactive shortest path + greedy handover) **have not been started** — Zhangcheng will *strictly wait* and define a `Controller` protocol that Yiyu's code can plug into later.

**Scope decisions (confirmed by user):**
1. **Predictor framing — System Identification.** The simulator's fading is i.i.d. Gaussian (`link_model.py:42-45`) and geometry is deterministic Keplerian, so `mode="mean"` is Bayes-optimal. The learned predictor's job is to *recover* the SNR-vs-range curve and fading parameters from telemetry **without** being told the analytical formula. Goal is to **match** `mode="mean"`, not beat it. Story: *"MPC is robust across predictor implementations."*
2. **MPC variants.** Rolling-replan as primary, time-expanded routing (TEG) as ablation.
3. **Baselines.** Strict wait for Yiyu. The verification narrative will lean on **forecast-error injection sweeps** (10/30/50/100%) instead — at 100% error, performance approaches reactive (the §VI sanity invariant), so we *can* tell a story without Yiyu's baselines for the demo.
4. **Routing scope.** Multiple GS–GS pairs (3 pairs: SF↔LON, SF↔SIN, LON↔SIN).

## Deliverable

A new sibling Python package `prediction_and_mpc/` at the project root containing predictors, controllers, runner, metrics, training script, and tests. It depends on the simulator package via `from sim import GraphBuilder, ...`.

## Package layout

```
LEO-Satellite-Predictor/
├── graph_constructor_and_link_model/   (existing — Zhuokai)
└── prediction_and_mpc/                 (NEW — Zhangcheng)
    ├── README.md
    ├── requirements.txt              # adds scikit-learn
    ├── pred_mpc/
    │   ├── __init__.py
    │   ├── flows.py                  # Flow dataclass + flow registry
    │   ├── interfaces.py             # Predictor + Controller protocols
    │   ├── predictors.py             # 3 predictors + parametric error injection
    │   ├── controllers.py            # RollingReplanMPC + TimeExpandedMPC
    │   ├── runner.py                 # SimulationRunner + StepRecord dataclass
    │   ├── metrics.py                # outage / latency CDF / handover stats
    │   └── train.py                  # telemetry generation + LearnedSysID training
    ├── scripts/
    │   ├── train_predictor.py        # CLI: generate telemetry + train + checkpoint
    │   ├── evaluate.py               # CLI: run controllers × predictors × seeds
    │   └── plot_results.py           # produce report figures
    └── tests/
        ├── test_predictors.py
        ├── test_controllers.py
        ├── test_runner.py
        └── test_metrics.py
```

## Module specifications

### `pred_mpc/interfaces.py`
Two `typing.Protocol` definitions (no runtime dependency on a base class):

```python
class Predictor(Protocol):
    def predict(self, t_idx: int, H: int) -> list[nx.Graph]: ...

class Controller(Protocol):
    def step(self, t_idx: int) -> dict[str, list[str]]: ...
    # returns {flow_name: path_node_ids}; [] denotes outage at this t for this flow
```

This is the contract Yiyu's `Reactive` and `GreedyHandover` will implement.

### `pred_mpc/flows.py`
```python
@dataclass(frozen=True)
class Flow:
    name: str
    src: str   # ground-station node_id, e.g. "GS-SF"
    dst: str   # ground-station node_id
```
Plus a default registry of 3 GS–GS pairs (SF, LON, SIN — matching `visualize_snapshot.py`'s synthetic config).

### `pred_mpc/predictors.py`
Three implementations + one error-injection factory:

1. **`GeometricMeanPredictor(builder)`** — wraps `builder.forecast_at(t, H, mode="mean")`. Bayes-optimal baseline for the current simulator.
2. **`LearnedSysIDPredictor(builder, model)`** — calls `forecast_at(..., mode="mean")` to get geometric topology, then **overrides each edge's `snr_db`** with the learned model's prediction given features `(range_m, kind_one_hot)`. Recomputes `p_succ`, `penalty`, `weight` using the *same* logistic from `link_model.py` (reuse `link_metrics`/`edge_attributes` so semantics are identical to the simulator). Goal of training: recover the analytical SNR-vs-range curve to within ~1 dB without ever being shown the Friis formula.
3. **`OracleSamplePredictor(builder)`** — calls `forecast_at(..., mode="sample")` for ablation only; cheats by seeing realized fading. Used as upper-bound reference in plots.
4. **`make_pred_error(error_rate, kind="snr_jitter"|"edge_flip", sigma_db=...)`** — returns a `pred_error` callable suitable for `forecast_at(..., pred_error=...)`. At `error_rate=0` it is identity; at `error_rate=1.0` predictions are uninformative. This is the workhorse for the §VI robustness sweep.

**Reused from simulator:** `link_metrics`, `edge_attributes` (`sim/link_model.py`), `LinkModelConfig` (`sim/config.py`), `forecast_at` (`sim/graph_builder.py:241`).

### `pred_mpc/controllers.py`

**`RollingReplanMPC(builder, predictor, flows, H, lambda_handover, K_candidates=5)`** — primary deliverable.

At each `step(t_idx)`:
1. Get `G_now = builder.graph_at(t_idx, mode="sample")` (the realized network for this timestep — only used to bound feasible paths at `t`).
2. Get `G_hat = predictor.predict(t_idx, H)` — list of H predicted graphs.
3. For each flow:
   - Enumerate up to K candidate `(src→dst)` paths on `G_now` via `nx.shortest_simple_paths(G_now, src, dst, weight="weight")` (truncated to K).
   - For each candidate path `p`, score
     `cost(p) = Σ_{k=0..H-1} Σ_{e ∈ p ∩ E(G_hat[k])} weight_hat[k][e] + λ · 𝟙[access_sat(p) ≠ access_sat(prev_path)]`
     If an edge of `p` is missing in `G_hat[k]`, treat that timestep as outage with a fixed large penalty (configurable `outage_penalty_s`).
   - Pick argmin path; remember chosen path for next-step handover comparison.
4. Return `{flow.name: chosen_path}`.

**`TimeExpandedMPC(builder, predictor, flows, H, lambda_handover)`** — ablation.

Build a TEG over horizon H:
- Nodes: `(node_id, k)` for `k ∈ [0, H)`.
- Intra-time edges: from `G_hat[k]` with their `weight`.
- Inter-time "carry" edges: `(n, k) → (n, k+1)` with weight 0 (node persists).
- For ground stations: an edge `(gs, k) → (gs, k+1)` of weight 0 unconditionally.
- Handover penalty: applied to a *meta-edge* model — easiest implementation is to encode the access-satellite as path state. Practical shortcut for 2 days: post-hoc penalty by replacing each access edge weight with `weight + λ` if the predecessor access edge differs from the previous timestep's. Run `nx.dijkstra_path` from `(src, 0)` to `(dst, H-1)`. Project back to a per-timestep path sequence.

**Invariant test** the suite must check: when `H=1` and `λ=0`, both controllers must return identical paths to a reactive shortest-path on `G_hat[0]`.

### `pred_mpc/runner.py`

```python
@dataclass
class StepRecord:
    t_idx: int
    flow_name: str
    outage: bool
    latency_s: float | None
    path: list[str]
    access_sat_src: str | None
    access_sat_dst: str | None
    handover_at_src: bool
    handover_at_dst: bool
```

`SimulationRunner(builder, controller, flows)` runs the loop:
- For each `t_idx` in `[t_start, t_end)`:
  - `paths = controller.step(t_idx)`
  - `G_truth = builder.graph_at(t_idx, mode="sample")` (realized network)
  - For each flow: validate path against `G_truth` (every consecutive node pair must be a real edge in `G_truth`). If invalid → outage. Else sum `delay_s` along the path → `latency_s`.
  - Record handovers by comparing `access_sat` to previous timestep's record.

### `pred_mpc/metrics.py`

Functions over a list of `StepRecord`:
- `outage_probability(records, by_flow=True) -> dict[str, float]`
- `latency_cdf(records, percentiles=(50, 95, 99)) -> dict[str, dict[int, float]]`
- `handover_stats(records, dwell_window_tau: int = 5) -> dict[str, dict]` — both raw rate and the **regret** variant from the interim report (handovers within τ that were not necessary).

### `pred_mpc/train.py`

Two functions:
- `generate_telemetry(builder, t_start, t_end) -> pd.DataFrame` — drives `builder.graph_at(t, mode="sample")` over a long simulation and collects per-edge rows `(t_idx, range_m, kind, snr_db_observed)`.
- `train_learned_sysid(df, save_path, model_kind="mlp")` — uses `sklearn.neural_network.MLPRegressor` with hidden layer `(32,)`. Features: `[range_m_normalized, is_access, is_isl]`. Target: `snr_db_observed`. Saves to disk via `joblib`.

## Scripts

- `scripts/train_predictor.py` — generates ~3000 timesteps of telemetry from a fixed config (3 planes × 8 sats, 3 GS), trains, saves to `prediction_and_mpc/checkpoints/learned_sysid.joblib`. Reports MAE on a held-out test split.
- `scripts/evaluate.py` — runs the cartesian product of {RollingReplanMPC, TimeExpandedMPC} × {GeometricMeanPredictor, LearnedSysIDPredictor} × error_rates {0, 0.1, 0.3, 0.5, 1.0} × 5 seeds × 3 GS-pair flows. Writes JSONL records to `outputs/eval_records.jsonl`.
- `scripts/plot_results.py` — loads JSONL and produces:
  - Bar chart of outage probability per (controller, predictor)
  - Latency CDF overlay (50th/95th)
  - Handover rate + regret bars
  - Robustness sweep: metric vs forecast error rate (the §VI sanity plot — should asymptote to "no prediction" at 100% error)

## Dependencies

`prediction_and_mpc/requirements.txt`:
```
numpy
networkx
pandas
matplotlib
pytest
scikit-learn
joblib
```

Reuses simulator's existing deps; adds `scikit-learn` + `joblib` only.

## Critical files to reference (do not modify)

- `graph_constructor_and_link_model/sim/graph_builder.py:150` — `graph_at`
- `graph_constructor_and_link_model/sim/graph_builder.py:241` — `forecast_at` (the `pred_error` hook)
- `graph_constructor_and_link_model/sim/link_model.py:29-56` — `link_metrics`
- `graph_constructor_and_link_model/sim/link_model.py:63-85` — `edge_attributes`
- `graph_constructor_and_link_model/docs/GRAPH_BUILDER_CONTRACT.md` — node/edge attribute contract
- `graph_constructor_and_link_model/sim/__init__.py` — public API surface

## Verification

### Unit tests (`prediction_and_mpc/tests/`)
- `test_predictors.py`:
  - `GeometricMeanPredictor.predict(t, H)` returns exactly H graphs with `t_idx` advancing.
  - `LearnedSysIDPredictor` after training recovers `snr_db_nominal` to within 1.5 dB MAE on held-out edges.
  - `make_pred_error(0.0)` is identity (graph unchanged).
  - `make_pred_error(1.0, kind="edge_flip")` produces a graph with edge set independent of input.
- `test_controllers.py`:
  - On a small connected graph with `H=1, λ=0`, both `RollingReplanMPC` and `TimeExpandedMPC` return the same path as `nx.dijkstra_path(G_hat[0])`.
  - With `λ` very large, `RollingReplanMPC` keeps the same access satellite when feasible.
  - When source and destination are disconnected, controller returns `[]` (outage).
- `test_runner.py`:
  - End-to-end loop produces one `StepRecord` per (t, flow).
  - Outage flag set correctly when path uses an edge missing from `G_truth`.
- `test_metrics.py`:
  - `outage_probability` matches hand-computed value on a synthetic record list.
  - `handover_stats` regret count matches expected on a hand-crafted dwell sequence.

### End-to-end smoke (after Day 1)
```
python -m pytest prediction_and_mpc/ -q
python prediction_and_mpc/scripts/train_predictor.py
python prediction_and_mpc/scripts/evaluate.py --quick
python prediction_and_mpc/scripts/plot_results.py
```
`--quick` flag should run a 30-step single-seed config; full sweep runs nightly-style.

### Sanity invariant from interim report §VI
After the full sweep, `plot_results.py` must show: as `error_rate → 1.0`, MPC outage and 95th latency converge to the "no prediction" reference (which is just `RollingReplanMPC` with `H=1`). If they don't, the predictor or error-injection model is buggy.

## Coordination

- **For Yiyu (baselines):** share `pred_mpc/interfaces.py` (Controller protocol) + `pred_mpc/flows.py` (Flow type) + `pred_mpc/runner.py` (StepRecord type) so his Reactive/GreedyHandover slot in cleanly. Do not block on him; his code is a future drop-in.
- **For Kudzai (figures):** the JSONL schema in `outputs/eval_records.jsonl` is the contract. Plot script is starter material — Kudzai can extend.
- **For Zhuokai (simulator):** no changes requested. The `pred_error` hook on `forecast_at` is already enough.

## Risks and mitigation

| Risk | Mitigation |
|------|-----------|
| Yiyu's baselines don't land before demo | Tell story via forecast-error sweep (0%/100% bracket replaces "MPC vs reactive") |
| Learned predictor MAE is too high | Falls back gracefully — comparable to GeometricMean by design (system-ID framing) |
| TimeExpandedMPC handover encoding has bugs | Invariant test (`H=1, λ=0` ≡ Dijkstra) catches gross errors; if subtle bugs persist, demote TEG to "future work" and ship rolling-replan only |
| Multi-flow coordination amplifies bugs | Test single-flow first; add second/third flow only after single-flow passes |
| 2-day timeline too tight | Day-1 hard cutoff is "rolling-replan + GeometricMean + runner + metrics smoke-tested"; everything after is bonus |

## Day-by-day execution (informational — for the user's planning)

**Day 1 (Wed 2026-05-07):**
- AM: package skeleton, `interfaces.py`, `flows.py`, `GeometricMeanPredictor`, `make_pred_error`, tests for those.
- Midday: `RollingReplanMPC` (single flow first, then multi-flow), `runner.py`, `metrics.py`. Smoke test on 30 timesteps.
- PM: `TimeExpandedMPC` + invariant test.
- Evening: `train.py` + `LearnedSysIDPredictor` + checkpoint script. Train + verify MAE.

**Day 2 (Thu 2026-05-08, presentation day):**
- AM: full `evaluate.py` sweep, `plot_results.py`, regenerate figures.
- Midday: integrate Yiyu's baselines if available; otherwise present forecast-error sweep narrative.
- PM: presentation polish, demo run.
