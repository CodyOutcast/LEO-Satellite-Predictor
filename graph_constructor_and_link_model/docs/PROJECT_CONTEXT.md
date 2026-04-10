# Project Context (Background)

This file preserves the broader course-project context around the simulator.
It is **not required** to use the graph constructor/link model, but it may be
helpful for understanding what downstream routing/handover modules will
optimize and measure.

---

# Interim Report Instructions (Markdown)

## Title
Prediction-Aware Routing and Handover in Time-Varying LEO-Ground Networks

## Authors
- Zhuokai Chen (123090057)
- Yiyu Ren (122090445)
- Zhangcheng Kang (122090873)
- Kudzai Dandadzi (122010002)

## Abstract
LEO connectivity changes on the order of seconds to minutes, so a path that is optimal now can fail soon after. We will build a time-varying LEO-ground network simulator and evaluate prediction-aware routing/handover that optimizes over a short forecast horizon. We measure outage probability, 95th-percentile latency, and handover rate versus reactive baselines.

## 1) Title + Type (Frontier)
- **Type:** Frontier
- **Artifact:** A time-varying network-graph simulator plus routing/handover policies that use short-horizon forecasts of link availability to reduce outage probability and tail latency compared to purely reactive routing.

## 2) Problem + Metrics
### Setting
Geometry drives connectivity in LEO: visibility windows open/close quickly and inter-satellite distances evolve continuously. Reactive routing can pick paths that are about to break, causing short outages or excessive switching.

### Success Metrics (measured over simulation time)
- **Outage probability:** fraction of timesteps with no feasible end-to-end path between the chosen endpoints.
- **95th-percentile latency:** tail end-to-end delay across delivered packets/flows.
- **Handover rate:** handovers per minute, plus a regret variant that counts switches that were not needed within a short dwell window $\tau$.

## 3) Technical Core
### Model / Setup
We model the network as snapshots $G(t)=(V,E(t))$ sampled every $\Delta t$ seconds. At each timestep, we route traffic for a fixed set of source-destination pairs (e.g., ground station to ground station via satellites).

- **Nodes:** satellites (LEO) and ground stations.
- **Edges:** links exist when constraints hold (elevation $\geq \theta_{\min}$; range $\leq r_{\max}$).
- **Weights:** per-edge cost combines propagation delay and a reliability penalty (SNR proxy), yielding an end-to-end path cost.
- **Stochasticity:** additive noise models fading/interference so we can report distributions.

Implementation: Python + NetworkX; log paths, outages, and handovers each timestep.

### Methodology Anchors
- **Graph algorithms:** snapshot shortest paths on $G(t)$ and time-expanded constructions for look-ahead.
- **Stochastic modeling:** link states/costs vary with geometry plus noise, yielding outage/latency distributions.
- **Optimization/control:** a short-horizon objective that trades expected latency vs. handover stability.

## 4) Baselines + Evaluation Plan
### Baselines
- **Reactive shortest path:** run Dijkstra on current snapshot $G(t)$ and route immediately.
- **Greedy handover:** at each timestep, attach each ground station to the currently best visible satellite (strongest link-quality proxy), ignoring near-future drops.

### Prediction-Aware Policy (Candidate)
At each time $t$, the controller consumes a horizon-$H$ forecast of edge availability and/or costs (from geometry + extrapolation; optional learned predictor). It chooses a route and access links by minimizing predicted cost with an explicit switching penalty:

$$
\min \sum_{k=0}^{H-1} \widehat{C}(t+k) + \lambda\,\mathbf{1}[\text{handover at } t+k].
$$

Compare:
- Time-expanded routing over $H$.
- Cheaper rolling replan that scores candidate paths by predicted link survival.

### Experiments / Scenarios
- **Scale/geometry:** vary satellite count, plane count/inclination, and ISL radius.
- **Stress:** random link/satellite failures and noisy link costs.
- **Forecast quality:** inject controlled forecast error (e.g., 10%/30%/50%) and measure robustness.

### Outputs
- Outage probability
- Latency CDF (50th/95th)
- Handover statistics with confidence intervals

## 5) Plan + Risks
### Milestones
1. **Simulator MVP:** implement $G(t)$ generator + link-cost model; validate connectivity and basic sanity checks.
2. **Baselines + harness:** implement reactive shortest path + greedy handover; metric logging and plotting.
3. **Prediction-aware policy + eval:** implement horizon-$H$ controller; run robustness-to-forecast-error sweeps.

### Top Risks + Mitigation
- **Model fidelity vs. time:** keep a simple, testable geometry/link model first; only add realism (e.g., SGP4) after the baseline simulator is stable.
- **Forecast errors:** design controller to degrade gracefully (bounded horizon, limited switching) and benchmark break-even accuracy where look-ahead helps.
- **Overfitting to one scenario:** evaluate across diverse constellation/stressor settings and report robustness curves instead of a single best case.

## 6) Frontier-Only: Verification + Fail-Soft Fallback
### Verification Plan
- **Ablation:** oracle forecasts vs. noisy forecasts vs. no forecasts (reactive).
- **Robustness:** plot outage and 95th latency as a function of injected forecast error.
- **Sanity checks:** ensure invariants (e.g., as forecast error approaches 100%, performance approaches reactive) and report regimes where look-ahead hurts.

### Fail-Soft Fallback Deliverable
If learning-based prediction is unreliable, deliver the simulator + a prediction-aware controller using a simple hand-crafted predictor, with full baseline comparisons.

## 7) AI Usage Statement
- **Tools:** GitHub Copilot (boilerplate and refactors); LLM chat tools for drafting text and pseudocode.
- **What we will verify:** geometry/link calculations via unit tests and sanity checks; all plots and summary tables reproduced from scripts with fixed seeds.
- **Foreseeable failure mode:** plausible-looking but incorrect modeling choice (e.g., link budget scaling), mitigated by cross-checking formulas and validating monotonicity/invariants (e.g., path-loss increases with range).

## 8) Team Plan (Roles + Percentages)
- **Zhuokai Chen (25%):** graph simulator + link model.
- **Yiyu Ren (25%):** baseline routing + evaluation harness.
- **Zhangcheng Kang (25%):** prediction module + MPC integration.
- **Kudzai Dandadzi (25%):** experiments, analysis, report/figures.
