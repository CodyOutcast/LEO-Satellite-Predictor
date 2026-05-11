import sys
import json
import argparse
import time
from pathlib import Path

# 设置路径，确保可以导入 pred_mpc 和 scripts
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# 从现有模块导入组件
from scripts.evaluate import make_builder
from pred_mpc.controllers import RollingReplanMPC
from pred_mpc.my_baselines import ReactiveBaselineController, GreedyHandoverController
from pred_mpc.flows import DEFAULT_FLOWS
from pred_mpc.predictors import GeometricMeanPredictor, make_pred_error
from pred_mpc.runner import SimulationRunner
from pred_mpc.metrics import latency_cdf, outage_probability, handover_stats


def main():
    # 1. 增加命令行参数解析，保留 --quick 能力
    parser = argparse.ArgumentParser(description="Run full sweep for custom baselines")
    parser.add_argument("--quick", action="store_true", help="Run a short single-seed sweep")
    parser.add_argument("--output", type=Path, default=PKG_ROOT / "outputs" / "my_eval_records.jsonl")
    args = parser.parse_args()

    # 2. 根据 --quick 标志设置实验规模
    if args.quick:
        seeds = [42]
        steps = 30
        error_rates = [0.0, 0.5, 1.0]
    else:
        seeds = [0, 1, 2, 3, 4]  # 大规模扫描使用多个种子
        steps = 180
        error_rates = [0.0, 0.1, 0.3, 0.5, 1.0]

    rows = []
    total_runs = len(seeds) * len(error_rates) * 3  # 3个控制器
    completed_runs = 0
    start_time = time.perf_counter()

    # 3. 自动化遍历：种子 -> 错误率 -> 控制器
    for seed in seeds:
        # 为每个种子构建特定的仿真环境
        builder = make_builder(seed=seed, t_end_s=float(steps * 2))

        for error_rate in error_rates:
            # 注入预测误差
            pred_error = make_pred_error(error_rate, kind="snr_jitter", link_model_cfg=builder.cfg.link_model)
            predictor = GeometricMeanPredictor(builder, pred_error=pred_error)

            # 定义参与对比的控制器集合
            test_suite = {
                "MPC_Rolling": RollingReplanMPC(builder, predictor, DEFAULT_FLOWS, H=5, lambda_handover=0.5),
                "Baseline_Reactive": ReactiveBaselineController(builder, predictor, DEFAULT_FLOWS),
                "Baseline_Greedy": GreedyHandoverController(builder, predictor, DEFAULT_FLOWS)
            }

            for name, controller in test_suite.items():
                completed_runs += 1
                print(f"[{completed_runs}/{total_runs}] Running: Seed={seed}, Error={error_rate}, Controller={name}")

                # 运行仿真并搜集原始数据
                runner = SimulationRunner(builder, controller, DEFAULT_FLOWS)
                records = runner.run(t_start=0, t_end=steps)

                # 计算核心指标
                outage_dict = outage_probability(records, by_flow=True)
                latency_dict = latency_cdf(records)
                handover_dict = handover_stats(records, dwell_window_tau=5, dt_s=builder.cfg.sim.dt_s)

                # 保存每个 flow 的结果
                for flow in DEFAULT_FLOWS:
                    f_name = flow.name
                    flow_lat = latency_dict.get(f_name, {})
                    flow_handover = handover_dict.get(f_name, {})

                    rows.append({
                        "seed": seed,
                        "error_rate": error_rate,
                        "predictor": "GeometricMean",
                        "controller": name,
                        "flow": f_name,
                        "outage_prob": float(outage_dict.get(f_name, 0.0)),
                        "p50_latency_s": float(flow_lat.get(50, 0.0)),
                        "p95_latency_s": float(flow_lat.get(95, 0.0)),
                        "handover_rate_per_min": float(flow_handover.get("handover_rate_per_min", 0.0)),
                        "regret_rate": float(flow_handover.get("regret_rate", 0.0))
                    })

    # 4. 结果持久化与自动绘图
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n[Done] Sweep completed in {(time.perf_counter() - start_time) / 60:.1f} min.")
    print(f"Results saved to: {args.output}")

    import subprocess
    print("\n[Step 2/2] Generating Visualizations...")
    plot_script = PKG_ROOT / "scripts" / "plot_results.py"
    try:
        subprocess.run([sys.executable, str(plot_script), "--input", str(args.output)], check=True)
        print(f"Success! Check plots in: {PKG_ROOT / 'outputs'}")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()