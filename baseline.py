import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class MultiOrbitLEOSim:
    def __init__(self):
        # 1. 物理常数
        self.R_EARTH = 6371.0
        self.MU = 398600.44
        self.DT = 2.0
        self.TOTAL_TIME = 6000

        # 2. 轨道平面配置：现在每个平面可以有不同的卫星数量
        self.planes_config = [
            {'h': 550.0, 'inc': np.radians(45), 'raan': 0, 'num_sats': 12},
            {'h': 600.0, 'inc': np.radians(-45), 'raan': np.radians(60), 'num_sats': 8},
            {'h': 500.0, 'inc': np.radians(90), 'raan': np.radians(120), 'num_sats': 20}
        ]

        # 3. 初始化每颗卫星的独立属性
        self.sat_metadata = {}
        for p_idx, config in enumerate(self.planes_config):
            # 为该轨道面计算角速度
            orbit_omega = np.sqrt(self.MU / (self.R_EARTH + config['h']) ** 3)
            num_sats = config['num_sats']

            for s_idx in range(num_sats):
                sat_id = f"P{p_idx}-S{s_idx}"
                # 在该轨道面内均匀分布，并加入随机初始偏移
                initial_phase = (2 * np.pi / num_sats) * s_idx + np.random.uniform(0, 0.1)

                self.sat_metadata[sat_id] = {
                    'omega': orbit_omega,
                    'initial_phase': initial_phase,
                    'h': config['h'],
                    'inc': config['inc'],
                    'raan': config['raan']
                }

        # 4. 约束与基站
        self.ELEVATION_MIN = 15.0
        self.RANGE_MAX = 2500.0
        self.gs_src = self.lat_lon_to_cartesian(22.5, 114.0, 0)  # 深圳
        self.gs_dst = self.lat_lon_to_cartesian(31.2, 121.5, 0)  # 上海

        self.metrics = {"time": [], "latency": [], "handovers": 0}

    def lat_lon_to_cartesian(self, lat, lon, alt):
        phi, theta = np.radians(lat), np.radians(lon)
        r = self.R_EARTH + alt
        return r * np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])

    def get_sat_pos_3d(self, t, sat_id):
        meta = self.sat_metadata[sat_id]
        current_angle = meta['initial_phase'] + meta['omega'] * t
        r = self.R_EARTH + meta['h']
        x_p, y_p = r * np.cos(current_angle), r * np.sin(current_angle)
        inc, raan = meta['inc'], meta['raan']
        x = x_p * np.cos(raan) - y_p * np.cos(inc) * np.sin(raan)
        y = x_p * np.sin(raan) + y_p * np.cos(inc) * np.cos(raan)
        z = y_p * np.sin(inc)
        return np.array([x, y, z])

    def is_visible(self, gs_pos, sat_pos):
        dist = np.linalg.norm(sat_pos - gs_pos)
        if dist > self.RANGE_MAX: return False, dist
        vec_gs_sat = sat_pos - gs_pos
        cos_ele = np.dot(gs_pos, vec_gs_sat) / (np.linalg.norm(gs_pos) * dist)
        elevation = np.degrees(np.arcsin(np.clip(cos_ele, -1.0, 1.0)))
        return elevation >= self.ELEVATION_MIN, dist

    def run_reactive_baseline(self):
        prev_access_sat = None

        for t in np.arange(0, self.TOTAL_TIME, self.DT):
            G_t = nx.Graph()

            # --- 修正：显式添加基站节点，防止 NodeNotFound 报错 ---
            #G_t.add_node('SRC')
            #G_t.add_node('DST')

            sat_positions = {}
            for sat_id in self.sat_metadata.keys():
                pos = self.get_sat_pos_3d(t, sat_id)
                sat_positions[sat_id] = pos
                G_t.add_node(sat_id)

            # 添加链路
            for gs_name, gs_pos in [('SRC', self.gs_src), ('DST', self.gs_dst)]:
                for sat_id, s_pos in sat_positions.items():
                    visible, dist = self.is_visible(gs_pos, s_pos)
                    if visible:
                        G_t.add_edge(gs_name, sat_id, weight=dist / 300.0)

            # 星间链路 (ISL)
            sat_ids = list(self.sat_metadata.keys())
            for i in range(len(sat_ids)):
                for j in range(i + 1, len(sat_ids)):
                    d = np.linalg.norm(sat_positions[sat_ids[i]] - sat_positions[sat_ids[j]])
                    if d < 2000.0:
                        G_t.add_edge(sat_ids[i], sat_ids[j], weight=d / 300.0)

            # 计算路径
            try:
                # 检查节点是否连通（安全检查）
                if G_t.has_node('SRC') and G_t.has_node('DST'):
                    path = nx.shortest_path(G_t, 'SRC', 'DST', weight='weight')
                    latency = nx.shortest_path_length(G_t, 'SRC', 'DST', weight='weight')

                    current_access = path[1]
                    if prev_access_sat and current_access != prev_access_sat:
                        self.metrics["handovers"] += 1
                    prev_access_sat = current_access
                    self.metrics["latency"].append(latency)
                else:
                    self.metrics["latency"].append(None)
            except (nx.NetworkXNoPath, IndexError):
                self.metrics["latency"].append(None)
            self.metrics["time"].append(t)

        self.plot_results()

    def run_greedy_baseline(self):
        """
        贪心切换基线：只要当前路径有效，就保持原路径不切换，以最小化切换次数。
        """
        current_path = None
        prev_access_sat = None
        self.metrics = {"time": [], "latency": [], "handovers": 0}

        for t in np.arange(0, self.TOTAL_TIME, self.DT):
            G_t = nx.Graph()

            # 1. 构建当前时刻的网络拓扑（与最短路径基线逻辑完全一致）
            sat_positions = {}
            for sat_id in self.sat_metadata.keys():
                pos = self.get_sat_pos_3d(t, sat_id)
                sat_positions[sat_id] = pos
                G_t.add_node(sat_id)

            for gs_name, gs_pos in [('SRC', self.gs_src), ('DST', self.gs_dst)]:
                for sat_id, s_pos in sat_positions.items():
                    visible, dist = self.is_visible(gs_pos, s_pos)
                    if visible:
                        G_t.add_edge(gs_name, sat_id, weight=dist / 300.0)

            sat_ids = list(self.sat_metadata.keys())
            for i in range(len(sat_ids)):
                for j in range(i + 1, len(sat_ids)):
                    d = np.linalg.norm(sat_positions[sat_ids[i]] - sat_positions[sat_ids[j]])
                    if d < 2000.0:
                        G_t.add_edge(sat_ids[i], sat_ids[j], weight=d / 300.0)

            # 2. 贪心路由核心逻辑
            path_is_valid = False

            # 检查当前持有的路径在此时刻的拓扑 G_t 中是否依然完全连通
            if current_path is not None:
                path_is_valid = True
                for i in range(len(current_path) - 1):
                    u = current_path[i]
                    v = current_path[i + 1]
                    if not G_t.has_edge(u, v):
                        path_is_valid = False
                        break

            if path_is_valid:
                # 路径仍然有效，继续使用原路径，不发生切换
                # 计算该旧路径在当前时刻的新延迟（权重可能因为距离变化而微调）
                latency = sum(G_t[current_path[i]][current_path[i + 1]]['weight'] for i in range(len(current_path) - 1))
                self.metrics["latency"].append(latency)

                # 接入卫星保持不变
                prev_access_sat = current_path[1]

            else:
                # 原路径断开了，或者当前处于无连接状态，被迫寻找新路径
                try:
                    if G_t.has_node('SRC') and G_t.has_node('DST'):
                        # 只有在迫不得已时，才使用 Dijkstra 寻找当前的最短路径
                        new_path = nx.shortest_path(G_t, 'SRC', 'DST', weight='weight')
                        latency = nx.shortest_path_length(G_t, 'SRC', 'DST', weight='weight')

                        current_access = new_path[1]
                        # 判定是否发生切换：之前有连接，且现在的接入星和刚才的不同
                        if prev_access_sat is not None and current_access != prev_access_sat:
                            self.metrics["handovers"] += 1

                        prev_access_sat = current_access
                        current_path = new_path
                        self.metrics["latency"].append(latency)
                    else:
                        # 彻底没有路径
                        self.metrics["latency"].append(None)
                        current_path = None
                        prev_access_sat = None
                except (nx.NetworkXNoPath, IndexError):
                    self.metrics["latency"].append(None)
                    current_path = None
                    prev_access_sat = None

            self.metrics["time"].append(t)
        self.plot_results()
        return self.metrics

    def evaluate_metrics(self):
        # 提取有效的延迟数据（剔除中断状态的 None）
        valid_latencies = [l for l in self.metrics["latency"] if l is not None]
        total_steps = len(self.metrics["latency"])

        # 1. 计算中断概率 (Outage Probability)
        outage_steps = total_steps - len(valid_latencies)
        outage_probability = outage_steps / total_steps if total_steps > 0 else 0

        # 2. 计算切换频率 (Handover Frequency)
        # 转换为标准化指标，例如：每小时切换次数
        total_hours = self.TOTAL_TIME / 3600.0
        handover_freq = self.metrics["handovers"] / total_hours if total_hours > 0 else 0

        # 3. 延迟分布统计 (Latency Distribution)
        if valid_latencies:
            avg_latency = np.mean(valid_latencies)
            p99_latency = np.percentile(valid_latencies, 99)  # 99分位延迟，衡量长尾效应
        else:
            avg_latency, p99_latency = float('inf'), float('inf')

        # 打印综合评估报告
        print("=" * 40)
        print("Baseline Simulation Metrics Report")
        print("=" * 40)
        print(f"Total Simulation Time: {self.TOTAL_TIME} seconds ({total_hours:.2f} hours)")
        print(f"Outage Probability:    {outage_probability:.2%}")
        print(f"Handover Frequency:    {handover_freq:.2f} handovers/hour (Total: {self.metrics['handovers']})")
        if valid_latencies:
            print(f"Average Latency:       {avg_latency:.2f}")
            print(f"99th Pct Latency:      {p99_latency:.2f}")
        print("=" * 40)

        return valid_latencies, outage_probability

    def plot_results(self):
        valid_latencies, outage_prob = self.evaluate_metrics()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 图1: 延迟随时间变化及中断情况
        latencies_plot = [l if l is not None else np.nan for l in self.metrics["latency"]]
        ax1.plot(self.metrics["time"], latencies_plot, label='Shortest Path Latency', color='#1f77b4', linewidth=1.5)

        # 标记中断区域 (Outage)
        outage_times = [self.metrics["time"][i] for i, l in enumerate(self.metrics["latency"]) if l is None]
        if outage_times:
            ax1.scatter(outage_times, [0] * len(outage_times), color='red', marker='x', s=10, label='Outage (No Path)')

        ax1.set_title(f"Time-varying Latency (Handovers: {self.metrics['handovers']})")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Latency (Weight)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # 图2: 延迟分布 (Latency Distribution)
        if valid_latencies:
            # 使用直方图展示延迟的概率密度分布
            ax2.hist(valid_latencies, bins=40, color='#2ca02c', edgecolor='black', alpha=0.7, density=True)
            ax2.set_title(f"Latency Distribution (Outage: {outage_prob:.2%})")
            ax2.set_xlabel("Latency (Weight)")
            ax2.set_ylabel("Density")
            ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = MultiOrbitLEOSim()
    sim.run_reactive_baseline()
    sim.run_greedy_baseline()