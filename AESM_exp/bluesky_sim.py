import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. 实验参数配置 (基于 PX4 实测数据)
# ==========================================
# 你的实测数据
SIGMA_BARO = 23.65  # meters
SIGMA_HAE  = 1.67   # meters

# 安全系数 (Safety Factor): 3.5 sigma (99.95% confidence)
SAFETY_FACTOR = 3.5

# 计算导出的 VSM (垂直间隔标准)
VSM_BARO = SAFETY_FACTOR * np.sqrt(2) * SIGMA_BARO
VSM_HAE  = SAFETY_FACTOR * np.sqrt(2) * SIGMA_HAE
# 工程修正：HAE虽准，但考虑到机身尺寸，给予最小 10m 的硬限制
VSM_HAE = max(VSM_HAE, 10.0)

# 空域设置：深圳某物流走廊
CORRIDOR_LENGTH = 10000 # 10 km
CORRIDOR_WIDTH  = 500   # 500 m
AIRSPACE_CEILING = 300  # 300 m AGL
SPEED_MPS = 15          # 无人机速度 15 m/s (~54 km/h)

# 仿真设置
SIM_DURATION = 3600     # 模拟 1小时
DT = 5                  # 时间步长 5秒

print(f"=== Simulation Parameters ===")
print(f"Empirical Baro Sigma: {SIGMA_BARO}m -> Calculated VSM: {VSM_BARO:.2f}m")
print(f"Empirical HAE Sigma : {SIGMA_HAE}m  -> Calculated VSM: {VSM_HAE:.2f}m")
print(f"=============================")

# ==========================================
# 2. 仿真核心类定义
# ==========================================
class Aircraft:
    def __init__(self, id, x, y, z_true, sigma_error):
        self.id = id
        self.x = x
        self.y = y
        self.z_true = z_true
        self.sigma_error = sigma_error
        self.vx = SPEED_MPS # 简单假设都向右飞
        self.active = True

        # 初始测量高度
        self.update_measurement()

    def update_measurement(self):
        # 注入高斯噪声 (这是关键！基于你的 PX4 数据)
        noise = np.random.normal(0, self.sigma_error)
        self.z_measured = self.z_true + noise

    def move(self, dt):
        self.x += self.vx * dt
        # 更新测量值 (模拟传感器跳变/噪声)
        self.update_measurement()

        # 飞出走廊则消失
        if self.x > CORRIDOR_LENGTH:
            self.active = False

def run_saturation_test(traffic_density_per_hour, system_type):
    """
    运行一次压力测试
    traffic_density_per_hour: 流量注入速率 (架次/小时)
    system_type: 'BARO' 或 'HAE'
    """

    # 设定该模式下的 VSM 和 误差 Sigma
    if system_type == 'BARO':
        current_vsm = VSM_BARO
        current_sigma = SIGMA_BARO
    else:
        current_vsm = VSM_HAE
        current_sigma = SIGMA_HAE

    # 计算可用高度层 (Flight Levels)
    # 我们使用简单的层级分配逻辑：尽量填满空域
    # 第1层高度 = VSM/2, 第2层 = VSM/2 + VSM...
    available_levels = []
    h = current_vsm / 2
    while h < AIRSPACE_CEILING:
        available_levels.append(h)
        h += current_vsm

    if len(available_levels) == 0:
        return 0, 0 # 没法飞

    # 初始化
    aircraft_list = []
    total_conflicts = 0
    spawn_interval = 3600 / traffic_density_per_hour # 秒/架
    next_spawn_time = 0
    t = 0
    ac_counter = 0

    # 循环模拟时间步
    for t in np.arange(0, SIM_DURATION, DT):

        # 1. 生成飞机 (Traffic Generation)
        while t >= next_spawn_time:
            # 随机选择一个高度层
            z_assign = np.random.choice(available_levels)
            # 随机Y轴位置 (增加一点随机性)
            y_assign = np.random.uniform(0, CORRIDOR_WIDTH)

            new_ac = Aircraft(ac_counter, 0, y_assign, z_assign, current_sigma)
            aircraft_list.append(new_ac)

            ac_counter += 1
            next_spawn_time += spawn_interval

        # 2. 更新位置
        active_ac = []
        for ac in aircraft_list:
            ac.move(DT)
            if ac.active:
                active_ac.append(ac)
        aircraft_list = active_ac

        # 3. 冲突检测 (Conflict Detection) - BlueSky Logic
        # 复杂度 O(N^2)，但在低空容量测试中是必要的
        count = len(aircraft_list)
        for i in range(count):
            for j in range(i + 1, count):
                ac1 = aircraft_list[i]
                ac2 = aircraft_list[j]

                # 水平距离判断 (简单设定水平保护区 50m)
                d_horiz = np.sqrt((ac1.x - ac2.x)**2 + (ac1.y - ac2.y)**2)
                if d_horiz < 50:
                    # 垂直距离判断 (关键！使用测量高度 + VSM)
                    d_vert = abs(ac1.z_measured - ac2.z_measured)

                    if d_vert < current_vsm:
                        # 这是一个由于测量误差导致的"感知冲突"
                        # 或者由于空间拥挤导致的"真实冲突"
                        total_conflicts += 1

    # 返回：平均每架飞机的冲突数 (Conflict Ratio)
    if ac_counter == 0: return 0
    return total_conflicts / ac_counter

# ==========================================
# 3. 批量执行压力测试 (Monte Carlo)
# ==========================================
densities = np.arange(50, 1050, 50) # 测试流量：从 50 到 1000 架次/小时
baro_results = []
hae_results = []

print("Running Saturation Stress Test...")
for d in tqdm(densities):
    # 跑 Baro 模式
    c_baro = run_saturation_test(d, 'BARO')
    baro_results.append(c_baro)

    # 跑 HAE 模式
    c_hae = run_saturation_test(d, 'HAE')
    hae_results.append(c_hae)

# ==========================================
# 4. 绘图与结果输出
# ==========================================
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(densities, baro_results, 'o-', color='orange', linewidth=2, label=f'Legacy (Baro): VSM={int(VSM_BARO)}m')
plt.plot(densities, hae_results, 's-', color='blue', linewidth=2, label=f'Proposed (HAE): VSM={int(VSM_HAE)}m')

# 设定安全阈值线 (假设每架飞机 0.1 次冲突是不可接受的)
THRESHOLD = 0.1
plt.axhline(THRESHOLD, color='red', linestyle='--', alpha=0.5, label='Safety Threshold')

# 标注容量点
def find_capacity(results):
    for i, r in enumerate(results):
        if r > THRESHOLD:
            return densities[i]
    return densities[-1]

cap_baro = find_capacity(baro_results)
cap_hae = find_capacity(hae_results)

plt.annotate(f'Baro Capacity: ~{cap_baro} flight/h', xy=(cap_baro, THRESHOLD), xytext=(cap_baro, THRESHOLD+0.5),
             arrowprops=dict(facecolor='orange', shrink=0.05))
plt.annotate(f'HAE Capacity: >{cap_hae} flight/h', xy=(cap_hae, THRESHOLD), xytext=(cap_hae, THRESHOLD+0.2),
             arrowprops=dict(facecolor='blue', shrink=0.05))

plt.title('Airspace Capacity Saturation Analysis\n(Based on Empirical PX4 Flight Log Errors)', fontsize=14)
plt.xlabel('Traffic Demand (flights per hour)', fontsize=12)
plt.ylabel('Conflict Risk (conflicts per flight)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

# 保存图片
plt.savefig('Capacity_Saturation_Result.png', dpi=300)
plt.show()

print(f"Simulation Complete.")
print(f"Baro Capacity Limit: ~{cap_baro} flights/hour (Limited by large VSM of {int(VSM_BARO)}m)")
print(f"HAE Capacity Limit : >{cap_hae} flights/hour (Enabled by small VSM of {int(VSM_HAE)}m)")