import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. 参数配置（最新不确定性建模结果）
# ==========================================
STATS_PATH = Path("sensor_uncertainty_stats.json")
if STATS_PATH.exists():
    with STATS_PATH.open() as fp:
        STATS = json.load(fp)
else:
    STATS = {}

SIGMA_BARO = STATS.get("abs_error_fit_sigma", 23.65)
SIGMA_HAE = STATS.get("epv_fit_sigma", 1.67)

# 采用 RVSM 风险缓释：VSM = 3.5 * sqrt(2) * sigma
VSM_BARO = 3.5 * np.sqrt(2) * SIGMA_BARO
VSM_HAE = 3.5 * np.sqrt(2) * SIGMA_HAE

AIRSPACE_CEILING = 1000.0  # m
CAPACITY_BARO = max(1, int(AIRSPACE_CEILING / VSM_BARO))
CAPACITY_HAE = max(1, int(AIRSPACE_CEILING / VSM_HAE))
QOS_LIMIT = 0.05

SERVICE_TIME_HOURS = 10.0 / 60.0  # 10 min = 1/6 hr

print("=== Airspace Resource Configuration (RTK-Calibrated) ===")
print(f"Barometric Mode : σ={SIGMA_BARO:.2f} m -> VSM={VSM_BARO:.1f} m -> {CAPACITY_BARO} layers")
print(f"GNSS HAE Mode   : σ={SIGMA_HAE:.2f} m -> VSM={VSM_HAE:.1f} m -> {CAPACITY_HAE} layers")
print("========================================================")


def erlang_b(c, traffic):
    """Erlang-B blocking probability for c servers and traffic load a."""
    B = 1.0
    for k in range(1, c + 1):
        B = (traffic * B) / (k + traffic * B)
    return B


def rejection_curve(arrival_rates, layers):
    """Compute denied-service probability using Erlang-B loss model."""
    probs = []
    for rate in arrival_rates:
        offered_load = rate * SERVICE_TIME_HOURS  # Erlangs
        probs.append(erlang_b(layers, offered_load))
    return np.array(probs)


rates = np.linspace(10, 4200, 60)  # flights per hour
res_baro = rejection_curve(rates, CAPACITY_BARO)
res_hae = rejection_curve(rates, CAPACITY_HAE)


def find_threshold(rates, values, threshold):
    mask = values >= threshold
    if not mask.any():
        return rates[-1], values[-1], False
    idx = np.argmax(mask)
    return rates[idx], values[idx], True


baro_pt = find_threshold(rates, res_baro, QOS_LIMIT)
hae_pt = find_threshold(rates, res_hae, QOS_LIMIT)

# ==========================================
# 绘图
# ==========================================
plt.figure(figsize=(12, 6.5))
sns.set_theme(style="ticks", font_scale=1.35)

plt.plot(rates, res_baro, color="#ff7f0e", linewidth=2.5,
         label=f"Legacy Barometric Stack\n(VSM≈{VSM_BARO:.0f} m, Layers={CAPACITY_BARO})")
plt.plot(rates, res_hae, color="#1f77b4", linewidth=2.5,
         label=f"GNSS HAE Stack\n(VSM≈{VSM_HAE:.0f} m, Layers={CAPACITY_HAE})")

plt.axhline(QOS_LIMIT, color="red", linestyle="--", alpha=0.5, label="QoS Limit: 5% denials")

plt.xlabel("Traffic Demand (flights per hour)", fontsize=13, fontweight="bold")
plt.ylabel("Denied-Service Probability", fontsize=13, fontweight="bold")
plt.title("Airspace Slot Availability vs. Demand (Analytical Loss Model)",
          fontsize=16, fontweight="bold")
plt.ylim(0, 1.05)
plt.xlim(0, rates.max())
plt.legend(loc="upper left", frameon=True, fontsize=12, title="Height System")

if baro_pt[2]:
    target_x = baro_pt[0] - rates.max() * 0.2 if baro_pt[0] > rates.max() * 0.4 else baro_pt[0] + rates.max() * 0.15
    ha = "right" if baro_pt[0] > rates.max() * 0.4 else "left"
    plt.annotate(f"Barometric QoS limit\n≈{baro_pt[0]:.0f} flights/hr",
                 xy=(baro_pt[0], baro_pt[1]),
                 xytext=(target_x, min(0.95, baro_pt[1] + 0.25)),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=11, ha=ha)

if hae_pt[2]:
    target_x = hae_pt[0] + rates.max() * 0.15 if hae_pt[0] < rates.max() * 0.6 else hae_pt[0] - rates.max() * 0.15
    ha = "left" if hae_pt[0] < rates.max() * 0.6 else "right"
    plt.annotate(f"GNSS QoS limit\n≈{hae_pt[0]:.0f} flights/hr",
                 xy=(hae_pt[0], hae_pt[1]),
                 xytext=(target_x, max(0.12, hae_pt[1] + 0.4)),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=11, ha=ha)
else:
    plt.annotate("GNSS/HAE stack remains\nunder 5% denials in this range",
                 xy=(rates[-1], res_hae[-1]),
                 xytext=(rates[-1]*0.4, max(0.15, res_hae[-1] + 0.5)),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=11, ha="center")

plt.tight_layout()
plt.savefig("Capacity_Gap_Result.png", dpi=300)
plt.show()