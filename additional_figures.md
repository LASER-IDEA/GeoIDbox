1. 按 held-out sensor / fold 的 LOSO 性能图

这是我最推荐的一幅，可以替代原来的 baseline comparison bar chart。

想表达什么

现在表格只能告诉读者“平均 MAE 谁更好”，但 reviewer 更关心的是：
PINF 的提升是不是在不同 held-out sensor 上都成立？
因为你的核心 claim 不是普通 supervised fitting，而是 cross-sensor generalization。所以最应该画的是 per-fold performance。

推荐画法

横轴放 8 个 held-out GeoBox sensor（或者 fold ID），纵轴放 MAE。
每个方法一条线，或者若觉得太拥挤，可以只放：

Physics baseline
RF
XGBoost
PINF (Ours)

如果 SIREN/TabNet 也想保留，可以在正文表里保留，在这张图里不一定全放，不然会太乱。

为什么这张图有用

它回答的是：

提升是不是稳定地出现在每个 fold；
哪些 sensor 比较难；
PINF 是不是只在少数 sensor 上特别好，从而拉低了平均值。
进一步优化

你可以把 8 个 sensor 按难度排序，比如按 physics baseline 的 MAE 从低到高排列。这样图会更有阅读性。
也可以在图中标注 mean MAE，或者在右侧加一个小的 overall mean marker。

2 boxplot / violin plot
横轴为不同高度关系，例如：

Raw barometric height minus HAE
Physics baseline height minus HAE
PINF-converted height minus HAE

纵轴为 height discrepancy。这样可以非常清楚地展示：
raw barometric height 和 physics baseline 仍存在大偏差，而 PINF conversion field 明显收敛到统一 reference。

这幅图支撑什么

它支撑的是文章最基础的命题：

VLL/AAM 中的问题不是“没有高度测量”，而是“不同高度测量不在同一 reference system 下”。

推荐放置位置

最好放在 Introduction 末尾或 Experiment 开头，作为 motivating quantitative evidence。

推荐 caption

Quantitative illustration of vertical-reference inconsistency in the GeoBox deployment area. The raw barometric height and the physics-based conversion exhibit large discrepancies with respect to the GNSS-derived geometric height reference, while the proposed PINF-based conversion field substantially reduces the discrepancy. This result demonstrates the necessity of a unified height conversion framework for VLL airspace applications.


3. 水平坐标扰动实验：Horizontal-position sensitivity analysis

非常适合回应 Reviewer 3.1。

Reviewer 3.1 关心的是 horizontal position 是否重要。你现在的回复说得很合理：水平坐标当然重要，但在本文中它不是估计目标，而是 neural field 的 spatial index。那最好的实验证据就是：给水平坐标加入合理范围的噪声，看看高度转换误差是否稳定。

可以怎么做

在 test / LOSO inference 阶段，对 held-out GeoBox 的水平坐标加入随机扰动：

0 m
1 m
2 m
5 m
10 m
20 m

扰动可以在 ENU 平面上加 Gaussian noise 或 uniform random direction。然后重复 20 或 50 次，统计 MAE / RMSE。

横轴：horizontal position perturbation magnitude
纵轴：altitude MAE

可以比较：

PINF
Physics baseline
maybe XGBoost 或 RF

但最关键的是 PINF 曲线。

你希望看到的结果

如果 1–5 m 水平扰动下 MAE 变化很小，就能强有力说明：

GeoBox GNSS 的 1.5 m CEP 对本文足够；
本文不是在解决 horizontal localization；
horizontal position 在本文中主要作为 spatial index；
模型对 realistic horizontal uncertainty 不敏感。
这幅图支撑什么

它直接支撑 Reviewer 3.1 的 two-fold 回复：

Horizontal position is important for VLL flights, but its reference system is already standardized through latitude/longitude. In this work, horizontal position is used as the spatial index of the height conversion field, and consumer-grade GNSS accuracy is sufficient for activating the relevant spatial features.

推荐 caption

Sensitivity of altitude conversion accuracy to horizontal-position perturbation. The held-out GeoBox coordinates are perturbed during inference to emulate horizontal positioning uncertainty. The proposed framework remains stable under meter-level perturbations, confirming that horizontal position is used as a spatial index rather than the target variable of the proposed height conversion task.

这幅图我认为很值得加，因为它既是新实验，又直接服务于 reviewer concern。



4. Error versus distance to nearest training GeoBox

这幅图和第 3 幅类似，但更空间化，适合解释泛化边界。

可以怎么做

在 LOSO 中，对于每个 held-out sensor，计算它到最近 training sensor 的水平距离。然后画：

横轴：distance to nearest training GeoBox
纵轴：held-out MAE

如果每个 sensor 有很多 timestamp，可以每个点用该 sensor 的 aggregate MAE。
如果只有 8 个点太少，可以在不同 random split 下生成更多 held-out cases。

这幅图支撑什么

它说明 neural field 的误差是否与空间外推距离有关。这个结果很有解释性：

如果距离越大误差越大，说明方法合理，但部署密度很关键；
如果距离大也稳定，说明 field reconstruction 更强；
无论哪种结果，都能帮助限定方法适用范围。
推荐 caption

Relationship between conversion error and spatial separation from training GeoBox nodes. The analysis characterizes the spatial generalization behavior of the proposed conversion field and clarifies the role of deployment density in practical VLL airspace applications.


5. Learned residual / physics residual decomposition

这幅图适合解释方法不是 black-box。

你现在的方法核心是 physics-informed residual learning。最好的解释图不是再画网络结构，因为你已经有方法框架和网络结构图了。当前已有 pipeline figure 和 architecture figure 已经展示了 Geobox、ERA5、terrain、hash encoding、MLP residual 等模块。
新增图应展示：模型到底学到了什么 residual。

可以怎么做

选择一个代表性 held-out sensor，画时间序列：

GNSS/HAE reference
Physics baseline height
PINF prediction
Learned residual correction

或者上下两幅：

上：height prediction time series
下：learned residual over time

这幅图支撑什么

它能说明 PINF 不是凭空回归高度，而是在 physics baseline 基础上学习 residual correction。
这对 rebuttal 很有用，尤其是 reviewer 质疑 deep learning black-box 时。

推荐 caption

Example of physics-informed residual correction on a held-out GeoBox node. The physics-only baseline captures the coarse altitude trend but retains systematic residual errors, while the learned correction term compensates for sensor-specific and local environmental effects, leading to a prediction closer to the geometric height reference.

6. Prediction uncertainty / risk-coverage curve

如果你想强调安全性，这是最有价值的新增实验。

VLL/AAM 不是普通 regression，安全相关系统更关心 large error。
如果你可以训练 ensemble，或者用 MC dropout / conformal prediction，能做一个 uncertainty-aware analysis。

可以怎么做

训练 5 个模型，使用不同 random seeds。对每个 sample 得到 prediction variance。
然后画：

Risk-coverage curve
横轴：coverage，即保留多少百分比的 samples
纵轴：MAE 或 95th percentile error

按 uncertainty 从低到高排序，逐步丢弃 high-uncertainty samples。
如果 uncertainty 有意义，coverage 降低时 MAE 会明显下降。

这幅图支撑什么

它说明模型不仅能输出高度，也能识别高风险预测区域。
这对“spike error cannot be fully eliminated but can be monitored”这一回复非常有用。

推荐 caption

Risk-coverage analysis based on ensemble predictive uncertainty. Removing high-uncertainty predictions progressively reduces the altitude conversion error, indicating that uncertainty can serve as an operational indicator for identifying potentially unreliable conversion results in VLL applications.