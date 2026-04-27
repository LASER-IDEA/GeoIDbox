"""
交互式分位数缩放工具。

功能：
1. 复用 rebuttal 脚本的 LOSO 推理结果收集逻辑
2. 将绝对误差按 10 个分位区间划分
3. 每个分位区间提供一个滑条缩放系数
4. 实时显示 MAE / Median / P90 / P99

运行方式：
streamlit run experiments/interactive_percentile_scaling_tool.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import pickle
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from generate_rebuttal_figures import collect_all_fold_errors


CACHE_DIR = Path("experiments/artifacts")
CACHE_PATH = CACHE_DIR / "fold_data_cache.pkl"


def is_streamlit_runtime() -> bool:
    """判断当前是否在 streamlit run 上下文中运行。"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


@st.cache_data(show_spinner=True)
def load_abs_errors() -> np.ndarray:
    """加载 8-fold 的绝对误差并拼接（优先使用本地缓存）。"""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            fold_data = pickle.load(f)
    else:
        fold_data = collect_all_fold_errors()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(fold_data, f)

    all_abs_errors = np.concatenate([fold_data[f]["abs_errors"] for f in range(8)])
    return all_abs_errors.astype(np.float64)


def build_percentile_bins(values: np.ndarray):
    """构建自定义分位区间以及每个样本所属区间。"""
    # 用户要求：P0-P80, P81-P90, P91-P93，然后其余部分均匀划分
    # 为保持 10 个可调滑条，这里将 P93-P100 再均匀切成 7 段。
    pct_edges = np.array([0.0, 80.0, 90.0, 95.0, 97.0, 100.0])
    edges = np.percentile(values, pct_edges)
    bin_ids = np.digitize(values, edges[1:-1], right=False)
    return edges, bin_ids, pct_edges


def compute_metrics(abs_errors: np.ndarray) -> dict:
    """计算核心指标。"""
    return {
        "mae": float(np.mean(abs_errors)),
        "median": float(np.percentile(abs_errors, 50)),
        "p90": float(np.percentile(abs_errors, 90)),
        "p95": float(np.percentile(abs_errors, 95)),
        "p99": float(np.percentile(abs_errors, 99)),
    }

DEFAULT_SCALE = np.array([3, 1.84, 1.37, 1.06, 0.73], dtype=np.float64)
# DEFAULT_SCALE = np.array([1.0] * 5, dtype=np.float64)

def main():
    st.set_page_config(page_title="Percentile Scaling Tool", layout="wide")

    st.title("10-Percentile Error Scaling Tool")
    st.caption("按绝对误差分位区间进行缩放，并实时查看 MAE / Median / P90 / P99。")

    st.info(f"Fold cache: {CACHE_PATH}")

    with st.spinner("Loading fold errors from checkpoints..."):
        abs_errors = load_abs_errors()

    edges, bin_ids, pct_edges = build_percentile_bins(abs_errors)

    st.subheader("Scaling Factors (5 Percentiles)")
    st.write("每个滑条控制对应分位区间内样本的误差缩放系数。")

    col_left, col_right = st.columns([2, 3])

    factors = np.ones(5, dtype=np.float64)
    with col_left:
        for i in range(5):
            label = (
                f"P{pct_edges[i]:.0f}-P{pct_edges[i + 1]:.0f} "
                f"[{edges[i]:.3f}, {edges[i + 1]:.3f}]"
            )
            factors[i] = st.slider(
                label,
                min_value=0.0,
                max_value=8.0,
                value=float(DEFAULT_SCALE[i]),
                step=0.01,
                key=f"scale_{i}",
            )

    scaled_abs_errors = abs_errors * factors[bin_ids]

    base_metrics = compute_metrics(abs_errors)
    scaled_metrics = compute_metrics(scaled_abs_errors)

    with col_right:
        st.subheader("Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "MAE",
            f"{scaled_metrics['mae']:.4f}",
            delta=f"{scaled_metrics['mae'] - base_metrics['mae']:+.4f}",
        )
        m2.metric(
            "Median",
            f"{scaled_metrics['median']:.4f}",
            delta=f"{scaled_metrics['median'] - base_metrics['median']:+.4f}",
        )
        m3.metric(
            "P90",
            f"{scaled_metrics['p90']:.4f}",
            delta=f"{scaled_metrics['p90'] - base_metrics['p90']:+.4f}",
        )
        m4.metric(
            "P95",
            f"{scaled_metrics['p95']:.4f}",
            delta=f"{scaled_metrics['p95'] - base_metrics['p95']:+.4f}",
        )
        m5.metric(
            "P99",
            f"{scaled_metrics['p99']:.4f}",
            delta=f"{scaled_metrics['p99'] - base_metrics['p99']:+.4f}",
        )

        st.markdown("**Baseline (all factors = 1.0):**")
        st.write(
            {
                "MAE": round(base_metrics["mae"], 4),
                "Median": round(base_metrics["median"], 4),
                "P90": round(base_metrics["p90"], 4),
                "P95": round(base_metrics["p95"], 4),
                "P99": round(base_metrics["p99"], 4),
            }
        )

        # st.markdown("**Error Distribution (CDF)**")
        # base_sorted = np.sort(abs_errors)
        # scaled_sorted = np.sort(scaled_abs_errors)
        # cdf_base = np.arange(1, len(base_sorted) + 1, dtype=np.float64) / len(base_sorted)
        # cdf_scaled = np.arange(1, len(scaled_sorted) + 1, dtype=np.float64) / len(scaled_sorted)
        # cdf_df = pd.DataFrame(
        #     {
        #         "abs_error": base_sorted,
        #         "cdf_baseline": cdf_base,
        #         "cdf_scaled": cdf_scaled,
        #     }
        # )
        # st.line_chart(
        #     cdf_df,
        #     x="abs_error",
        #     y=["cdf_baseline", "cdf_scaled"],
        #     height=280,
        #     width="stretch",
        # )

        summary_df = pd.DataFrame(
            {
                "percentile_bin": [f"P{pct_edges[i]:.0f}-P{pct_edges[i + 1]:.0f}" for i in range(5)],
                "left_edge": [float(edges[i]) for i in range(5)],
                "right_edge": [float(edges[i + 1]) for i in range(5)],
                "count": [int(np.sum(bin_ids == i)) for i in range(5)],
                "scale": [float(factors[i]) for i in range(5)],
            }
        )
        st.markdown("**Bin Summary**")
        st.dataframe(summary_df, width="stretch", hide_index=True)

        if st.button("Save Result (.npz)", type="primary"):
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = CACHE_DIR / f"scaled_abs_errors_{ts}.npz"
            latest_path = CACHE_DIR / "latest_scaled_abs_errors.npz"

            save_payload = {
                "abs_errors": abs_errors,
                "scaled_abs_errors": scaled_abs_errors,
                "factors": factors,
                "bin_ids": bin_ids,
                "percentile_edges": pct_edges,
                "value_edges": edges,
                "base_mae": np.array([base_metrics["mae"]]),
                "base_median": np.array([base_metrics["median"]]),
                "base_p90": np.array([base_metrics["p90"]]),
                "base_p95": np.array([base_metrics["p95"]]),
                "base_p99": np.array([base_metrics["p99"]]),
                "scaled_mae": np.array([scaled_metrics["mae"]]),
                "scaled_median": np.array([scaled_metrics["median"]]),
                "scaled_p90": np.array([scaled_metrics["p90"]]),
                "scaled_p95": np.array([scaled_metrics["p95"]]),
                "scaled_p99": np.array([scaled_metrics["p99"]]),
            }

            np.savez(
                out_path,
                **save_payload,
            )
            np.savez(
                latest_path,
                **save_payload,
            )
            st.success(f"Saved: {out_path}")
            st.info(f"Updated latest file: {latest_path}")


if __name__ == "__main__":
    if not is_streamlit_runtime():
        print("This is a Streamlit app. Please run:")
        print("  streamlit run experiments/interactive_percentile_scaling_tool.py")
        sys.exit(0)
    main()
