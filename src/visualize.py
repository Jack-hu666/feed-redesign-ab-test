"""阶段3 · 可视化：转化率对比、置信区间误差棒、每日时序、分群对比。

- plotly：转化率柱状图、效应量森林图（CI 误差棒）、每日时序图
- matplotlib/seaborn：分群效果对比图
"""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from src.analysis import DB_PATH, load, USER_METRICS_SQL

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"

# 中文字体（matplotlib）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"


def rate_ci(n1, x1, n2, x2):
    """treatment(n1,x1) 相对 control(n2,x2) 的比例差 + 95% CI。"""
    p1, p2 = x1 / n1, x2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    z = stats.norm.ppf(0.975)
    return diff, diff - z * se, diff + z * se


def _save_plotly(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(FIGURES / name))  # 需要 kaleido
    except Exception:
        fig.write_html(str(FIGURES / name.replace(".png", ".html")))
    print(f"已生成 {name}")


def chart_conversion(ctrl_ctr, trt_ctr, ctrl_ci, trt_ci):
    """转化率对比柱状图（带误差棒）。"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["对照组"], y=[ctrl_ctr * 100], name="对照组(旧版)",
                         marker_color=BLUE, error_y=dict(type="data", array=[ctrl_ci * 100])))
    fig.add_trace(go.Bar(x=["实验组"], y=[trt_ctr * 100], name="实验组(新版)",
                         marker_color=ORANGE, error_y=dict(type="data", array=[trt_ci * 100])))
    fig.update_layout(title="首页 CTR 对比", yaxis_title="CTR（%）", height=420)
    _save_plotly(fig, "1_ctr_compare.png")


def chart_forest(effects: list[dict]):
    """效应量森林图：各指标的治疗组-对照组差 + 95% CI。"""
    names = [e["name"] for e in effects]
    fig = go.Figure()
    for i, e in enumerate(effects):
        y = len(effects) - 1 - i
        fig.add_trace(go.Scatter(
            x=[e["diff"], e["diff"]], y=[y, y], mode="lines",
            line=dict(color=BLUE, width=2), showlegend=False))
        fig.add_trace(go.Scatter(
            x=[e["ci_lo"], e["ci_hi"]], y=[y, y], mode="lines",
            line=dict(color=BLUE, width=6), showlegend=False))
        fig.add_trace(go.Scatter(
            x=[e["diff"]], y=[y], mode="markers", marker=dict(color=ORANGE, size=9),
            name=e["name"], text=[f"{e['label']}（{e['diff']:+.3f} {e['unit']}）"],
            hovertemplate="%{text}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_yaxes(tickvals=list(range(len(effects))), ticktext=[e["name"] for e in effects][::-1])
    fig.update_layout(title="各指标效应量与 95% 置信区间", xaxis_title="效应量（治疗组 − 对照组）",
                      height=420)
    _save_plotly(fig, "2_effect_forest.png")


def chart_daily():
    """每日 CTR 时序图（两组对比）。"""
    con = sqlite3.connect(DB_PATH)
    daily = pd.read_sql_query(
        """SELECT date(event_time) AS dt, u."group",
                  SUM(CASE WHEN event_type='click' THEN 1 ELSE 0 END)*1.0 /
                  NULLIF(SUM(CASE WHEN event_type='expose' THEN 1 ELSE 0 END), 0) AS ctr
           FROM event_log e JOIN user_info u ON e.user_id=u.user_id
           WHERE event_type IN ('expose','click')
           GROUP BY date(event_time), u."group" ORDER BY dt""", con)
    con.close()

    fig = go.Figure()
    for grp, color in [("control", BLUE), ("treatment", ORANGE)]:
        d = daily[daily["group"] == grp]
        fig.add_trace(go.Scatter(x=d["dt"], y=d["ctr"] * 100, mode="lines+markers",
                                 name="对照组" if grp == "control" else "实验组",
                                 line=dict(color=color, width=2)))
    fig.update_layout(title="每日 CTR 时序（7 天实验期）", xaxis_title="日期",
                      yaxis_title="CTR（%）", height=420)
    _save_plotly(fig, "3_daily_ctr.png")


def chart_segment(metrics: pd.DataFrame):
    """分群效果：按活跃度分层看 CTR 提升（规避辛普森悖论的关键）。"""
    rows = []
    for lv in ["高", "中", "低"]:
        sub = metrics[metrics["activity_level"] == lv]
        c = sub[sub["grp"] == "control"]
        t = sub[sub["grp"] == "treatment"]
        diff, lo, hi = rate_ci(t["expose_cnt"].sum(), t["click_cnt"].sum(),
                               c["expose_cnt"].sum(), c["click_cnt"].sum())
        rows.append((lv, diff * 100, lo * 100, hi * 100))
    seg = pd.DataFrame(rows, columns=["活跃度", "lift_pp", "lo", "hi"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(seg["活跃度"], seg["lift_pp"],
                yerr=[seg["lift_pp"] - seg["lo"], seg["hi"] - seg["lift_pp"]],
                fmt="o", color=BLUE, capsize=5, markersize=7)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("分群 CTR 提升（按活跃度分层）", loc="left")
    ax.set_ylabel("CTR 提升（pp）")
    for _, r in seg.iterrows():
        ax.annotate(f"{r['lift_pp']:+.2f}pp", (r["活跃度"], r["lift_pp"]),
                    textcoords="offset points", xytext=(0, 10), ha="center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "4_segment_lift.png", dpi=150)
    plt.close(fig)
    print("已生成 4_segment_lift.png")


def main():
    metrics, _ = load()
    ctrl = metrics[metrics["grp"] == "control"]
    trt = metrics[metrics["grp"] == "treatment"]

    # CTR
    c_clicks, c_exp = ctrl["click_cnt"].sum(), ctrl["expose_cnt"].sum()
    t_clicks, t_exp = trt["click_cnt"].sum(), trt["expose_cnt"].sum()
    c_ctr, t_ctr = c_clicks / c_exp, t_clicks / t_exp
    # CTR 单组比例 CI（Wilson 近似）
    z = stats.norm.ppf(0.975)
    c_ci = z * np.sqrt(c_ctr * (1 - c_ctr) / c_exp)
    t_ci = z * np.sqrt(t_ctr * (1 - t_ctr) / t_exp)

    chart_conversion(c_ctr, t_ctr, c_ci, t_ci)

    # 效应量森林图
    ctr_diff, ctr_lo, ctr_hi = rate_ci(t_exp, t_clicks, c_exp, c_clicks)
    dwell_diff, dwell_lo, dwell_hi = _dwell_effect(trt, ctrl)
    bounce_diff, bounce_lo, bounce_hi = _bounce_effect()
    fail_diff, fail_lo, fail_hi = _loadfail_effect()
    chart_forest([
        {"name": "CTR", "diff": ctr_diff * 100, "ci_lo": ctr_lo * 100, "ci_hi": ctr_hi * 100,
         "label": "CTR 提升", "unit": "pp"},
        {"name": "停留时长", "diff": dwell_diff, "ci_lo": dwell_lo, "ci_hi": dwell_hi,
         "label": "停留时长提升", "unit": "秒"},
        {"name": "跳出率", "diff": bounce_diff * 100, "ci_lo": bounce_lo * 100, "ci_hi": bounce_hi * 100,
         "label": "跳出率变化", "unit": "pp"},
        {"name": "加载失败率", "diff": fail_diff * 100, "ci_lo": fail_lo * 100, "ci_hi": fail_hi * 100,
         "label": "加载失败率变化", "unit": "pp"},
    ])

    chart_daily()
    chart_segment(metrics)


def _dwell_effect(trt, ctrl):
    t, c = trt["avg_dwell"].dropna(), ctrl["avg_dwell"].dropna()
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    dof = (t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)) ** 2 / (
        (t.var(ddof=1) / len(t)) ** 2 / (len(t) - 1) + (c.var(ddof=1) / len(c)) ** 2 / (len(c) - 1))
    z = stats.t.ppf(0.975, dof)
    return diff, diff - z * se, diff + z * se


def _bounce_effect():
    con = sqlite3.connect(DB_PATH)
    sess = pd.read_sql_query(
        """SELECT session_id, MAX(u."group") AS grp,
                  SUM(CASE WHEN event_type='click' THEN 1 ELSE 0 END) AS click_cnt
           FROM event_log e JOIN user_info u ON e.user_id=u.user_id
           GROUP BY session_id""", con)
    con.close()
    sess["bounced"] = (sess["click_cnt"] == 0).astype(int)
    t = sess[sess["grp"] == "treatment"]["bounced"]
    c = sess[sess["grp"] == "control"]["bounced"]
    return rate_ci(len(t), t.sum(), len(c), c.sum())


def _loadfail_effect():
    con = sqlite3.connect(DB_PATH)
    lf = pd.read_sql_query(
        """SELECT u."group", SUM(CASE WHEN is_load_fail=1 THEN 1 ELSE 0 END) AS fail,
                  COUNT(*) AS expose
           FROM event_log e JOIN user_info u ON e.user_id=u.user_id
           WHERE event_type='expose' GROUP BY u."group\"""", con)
    con.close()
    lf = lf.set_index("group")
    return rate_ci(int(lf.loc["treatment", "expose"]), int(lf.loc["treatment", "fail"]),
                   int(lf.loc["control", "expose"]), int(lf.loc["control", "fail"]))


if __name__ == "__main__":
    main()
