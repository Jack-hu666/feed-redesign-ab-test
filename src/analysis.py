"""阶段3：平衡性检验 + 假设检验（核心）。

先做两件「结论可信」的校验，再做显著性检验：
  1) 分流均匀性 SRM：两组人数是否符合 50/50（卡方）
  2) 样本平衡性：activity_level / device / city 在两组分布是否一致（卡方）
然后：
  - CTR（比例）→ 卡方检验（等价于两比例 z 检验，z² = χ²）
  - 停留时长（连续）→ 独立样本 t 检验
  - 跳出率（比例）→ 卡方检验
  - 加载失败率（护栏）→ 卡方检验，验证「不变差」

每个检验都写清：原假设 H0、备择假设 H1、单/双侧、α、p 值、95% 置信区间、效应量。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ab_test.db"

# 用户级指标 SQL（与 sql/02_user_metrics.sql 一致）
USER_METRICS_SQL = """
WITH dedup_event AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_time) AS rn FROM event_log
),
clean_event AS (
  SELECT * FROM dedup_event
  WHERE rn = 1
    AND NOT (event_type = 'session_end' AND (duration_sec < 0 OR duration_sec > 36000))
    AND NOT (event_type = 'expose' AND CAST(load_time_ms AS REAL) > 100000)
),
clean_joined AS (
  SELECT e.*, u."group", u.activity_level, u.device, u.city
  FROM clean_event e JOIN user_info u ON e.user_id = u.user_id
  WHERE u.city IS NOT NULL
)
SELECT
  user_id,
  MAX("group") AS grp,
  MAX(activity_level) AS activity_level,
  MAX(device) AS device,
  SUM(CASE WHEN event_type='expose' THEN 1 ELSE 0 END) AS expose_cnt,
  SUM(CASE WHEN event_type='click'  THEN 1 ELSE 0 END) AS click_cnt,
  ROUND(SUM(CASE WHEN event_type='click' THEN 1 ELSE 0 END)*1.0 /
        NULLIF(SUM(CASE WHEN event_type='expose' THEN 1 ELSE 0 END), 0), 6) AS ctr,
  AVG(CASE WHEN event_type='session_end' THEN duration_sec END) AS avg_dwell,
  SUM(CASE WHEN event_type='expose' AND is_load_fail=1 THEN 1 ELSE 0 END)*1.0 /
    NULLIF(SUM(CASE WHEN event_type='expose' THEN 1 ELSE 0 END), 0) AS load_fail_rate
FROM clean_joined
GROUP BY user_id
"""


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(DB_PATH)
    metrics = pd.read_sql_query(USER_METRICS_SQL, con)
    users = pd.read_sql_query("SELECT * FROM user_info", con)
    con.close()
    return metrics, users


def check_srm(users: pd.DataFrame) -> float:
    """分流均匀性：卡方拟合优度，H0=两组人数 50/50。"""
    n_c = (users["group"] == "control").sum()
    n_t = (users["group"] == "treatment").sum()
    n = len(users)
    chi2, p = stats.chisquare([n_c, n_t], [n / 2, n / 2])
    print(f"[SRM] control={n_c}, treatment={n_t}, 卡方={chi2:.3f}, p={p:.4f} -> "
          f"{'PASS' if p > 0.05 else 'FAIL'}")
    return p


def check_balance(users: pd.DataFrame) -> None:
    """样本平衡性：分流前维度（活跃度/设备/城市）在两组分布是否一致。"""
    print("[平衡性] 各维度卡方检验（p>0.05 视为平衡）：")
    for col in ["activity_level", "device", "city"]:
        ct = pd.crosstab(users["group"], users[col])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        print(f"  {col}: χ²={chi2:.2f}, p={p:.4f} -> {'PASS' if p > 0.05 else 'FAIL'}")


def proportion_test(n1: int, x1: int, n2: int, x2: int, label: str):
    """比例指标检验（卡方 + 两比例差置信区间）。

    H0: 两组比例相等；H1: 两组比例不等（双侧，α=0.05）。
    """
    p1, p2 = x1 / n1, x2 / n2
    # 卡方检验（2x2）
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    # 两比例差的 95% 置信区间（非合并标准误）
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    z = stats.norm.ppf(0.975)
    ci = (diff - z * se, diff + z * se)
    print(f"[{label}] {p1:.4%} vs {p2:.4%} | 差={diff:+.4%} | "
          f"95%CI=[{ci[0]:+.4%}, {ci[1]:+.4%}] | χ²={chi2:.2f}, p={p:.4f} | "
          f"{'显著' if p < 0.05 else '不显著'}")
    return p, diff, ci


def t_test_continuous(ctrl: pd.Series, trt: pd.Series, label: str):
    """连续指标检验（独立样本 t 检验，Welch 不假设方差齐）。

    H0: 两组均值相等；H1: 不等（双侧，α=0.05）。
    """
    m1, m2 = ctrl.mean(), trt.mean()
    t, p = stats.ttest_ind(ctrl, trt, equal_var=False)
    # 均值差 95% CI
    diff = m1 - m2
    se = np.sqrt(ctrl.var(ddof=1) / len(ctrl) + trt.var(ddof=1) / len(trt))
    dof = (ctrl.var(ddof=1) / len(ctrl) + trt.var(ddof=1) / len(trt)) ** 2 / (
        (ctrl.var(ddof=1) / len(ctrl)) ** 2 / (len(ctrl) - 1) +
        (trt.var(ddof=1) / len(trt)) ** 2 / (len(trt) - 1))
    ci = (diff - stats.t.ppf(0.975, dof) * se, diff + stats.t.ppf(0.975, dof) * se)
    print(f"[{label}] {m1:.2f} vs {m2:.2f} | 差={diff:+.2f} | "
          f"95%CI=[{ci[0]:+.2f}, {ci[1]:+.2f}] | t={t:.2f}, p={p:.4f} | "
          f"{'显著' if p < 0.05 else '不显著'}")
    return p, diff, ci


def main():
    metrics, users = load()

    print("=" * 70)
    print("阶段3 · 假设检验")
    print("=" * 70)
    print("\n--- 数据质量与分流校验 ---")
    check_srm(users)
    check_balance(users)

    ctrl = metrics[metrics["grp"] == "control"]
    trt = metrics[metrics["grp"] == "treatment"]

    print("\n--- 指标均值对比 ---")
    print(f"用户数：control={len(ctrl)}, treatment={len(trt)}")

    # CTR：卡方检验（点击/非点击）
    c_clicks = ctrl["click_cnt"].sum()
    c_expose = ctrl["expose_cnt"].sum()
    t_clicks = trt["click_cnt"].sum()
    t_expose = trt["expose_cnt"].sum()
    print("\n--- 核心指标：CTR（卡方检验）---")
    proportion_test(t_expose, t_clicks, c_expose, c_clicks, "CTR")

    # 停留时长：t 检验
    print("\n--- 次级指标：平均停留时长（t 检验）---")
    t_test_continuous(trt["avg_dwell"].dropna(), ctrl["avg_dwell"].dropna(), "停留时长(秒)")

    # 跳出率：会话级（无点击会话占比）
    print("\n--- 次级指标：跳出率（卡方检验）---")
    con = sqlite3.connect(DB_PATH)
    sess = pd.read_sql_query(
        """SELECT session_id, MAX("group") AS grp,
                  SUM(CASE WHEN event_type='click' THEN 1 ELSE 0 END) AS click_cnt
           FROM event_log e JOIN user_info u ON e.user_id=u.user_id
           GROUP BY session_id""", con)
    con.close()
    sess["bounced"] = (sess["click_cnt"] == 0).astype(int)
    c_b = sess[sess["grp"] == "control"]["bounced"]
    t_b = sess[sess["grp"] == "treatment"]["bounced"]
    proportion_test(len(t_b), t_b.sum(), len(c_b), c_b.sum(), "跳出率")

    # 护栏指标：加载失败率（事件级计数，护栏要求「不变差」）
    print("\n--- 护栏指标：加载失败率（卡方检验，要求不变差）---")
    con = sqlite3.connect(DB_PATH)
    lf = pd.read_sql_query(
        """SELECT u."group",
                  SUM(CASE WHEN e.is_load_fail = 1 THEN 1 ELSE 0 END) AS fail_cnt,
                  COUNT(*) AS expose_cnt
           FROM event_log e JOIN user_info u ON e.user_id = u.user_id
           WHERE e.event_type = 'expose'
           GROUP BY u."group\"""", con)
    con.close()
    lf = lf.set_index("group")
    proportion_test(int(lf.loc["treatment", "expose_cnt"]), int(lf.loc["treatment", "fail_cnt"]),
                    int(lf.loc["control", "expose_cnt"]), int(lf.loc["control", "fail_cnt"]),
                    "加载失败率")

    print("\n结论：根据上述 p 值判断各指标是否显著，再综合护栏指标做上线/灰度/放弃决策。")


if __name__ == "__main__":
    main()
