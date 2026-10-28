"""模拟数据生成：用户基础信息 + 埋点日志，写进 SQLite。

固定种子 seed=42，可复现。真实效应（代码注释里说明，README 不剧透）：
- CTR：control 12% → treatment 提升约 +1.35pp（高活 +2.0pp / 中活 +1.5pp / 低活 +0.5pp）
- 停留时长：control 45s 基准 → treatment +3s
- 加载失败率：两组都是 1.0%（护栏指标，不允许变差）

埋点里故意留了几类数据质量问题，给后面 SQL 清洗用：
- 约 0.3% 重复曝光事件
- 部分 session_end 的 duration_sec 异常（负数 / 超大值）
- 部分 expose 的 load_time_ms 为空或超大
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ab_test.db"

N_PER_GROUP = 17200          # 略高于样本量下限 17168
N_DAYS = 7                   # 实验周期

ACTIVITY = {"高": 0.30, "中": 0.40, "低": 0.30}
SESSIONS = {"高": 8, "中": 5, "低": 3}        # 每人会话数（近似）
EXPOSES = {"高": 8, "中": 6, "低": 4}         # 每会话曝光数（近似）
DWELL_BASE = {"高": 60.0, "中": 45.0, "低": 30.0}
BASE_CTR = 0.12
CTR_LIFT = {"高": 0.020, "中": 0.015, "低": 0.005}  # treatment 按活跃度分层的提升
DWELL_LIFT = 3.0             # treatment 停留时长 +3s
LOAD_FAIL_RATE = 0.01

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]


def gen_user_info(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """生成用户基础信息表（含分层随机分流）。"""
    # 活跃度分层
    activity = rng.choice(list(ACTIVITY), n, p=list(ACTIVITY.values()))
    # 分层随机分流：每个活跃度层内 50/50，保证两组活跃度分布一致（规避辛普森悖论）
    group = np.empty(n, dtype=object)
    for lv in ACTIVITY:
        idx = np.where(activity == lv)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        group[idx[:half]] = "control"
        group[idx[half:]] = "treatment"

    df = pd.DataFrame({
        "user_id": [f"u{i:05d}" for i in range(1, n + 1)],
        "group": group,
        "activity_level": activity,
        "device": rng.choice(["iOS", "Android"], n),
        "city": rng.choice(CITIES, n),
        "register_days": rng.integers(1, 3650, n),
    })
    # 少量城市为空（数据质量问题）
    null_city = rng.random(n) < 0.02
    df.loc[null_city, "city"] = None
    return df


def gen_event_log(rng: np.random.Generator, users: pd.DataFrame) -> pd.DataFrame:
    """生成埋点日志：曝光 / 点击 / 会话结束 三类事件。"""
    start = pd.Timestamp("2026-10-01")
    rows = []
    session_no = 0
    for u in users.itertuples():
        n_sessions = max(1, int(rng.poisson(SESSIONS[u.activity_level])))
        p_click = BASE_CTR + (CTR_LIFT[u.activity_level] if u.group == "treatment" else 0.0)
        dwell_base = DWELL_BASE[u.activity_level] + (DWELL_LIFT if u.group == "treatment" else 0.0)

        for _ in range(n_sessions):
            session_no += 1
            sid = f"s{session_no:07d}"
            n_exp = max(1, int(rng.poisson(EXPOSES[u.activity_level])))
            day = rng.integers(0, N_DAYS)
            base_time = start + pd.Timedelta(days=day) + pd.Timedelta(seconds=int(rng.integers(0, 86400)))

            # 曝光事件
            for e in range(n_exp):
                t = base_time + pd.Timedelta(seconds=int(rng.integers(0, 300)))
                load_fail = int(rng.random() < LOAD_FAIL_RATE)
                load_time = int(rng.normal(120, 40)) if load_fail == 0 else int(rng.normal(800, 300))
                rows.append([f"e{session_no:07d}_{e}", u.user_id, sid, t, "expose",
                             None, load_fail, load_time])
                # 点击事件（按 CTR）
                if rng.random() < p_click:
                    rows.append([f"c{session_no:07d}_{e}", u.user_id, sid,
                                 t + pd.Timedelta(seconds=int(rng.integers(1, 60))),
                                 "click", None, 0, None])
            # 会话结束事件（带停留时长）
            duration = max(0.5, rng.normal(dwell_base, dwell_base * 0.3))
            rows.append([f"x{session_no:07d}", u.user_id, sid,
                         base_time + pd.Timedelta(seconds=int(dwell_base * 2)),
                         "session_end", round(duration, 1), 0, None])

    df = pd.DataFrame(rows, columns=[
        "event_id", "user_id", "session_id", "event_time", "event_type",
        "duration_sec", "is_load_fail", "load_time_ms",
    ])

    # ---- 注入数据质量问题 ----
    # 0.3% 重复曝光事件
    dup_idx = rng.choice(len(df), int(len(df) * 0.003), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    # 部分 session_end 时长异常（负数 / 超大）
    end_mask = df["event_type"] == "session_end"
    bad = rng.random(end_mask.sum()) < 0.01
    end_idx = df.index[end_mask][bad]
    df.loc[end_idx, "duration_sec"] = rng.choice([-1.0, 99999.0], len(end_idx))
    # 部分 expose 加载耗时空 / 超大
    exp_mask = df["event_type"] == "expose"
    bad2 = rng.random(exp_mask.sum()) < 0.01
    exp_idx = df.index[exp_mask][bad2]
    df["load_time_ms"] = df["load_time_ms"].astype(float)
    df.loc[exp_idx, "load_time_ms"] = rng.choice([np.nan, 999999.0], len(exp_idx))
    return df


def main():
    rng = np.random.default_rng(42)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    users = gen_user_info(rng, N_PER_GROUP * 2)
    events = gen_event_log(rng, users)

    con = sqlite3.connect(DB_PATH)
    users.to_sql("user_info", con, if_exists="replace", index=False)
    # 显式指定数值列类型，避免 NaN 导致被存成 TEXT
    events.to_sql("event_log", con, if_exists="replace", index=False,
                  dtype={"duration_sec": "REAL", "load_time_ms": "REAL",
                         "is_load_fail": "INTEGER"})
    con.close()

    print(f"用户数：{len(users)}（control={sum(users.group=='control')}, treatment={sum(users.group=='treatment')}）")
    print(f"事件数：{len(events)}")
    print(f"事件类型分布：\n{events['event_type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
