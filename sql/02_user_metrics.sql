-- =============================================================
-- 阶段2 · 用户级指标计算（窗口函数 rank / lag）
-- 口径：
--   CTR      = 点击事件数 / 曝光事件数（用户级）
--   停留时长  = session_end 的 duration_sec 平均
--   跳出率    = 无点击的会话数 / 总会话数
--   加载失败率 = is_load_fail=1 的曝光数 / 曝光数（护栏指标）
-- =============================================================

-- 复用清洗后的明细（同 01）
WITH dedup_event AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_time) AS rn
  FROM event_log
),
clean_event AS (
  SELECT * FROM dedup_event
  WHERE rn = 1
    AND NOT (event_type = 'session_end' AND (duration_sec < 0 OR duration_sec > 36000))
    AND NOT (event_type = 'expose' AND CAST(load_time_ms AS REAL) > 100000)
),
clean_joined AS (
  SELECT e.*, u."group", u.activity_level, u.device, u.city
  FROM clean_event e
  JOIN user_info u ON e.user_id = u.user_id
  WHERE u.city IS NOT NULL
),
-- 会话级：每个会话的点击数（用于跳出率）
session_click AS (
  SELECT session_id,
         SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS click_cnt
  FROM clean_joined
  GROUP BY session_id
),
-- 用户级指标
user_metrics AS (
  SELECT
    user_id,
    MAX(group)         AS grp,
    MAX(activity_level) AS activity_level,
    MAX(device)         AS device,
    SUM(CASE WHEN event_type = 'expose' THEN 1 ELSE 0 END) AS expose_cnt,
    SUM(CASE WHEN event_type = 'click'  THEN 1 ELSE 0 END) AS click_cnt,
    ROUND(SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) * 1.0 /
          NULLIF(SUM(CASE WHEN event_type = 'expose' THEN 1 ELSE 0 END), 0), 4) AS ctr,
    ROUND(AVG(CASE WHEN event_type = 'session_end' THEN duration_sec END), 2) AS avg_dwell,
    ROUND(SUM(CASE WHEN event_type = 'expose' AND is_load_fail = 1 THEN 1 ELSE 0 END) * 1.0 /
          NULLIF(SUM(CASE WHEN event_type = 'expose' THEN 1 ELSE 0 END), 0), 4) AS load_fail_rate
  FROM clean_joined
  GROUP BY user_id
)
-- 用 RANK 给用户按 CTR 排名（演示窗口函数）
SELECT *,
       RANK() OVER (ORDER BY ctr DESC) AS ctr_rank
FROM user_metrics;

-- =============================================================
-- 附：用 LAG 计算会话内相邻事件的间隔（演示窗口函数，也可用于异常检测）
-- =============================================================
WITH dedup_event AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_time) AS rn
  FROM event_log
),
clean_event AS (
  SELECT * FROM dedup_event
  WHERE rn = 1
),
gap AS (
  SELECT session_id, event_type, event_time,
         LAG(event_time) OVER (PARTITION BY session_id ORDER BY event_time) AS prev_time
  FROM clean_event
)
SELECT session_id, event_type, event_time, prev_time,
       ROUND((julianday(event_time) - julianday(prev_time)) * 86400, 1) AS gap_sec
FROM gap
WHERE prev_time IS NOT NULL
LIMIT 100;
