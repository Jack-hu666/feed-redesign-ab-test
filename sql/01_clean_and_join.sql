-- =============================================================
-- 阶段2 · 数据准备：清洗 + 多表关联
-- 口径：先对齐再取数。清洗规则如下，写清楚是为了和业务对齐：
--   1) 去重：同一 event_id 出现多条（埋点重放），用 ROW_NUMBER 保留最早一条
--   2) 异常停留时长：duration_sec < 0 或 > 36000（>10 小时）视为埋点异常，剔除
--   3) 异常加载耗时：CAST(load_time_ms AS REAL) > 100000（100 秒）视为异常，剔除
--   4) 城市为空的用户：维度缺失，剔除（占比极小，不影响实验）
-- =============================================================

WITH dedup_event AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_time) AS rn
  FROM event_log
),
clean_event AS (
  SELECT event_id, user_id, session_id, event_time, event_type,
         duration_sec, is_load_fail, load_time_ms
  FROM dedup_event
  WHERE rn = 1
    AND NOT (event_type = 'session_end' AND (duration_sec < 0 OR duration_sec > 36000))
    AND NOT (event_type = 'expose' AND CAST(load_time_ms AS REAL) > 100000)
),
-- 关联用户基础信息，拿到分组 / 活跃度 / 设备 / 城市
clean_joined AS (
  SELECT e.event_id, e.user_id, e.session_id, e.event_time, e.event_type,
         e.duration_sec, e.is_load_fail, e.load_time_ms,
         u."group", u.activity_level, u.device, u.city
  FROM clean_event e
  JOIN user_info u ON e.user_id = u.user_id
  WHERE u.city IS NOT NULL
)
SELECT * FROM clean_joined;
