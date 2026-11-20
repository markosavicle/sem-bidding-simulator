-- =============================================================================
-- 07_keyword_bid_landscape.sql
-- Weekly bid landscape per keyword: (max CPC -> clicks, cost) observations used
-- to fit diminishing-returns response curves, plus an empirical bid elasticity.
--
--   elasticity = %change(clicks) / %change(max_cpc)   (week over week)
-- Elasticity < 1 means a bid increase buys proportionally fewer clicks
-- (diminishing returns); it is NULL when the bid did not change.
-- =============================================================================
WITH weekly AS (
    SELECT
        f.keyword_id,
        f.week_start,
        -- click-weighted average bid would hide zero-click weeks; use plain avg
        AVG(f.max_cpc)      AS max_cpc,
        SUM(f.impressions)  AS impressions,
        SUM(f.clicks)       AS clicks,
        SUM(f.cost)         AS cost,
        SUM(f.conversions)  AS conversions,
        SUM(f.conv_value)   AS conv_value
    FROM fact_keyword_hourly AS f
    GROUP BY f.keyword_id, f.week_start
),
with_lag AS (
    SELECT
        w.*,
        LAG(w.max_cpc) OVER (PARTITION BY w.keyword_id ORDER BY w.week_start) AS prev_cpc,
        LAG(w.clicks)  OVER (PARTITION BY w.keyword_id ORDER BY w.week_start) AS prev_clicks
    FROM weekly AS w
)
SELECT
    k.keyword_id,
    k.keyword_text,
    k.match_type,
    l.week_start,
    ROUND(l.max_cpc, 2)                                       AS max_cpc,
    l.impressions,
    l.clicks,
    ROUND(l.cost, 2)                                          AS cost,
    ROUND(l.cost / NULLIF(l.clicks, 0), 3)                    AS avg_cpc,
    l.conversions,
    ROUND(l.conv_value, 2)                                    AS conv_value,
    ROUND(((l.clicks - l.prev_clicks) * 1.0 / NULLIF(l.prev_clicks, 0))
          / NULLIF((l.max_cpc - l.prev_cpc) / NULLIF(l.prev_cpc, 0), 0), 3) AS bid_elasticity
FROM with_lag AS l
JOIN dim_keyword AS k ON k.keyword_id = l.keyword_id
ORDER BY k.keyword_id, l.week_start;
