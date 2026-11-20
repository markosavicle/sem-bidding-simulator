-- =============================================================================
-- 06_hour_device_bid_modifiers.sql
-- Hour-of-day x device performance index -> suggested bid modifiers.
--
-- value per click (VPC) is the quantity a bid should track:
--     index = VPC(hour, device) / VPC(account)
-- The suggested modifier is the index clamped to [0.5, 1.5] (CASE instead of
-- GREATEST/LEAST keeps it portable to SQLite).
-- =============================================================================
WITH account AS (
    SELECT SUM(conv_value) / NULLIF(SUM(clicks), 0) AS account_vpc
    FROM fact_keyword_hourly
),
cells AS (
    SELECT
        f.device,
        f.hour,
        SUM(f.clicks)      AS clicks,
        SUM(f.cost)        AS cost,
        SUM(f.conversions) AS conversions,
        SUM(f.conv_value)  AS conv_value
    FROM fact_keyword_hourly AS f
    GROUP BY f.device, f.hour
),
indexed AS (
    SELECT
        c.*,
        (c.conv_value / NULLIF(c.clicks, 0)) / a.account_vpc AS vpc_index
    FROM cells AS c
    CROSS JOIN account AS a
)
SELECT
    device,
    hour,
    clicks,
    ROUND(cost, 2)                                     AS cost,
    conversions,
    ROUND(conversions * 1.0 / NULLIF(clicks, 0), 4)    AS cvr,
    ROUND(conv_value / NULLIF(cost, 0), 3)             AS roas,
    ROUND(vpc_index, 3)                                AS vpc_index,
    ROUND(CASE WHEN vpc_index < 0.5 THEN 0.5
               WHEN vpc_index > 1.5 THEN 1.5
               ELSE vpc_index END, 2)                  AS suggested_bid_modifier
FROM indexed
ORDER BY device, hour;
