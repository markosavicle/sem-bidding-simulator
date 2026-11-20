-- =============================================================================
-- 01_campaign_kpis.sql
-- Campaign scorecard: volume, efficiency and value KPIs + share of spend.
--
-- Notes
--   * Ratios are computed as ratio-of-sums (never average-of-ratios), which is
--     how ad platforms report CTR / CPC / ROAS.
--   * `* 1.0` forces floating-point division (SQLite truncates int / int).
--   * NULLIF(x, 0) avoids division-by-zero on campaigns with no conversions.
--   * Runs unchanged on DuckDB and SQLite (>= 3.25, window functions).
-- =============================================================================
WITH campaign_totals AS (
    SELECT
        f.campaign_id,
        SUM(f.impressions)  AS impressions,
        SUM(f.clicks)       AS clicks,
        SUM(f.cost)         AS cost,
        SUM(f.conversions)  AS conversions,
        SUM(f.conv_value)   AS conv_value
    FROM fact_keyword_hourly AS f
    GROUP BY f.campaign_id
)
SELECT
    c.campaign_name,
    t.impressions,
    t.clicks,
    ROUND(t.cost, 2)                                         AS cost,
    t.conversions,
    ROUND(t.conv_value, 2)                                   AS conv_value,
    ROUND(t.clicks * 1.0 / NULLIF(t.impressions, 0), 4)      AS ctr,
    ROUND(t.cost / NULLIF(t.clicks, 0), 3)                   AS cpc,
    ROUND(t.conversions * 1.0 / NULLIF(t.clicks, 0), 4)      AS cvr,
    ROUND(t.cost / NULLIF(t.conversions, 0), 2)              AS cpa,
    ROUND(t.conv_value / NULLIF(t.cost, 0), 2)               AS roas,
    -- window over the whole result set: each campaign's share of total spend
    ROUND(t.cost / SUM(t.cost) OVER (), 4)                   AS share_of_spend
FROM campaign_totals AS t
JOIN dim_campaign    AS c ON c.campaign_id = t.campaign_id
ORDER BY t.cost DESC;
