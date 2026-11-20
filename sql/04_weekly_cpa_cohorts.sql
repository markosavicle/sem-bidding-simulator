-- =============================================================================
-- 04_weekly_cpa_cohorts.sql
-- Weekly campaign cohorts: CPA / ROAS per ISO week with week-over-week deltas
-- and the rank of each week inside its campaign.
--
-- Technique
--   * `week_start` is pre-computed in the fact table -> no dialect-specific
--     date functions (DuckDB date_trunc vs SQLite strftime).
--   * LAG() for WoW deltas, RANK() for best/worst weeks, and a 4-week
--     rolling ROAS (ratio of rolling sums).
-- =============================================================================
WITH weekly AS (
    SELECT
        f.campaign_id,
        f.week_start,
        COUNT(DISTINCT f.date) AS days,
        SUM(f.cost)            AS cost,
        SUM(f.conversions)     AS conversions,
        SUM(f.conv_value)      AS conv_value
    FROM fact_keyword_hourly AS f
    GROUP BY f.campaign_id, f.week_start
),
enriched AS (
    SELECT
        w.*,
        w.cost / NULLIF(w.conversions, 0)                         AS cpa,
        w.conv_value / NULLIF(w.cost, 0)                          AS roas,
        LAG(w.cost / NULLIF(w.conversions, 0))
            OVER (PARTITION BY w.campaign_id ORDER BY w.week_start) AS prev_cpa,
        SUM(w.conv_value) OVER w4 / NULLIF(SUM(w.cost) OVER w4, 0) AS rolling_roas_4w,
        RANK() OVER (PARTITION BY w.campaign_id
                     ORDER BY w.conv_value / NULLIF(w.cost, 0) DESC) AS roas_rank_in_campaign
    FROM weekly AS w
    WINDOW w4 AS (PARTITION BY w.campaign_id ORDER BY w.week_start
                  ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
)
SELECT
    c.campaign_name,
    e.week_start,
    e.days,
    ROUND(e.cost, 2)                                   AS cost,
    e.conversions,
    ROUND(e.conv_value, 2)                             AS conv_value,
    ROUND(e.cpa, 2)                                    AS cpa,
    ROUND(e.roas, 3)                                   AS roas,
    ROUND((e.cpa - e.prev_cpa) / NULLIF(e.prev_cpa, 0), 4) AS cpa_wow_change,
    ROUND(e.rolling_roas_4w, 3)                        AS rolling_roas_4w,
    e.roas_rank_in_campaign
FROM enriched AS e
JOIN dim_campaign AS c ON c.campaign_id = e.campaign_id
ORDER BY c.campaign_name, e.week_start;
