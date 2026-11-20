-- =============================================================================
-- 02_daily_rolling_trends.sql
-- Daily campaign trend with 7-day rolling ROAS / CPA and week-over-week change.
--
-- Technique
--   * Rolling metrics are ratio-of-rolling-sums:
--         rolling_roas_7d = SUM(value) over 7 rows / SUM(cost) over 7 rows
--     Averaging 7 daily ROAS values would over-weight low-spend days.
--   * ROWS BETWEEN 6 PRECEDING AND CURRENT ROW = trailing 7 calendar days
--     (the fact table has a row for every day, so rows == days).
--   * LAG(..., 7) compares each day with the same weekday last week, which
--     removes the day-of-week seasonality from the comparison.
-- =============================================================================
WITH daily AS (
    SELECT
        f.campaign_id,
        f.date,
        SUM(f.clicks)      AS clicks,
        SUM(f.cost)        AS cost,
        SUM(f.conversions) AS conversions,
        SUM(f.conv_value)  AS conv_value
    FROM fact_keyword_hourly AS f
    GROUP BY f.campaign_id, f.date
),
rolling AS (
    SELECT
        d.*,
        SUM(d.cost) OVER w7        AS cost_7d,
        SUM(d.conv_value) OVER w7  AS value_7d,
        SUM(d.conversions) OVER w7 AS conv_7d,
        COUNT(*) OVER w7           AS days_in_window,
        LAG(d.conv_value, 7) OVER (PARTITION BY d.campaign_id ORDER BY d.date) AS value_same_day_last_week
    FROM daily AS d
    WINDOW w7 AS (PARTITION BY d.campaign_id ORDER BY d.date
                  ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
)
SELECT
    c.campaign_name,
    r.date,
    r.clicks,
    ROUND(r.cost, 2)                                       AS cost,
    r.conversions,
    ROUND(r.conv_value, 2)                                 AS conv_value,
    ROUND(r.conv_value / NULLIF(r.cost, 0), 3)             AS roas,
    -- only report the rolling metric once the window is full
    CASE WHEN r.days_in_window = 7
         THEN ROUND(r.value_7d / NULLIF(r.cost_7d, 0), 3) END AS rolling_roas_7d,
    CASE WHEN r.days_in_window = 7
         THEN ROUND(r.cost_7d / NULLIF(r.conv_7d, 0), 2) END  AS rolling_cpa_7d,
    ROUND((r.conv_value - r.value_same_day_last_week)
          / NULLIF(r.value_same_day_last_week, 0), 4)      AS value_wow_change
FROM rolling AS r
JOIN dim_campaign AS c ON c.campaign_id = r.campaign_id
ORDER BY c.campaign_name, r.date;
