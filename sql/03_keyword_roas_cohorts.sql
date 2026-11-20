-- =============================================================================
-- 03_keyword_roas_cohorts.sql
-- Keyword ROAS quartile cohorts: how is spend distributed across good and bad
-- keywords? (Pareto view of the account.)
--
-- Technique
--   * NTILE(4) assigns keywords (with enough clicks to be judged) to ROAS
--     quartiles, Q1 = worst.
--   * The cohort summary uses ratio-of-sums per quartile.
--   * Running SUM() OVER (ORDER BY quartile) gives cumulative spend share.
-- =============================================================================
WITH params AS (
    SELECT 100 AS min_clicks          -- ignore keywords too small to judge
),
keyword_perf AS (
    SELECT
        f.keyword_id,
        SUM(f.clicks)      AS clicks,
        SUM(f.cost)        AS cost,
        SUM(f.conversions) AS conversions,
        SUM(f.conv_value)  AS conv_value
    FROM fact_keyword_hourly AS f
    GROUP BY f.keyword_id
),
ranked AS (
    SELECT
        k.*,
        NTILE(4) OVER (ORDER BY k.conv_value / NULLIF(k.cost, 0), k.keyword_id) AS roas_quartile
    FROM keyword_perf AS k
    CROSS JOIN params AS p
    WHERE k.clicks >= p.min_clicks
),
cohorts AS (
    SELECT
        roas_quartile,
        COUNT(*)          AS keywords,
        SUM(clicks)       AS clicks,
        SUM(cost)         AS cost,
        SUM(conversions)  AS conversions,
        SUM(conv_value)   AS conv_value
    FROM ranked
    GROUP BY roas_quartile
)
SELECT
    'Q' || roas_quartile                                        AS roas_quartile,
    keywords,
    clicks,
    ROUND(cost, 2)                                              AS cost,
    conversions,
    ROUND(conv_value, 2)                                        AS conv_value,
    ROUND(cost / NULLIF(conversions, 0), 2)                     AS cpa,
    ROUND(conv_value / NULLIF(cost, 0), 3)                      AS roas,
    ROUND(cost / SUM(cost) OVER (), 4)                          AS share_of_spend,
    ROUND(conv_value / SUM(conv_value) OVER (), 4)              AS share_of_value,
    ROUND(SUM(cost) OVER (ORDER BY roas_quartile
                          ROWS UNBOUNDED PRECEDING)
          / SUM(cost) OVER (), 4)                               AS cumulative_spend_share
FROM cohorts
ORDER BY roas_quartile;
