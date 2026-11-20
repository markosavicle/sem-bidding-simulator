-- =============================================================================
-- 05_wasted_spend_search_terms.sql
-- Wasted-spend detection on low-converting search terms -> negative keyword
-- candidates.
--
-- A search term is flagged when it has enough evidence (cost >= min_cost) and
--   * ZERO_CONV : spent >= min_cost (well above its campaign CPA share) and never
--                 converted -> the whole cost is waste, or
--   * HIGH_CPA  : CPA > cpa_multiple x campaign CPA -> waste = the cost above
--                 what the campaign normally pays for those conversions.
-- Output is ranked by wasted cost with a running cumulative share, so the
-- top N rows are the fastest wins ("fix these 10 terms -> recover 60%").
-- =============================================================================
WITH params AS (
    SELECT
        25.0 AS min_cost,          -- EUR of evidence before judging a term
        3.0  AS cpa_multiple       -- "much worse than campaign average"
),
campaign_cpa AS (
    SELECT
        campaign_id,
        SUM(cost) / NULLIF(SUM(conversions), 0) AS campaign_cpa
    FROM fact_search_term_daily
    GROUP BY campaign_id
),
term_perf AS (
    SELECT
        s.campaign_id,
        s.keyword_id,
        s.search_term,
        SUM(s.impressions) AS impressions,
        SUM(s.clicks)      AS clicks,
        SUM(s.cost)        AS cost,
        SUM(s.conversions) AS conversions,
        SUM(s.conv_value)  AS conv_value
    FROM fact_search_term_daily AS s
    GROUP BY s.campaign_id, s.keyword_id, s.search_term
),
flagged AS (
    SELECT
        t.*,
        cc.campaign_cpa,
        CASE
            WHEN t.conversions = 0 THEN 'ZERO_CONV'
            WHEN t.cost / t.conversions > p.cpa_multiple * cc.campaign_cpa THEN 'HIGH_CPA'
        END AS waste_flag,
        CASE
            WHEN t.conversions = 0 THEN t.cost
            WHEN t.cost / t.conversions > p.cpa_multiple * cc.campaign_cpa
                THEN t.cost - t.conversions * cc.campaign_cpa
            ELSE 0
        END AS wasted_cost
    FROM term_perf AS t
    JOIN campaign_cpa AS cc ON cc.campaign_id = t.campaign_id
    CROSS JOIN params AS p
    WHERE t.cost >= p.min_cost
)
SELECT
    RANK() OVER (ORDER BY f.wasted_cost DESC)                    AS waste_rank,
    c.campaign_name,
    k.keyword_text,
    k.match_type,
    f.search_term,
    f.waste_flag,
    f.clicks,
    ROUND(f.cost, 2)                                             AS cost,
    f.conversions,
    ROUND(f.cost / NULLIF(f.conversions, 0), 2)                  AS term_cpa,
    ROUND(f.campaign_cpa, 2)                                     AS campaign_cpa,
    ROUND(f.wasted_cost, 2)                                      AS wasted_cost,
    ROUND(SUM(f.wasted_cost) OVER (ORDER BY f.wasted_cost DESC, f.search_term
                                   ROWS UNBOUNDED PRECEDING)
          / SUM(f.wasted_cost) OVER (), 4)                       AS cumulative_waste_share
FROM flagged AS f
JOIN dim_campaign AS c ON c.campaign_id = f.campaign_id
JOIN dim_keyword  AS k ON k.keyword_id  = f.keyword_id
WHERE f.waste_flag IS NOT NULL
ORDER BY f.wasted_cost DESC, f.search_term;
