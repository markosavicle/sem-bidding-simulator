"""Every SQL file runs on DuckDB and SQLite and returns the same answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sql_runner import list_queries, run_query


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            try:
                out[c] = pd.to_numeric(out[c])
            except (ValueError, TypeError):
                out[c] = out[c].astype(str)
    return out.reset_index(drop=True)


def test_all_queries_present():
    assert len(list_queries()) == 7


@pytest.mark.parametrize("name", list_queries())
def test_query_identical_on_duckdb_and_sqlite(name, small_dbs):
    d = _normalise(run_query(name, small_dbs["duckdb"]))
    s = _normalise(run_query(name, small_dbs["sqlite"]))
    assert list(d.columns) == list(s.columns)
    assert len(d) == len(s) and len(d) > 0
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]) and pd.api.types.is_numeric_dtype(s[c]):
            # ROUND() can differ by one unit in the last place on .5 ties
            np.testing.assert_allclose(d[c].astype(float), s[c].astype(float), rtol=1e-6, atol=0.011,
                                       equal_nan=True, err_msg=f"{name}.{c}")
        else:
            assert (d[c].astype(str) == s[c].astype(str)).all(), f"{name}.{c}"


def test_campaign_kpis_match_pandas(small_dbs, small_tables):
    kpi = run_query("01_campaign_kpis", small_dbs["duckdb"]).set_index("campaign_name")
    h = small_tables["fact_keyword_hourly"].merge(small_tables["dim_campaign"], on="campaign_id")
    g = h.groupby("campaign_name")[["cost", "conv_value", "clicks", "impressions"]].sum()
    np.testing.assert_allclose(kpi.loc[g.index, "roas"], (g.conv_value / g.cost).round(2), atol=0.006)
    np.testing.assert_allclose(kpi.loc[g.index, "ctr"], (g.clicks / g.impressions).round(4), atol=1e-4)
    assert kpi.share_of_spend.sum() == pytest.approx(1.0, abs=1e-3)


def test_rolling_roas_is_ratio_of_sums(small_dbs, small_tables):
    df = run_query("02_daily_rolling_trends", small_dbs["duckdb"])
    one = df[df.campaign_name == df.campaign_name.iloc[0]].reset_index(drop=True)
    assert one.rolling_roas_7d.iloc[:6].isna().all()
    expected = one.conv_value.rolling(7).sum() / one.cost.rolling(7).sum()
    np.testing.assert_allclose(one.rolling_roas_7d.iloc[6:], expected.iloc[6:], atol=0.002)


def test_wasted_spend_logic(small_dbs):
    w = run_query("05_wasted_spend_search_terms", small_dbs["duckdb"])
    assert set(w.waste_flag) <= {"ZERO_CONV", "HIGH_CPA"}
    z = w[w.waste_flag == "ZERO_CONV"]
    assert (z.conversions == 0).all() and np.allclose(z.wasted_cost, z.cost)
    hc = w[w.waste_flag == "HIGH_CPA"]
    assert (hc.term_cpa > 3 * hc.campaign_cpa - 0.01).all()
    assert (w.cost >= 25).all()
    assert w.wasted_cost.is_monotonic_decreasing
    assert w.cumulative_waste_share.iloc[-1] == pytest.approx(1.0, abs=1e-3)


def test_quartiles_cover_all_spend(small_dbs):
    q = run_query("03_keyword_roas_cohorts", small_dbs["duckdb"])
    assert list(q.roas_quartile) == ["Q1", "Q2", "Q3", "Q4"]
    assert q.roas.is_monotonic_increasing
    assert q.cumulative_spend_share.iloc[-1] == pytest.approx(1.0, abs=1e-3)


def test_bid_modifiers_clamped(small_dbs):
    m = run_query("06_hour_device_bid_modifiers", small_dbs["duckdb"])
    assert len(m) == 72
    assert m.suggested_bid_modifier.between(0.5, 1.5).all()
