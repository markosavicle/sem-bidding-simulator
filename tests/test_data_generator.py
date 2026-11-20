"""Sanity and realism checks for the synthetic SEM world."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_generator as dg


def test_auction_response_bounds_and_gsp_pricing():
    bids = np.linspace(0.01, 10, 500)
    share, mult, cpc = dg.auction_response(bids, 7, 7.0)
    assert np.all((share >= 0) & (share <= 1))
    assert np.all(cpc <= bids + 1e-12)  # second price: never pay more than the bid
    assert np.all(np.diff(share) >= 0) and np.all(np.diff(cpc) >= -1e-12)
    assert np.all((mult >= 0.55) & (mult <= 1.0))
    # CPC saturates at E[competitor rank] / QS + 0.01 = Gamma(1+1/a) + 0.01 for C = QS = 7
    from scipy.special import gamma

    assert cpc[-1] == pytest.approx(gamma(1 + 1 / 1.6) + 0.01, rel=1e-3)


def test_higher_quality_score_lowers_cpc_for_same_position():
    s1, _, c1 = dg.auction_response(1.0, 5, 7.0)
    # QS 10 reaches the same ad rank with half the bid -> same share, lower CPC
    s2, _, c2 = dg.auction_response(0.5, 10, 7.0)
    assert s1 == pytest.approx(s2)
    assert c2 < c1


def test_diminishing_returns_on_bid():
    bids = np.linspace(0.1, 6, 60)
    share, mult, cpc = dg.auction_response(bids, 7, 7.0)
    clicks = share * mult  # proportional to expected clicks
    cost = clicks * cpc
    marginal_clicks = np.diff(clicks) / np.diff(bids)
    marginal_cpc = np.diff(cost) / np.diff(clicks).clip(min=1e-12)
    upper = bids[1:] > 1.0  # past the inflection point of the S-curve
    assert np.all(np.diff(marginal_clicks[upper]) <= 1e-9)  # each extra euro buys fewer clicks
    assert marginal_cpc[upper][-5:].mean() > marginal_cpc[upper][:5].mean()  # ...at a higher marginal price


def test_world_is_deterministic():
    a, b = dg.SemWorld.build(seed=5), dg.SemWorld.build(seed=5)
    pd.testing.assert_frame_equal(a.keywords, b.keywords)
    pd.testing.assert_frame_equal(a.terms, b.terms)


def test_schema_and_row_counts(small_tables):
    t = small_tables
    assert set(t) >= {"dim_campaign", "dim_keyword", "fact_keyword_hourly", "fact_search_term_daily",
                      "truth_keyword", "meta"}
    hourly = t["fact_keyword_hourly"]
    n_kw = len(t["dim_keyword"])
    assert len(hourly) == 42 * n_kw * 24 * 3  # dense grid: every keyword x day x hour x device
    assert set(hourly.device) == {"desktop", "mobile", "tablet"}
    assert hourly.hour.between(0, 23).all()


def test_funnel_consistency(small_tables):
    h = small_tables["fact_keyword_hourly"]
    assert (h.clicks <= h.impressions).all()
    assert (h.conversions <= h.clicks).all()
    assert (h.cost >= 0).all() and (h.conv_value >= 0).all()
    assert ((h.conversions == 0) == (h.conv_value == 0)).all()
    # never pay more than max CPC per click (allow cent rounding)
    paid = h[h.clicks > 0]
    assert (paid.cost / paid.clicks <= paid.max_cpc + 0.01).all()


def test_search_terms_roll_up_to_keywords(small_tables):
    h = small_tables["fact_keyword_hourly"].groupby(["date", "keyword_id"])[["impressions", "clicks", "conversions"]].sum()
    s = small_tables["fact_search_term_daily"].groupby(["date", "keyword_id"])[["impressions", "clicks", "conversions"]].sum()
    pd.testing.assert_frame_equal(h.sort_index(), s.sort_index(), check_dtype=False)


def test_realistic_rates(small_tables):
    h = small_tables["fact_keyword_hourly"]
    ctr = h.clicks.sum() / h.impressions.sum()
    cvr = h.conversions.sum() / h.clicks.sum()
    assert 0.01 < ctr < 0.15
    assert 0.005 < cvr < 0.10


def test_seasonality_and_device_effects(small_tables):
    h = small_tables["fact_keyword_hourly"]
    by_hour = h.groupby("hour").impressions.sum()
    assert by_hour.loc[[11, 20, 21]].mean() > 3 * by_hour.loc[[2, 3, 4]].mean()  # night trough
    dev = h.groupby("device")[["clicks", "conversions"]].sum()
    cvr = dev.conversions / dev.clicks
    assert cvr["desktop"] > cvr["mobile"]


def test_junk_search_terms_convert_worse(small_tables):
    s = small_tables["fact_search_term_daily"].merge(
        small_tables["truth_search_term"][["term_id", "is_junk"]], on="term_id"
    )
    g = s.groupby("is_junk")[["clicks", "conversions"]].sum()
    cvr = g.conversions / g.clicks
    assert cvr[True] < 0.3 * cvr[False]


def test_promo_window_lifts_volume():
    dates = pd.date_range("2025-11-17", "2025-12-05")
    eff = dg.day_effects(dates)
    bf = dates.get_loc(pd.Timestamp("2025-11-28"))
    before = dates.get_loc(pd.Timestamp("2025-11-17"))
    assert eff["volume"][bf] > 1.5 * eff["volume"][before]
    assert eff["cvr"][bf] > eff["cvr"][before]
