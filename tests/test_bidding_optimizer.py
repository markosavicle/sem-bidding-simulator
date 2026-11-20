"""Bayesian shrinkage, EV bid formula, rules, landscape fit and the simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import bidding_optimizer as bo
from src import data_generator as dg


def test_beta_prior_recovers_simulated_population():
    rng = np.random.default_rng(0)
    true_a, true_b = 4.0, 196.0  # mean 2%, kappa 200
    p = rng.beta(true_a, true_b, 400)
    clicks = rng.integers(200, 3_000, 400)
    conv = rng.binomial(clicks, p)
    for method in ("mle", "moments"):
        a, b = bo.fit_beta_prior(conv, clicks, method=method)
        assert a / (a + b) == pytest.approx(0.02, rel=0.08)
        assert 100 < a + b < 400, method


def test_shrinkage_pulls_low_volume_keywords_harder():
    a, b = 2.0, 98.0  # prior mean 2%
    low = bo.shrunk_cvr(1, 10, a, b)  # raw 10%
    high = bo.shrunk_cvr(100, 1_000, a, b)  # raw 10%
    assert low == pytest.approx(3 / 110)
    assert abs(low - 0.02) < abs(high - 0.02)  # low volume stays near the prior
    assert high == pytest.approx(102 / 1_100)
    assert bo.shrunk_cvr(0, 40, a, b) > 0  # zero conversions != zero CVR


def test_ev_bid_formula():
    hist = pd.DataFrame({
        "date": ["2025-01-01"] * 3, "keyword_id": [1, 2, 3], "campaign_id": [1, 1, 1],
        "max_cpc": [1.0] * 3, "impressions": [1e4] * 3, "clicks": [1_000.0, 1_000.0, 1_000.0],
        "cost": [500.0] * 3, "conversions": [20.0, 30.0, 10.0], "conv_value": [2_000.0, 3_000.0, 1_000.0],
    })
    tgt = bo.campaign_target_cpa(hist, target_roas=4.0)
    assert tgt.loc[1] == pytest.approx(100.0 / 4.0)  # AOV 100 / ROAS 4
    raw = bo.ExpectedValueBidding(shrink=False).propose(hist, np.ones(3), np.array([1, 2, 3]), tgt)
    np.testing.assert_allclose(raw, [0.02 * 25, 0.03 * 25, 0.01 * 25])
    shrunk = bo.ExpectedValueBidding(shrink=True).propose(hist, np.ones(3), np.array([1, 2, 3]), tgt)
    assert np.all(np.abs(shrunk - 0.5) <= np.abs(raw - 0.5) + 1e-12)  # pulled toward the campaign mean


def test_rule_based_actions():
    hist = pd.DataFrame({
        "date": ["2025-01-01"] * 4, "keyword_id": [1, 2, 3, 4], "campaign_id": [1] * 4, "max_cpc": [1.0] * 4,
        "impressions": [1e3] * 4, "clicks": [50.0, 50.0, 50.0, 5.0], "cost": [50.0, 200.0, 10.0, 5.0],
        "conversions": [0.0, 1.0, 1.0, 0.0], "conv_value": [0.0, 100.0, 100.0, 0.0],
    })
    tgt = pd.Series({1: 50.0})
    bids = bo.RuleBasedBidding().propose(hist, np.ones(4), np.array([1, 2, 3, 4]), tgt)
    np.testing.assert_allclose(bids, [0.70, 0.85, 1.15, 1.0])  # cut, down, up, not enough data


def test_bid_landscape_recovers_curve():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.3, 3.0, 30)
    clicks = 200 * (1 - np.exp(-0.9 * b)) * rng.normal(1, 0.03, 30)
    cpc = 1.2 * (1 - np.exp(-0.6 * b))
    weekly = pd.DataFrame({"keyword_id": 1, "max_cpc": b, "clicks": clicks, "cost": clicks * cpc})
    land = bo.BidLandscape.fit(weekly, 1, pcvr=0.02, aov=300)
    assert land.a == pytest.approx(200, rel=0.08)
    assert land.c == pytest.approx(0.9, rel=0.15)
    fc = land.forecast(np.linspace(0.1, 3, 50))
    assert np.all(np.diff(fc.clicks) > 0) and np.all(np.diff(np.diff(fc.clicks)) < 0)  # concave
    assert (fc.avg_cpc <= fc.bid + 1e-12).all()
    best = land.best_bid(margin=0.35)
    assert land.bid_min_observed * 0.1 < best <= 1.5 * land.bid_max_observed


@pytest.fixture(scope="module")
def sim_setup(small_tables):
    world = dg.SemWorld.build(seed=123, keyword_scale=0.5)
    h = small_tables["fact_keyword_hourly"]
    hist = h.groupby(["date", "keyword_id", "campaign_id"], as_index=False).agg(
        max_cpc=("max_cpc", "mean"), impressions=("impressions", "sum"), clicks=("clicks", "sum"),
        cost=("cost", "sum"), conversions=("conversions", "sum"), conv_value=("conv_value", "sum"))
    return world, hist


def test_world_rebuild_matches_stored_truth(small_tables, sim_setup):
    world, _ = sim_setup
    stored = small_tables["truth_keyword"].set_index("keyword_id")
    np.testing.assert_allclose(world.keywords.set_index("keyword_id").true_cvr, stored.true_cvr)


def test_shrinkage_reduces_cvr_estimation_error(sim_setup):
    world, hist = sim_setup
    err = bo.cvr_estimation_error(hist, world).set_index("volume_bucket")
    assert err.loc["all keywords", "mae_shrunk"] < err.loc["all keywords", "mae_raw"]


def test_ev_bidding_beats_rules_in_forward_simulation(sim_setup):
    world, hist = sim_setup
    target = 1 / 0.35
    res = bo.compare_strategies(world, hist, bo.default_strategies(world, target), target_roas=target,
                                n_weeks=3, seeds=(0, 1, 2))
    s = res.summary.set_index("strategy")
    assert s.loc["ev_shrinkage", "roas"] > s.loc["rule_based", "roas"]
    assert s.loc["ev_shrinkage", "profit"] > s.loc["rule_based", "profit"]
    assert set(res.final_bids.columns) >= {"keyword_id", "last_historical_bid", "ev_shrinkage"}
    assert (res.final_bids.drop(columns="keyword_id") >= bo.MIN_BID - 1e-9).all().all()
