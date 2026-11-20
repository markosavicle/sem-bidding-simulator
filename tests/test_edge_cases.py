"""Edge cases found in review: zero counts, degenerate rates, extreme Beta parameters,
SQL file whitelisting and read-only connections."""

from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pytest

from src import bidding_optimizer as bo
from src import stats_engine as se
from src.sql_runner import connect, list_queries, load_sql


# ------------------------------------------------------------------ frequentist
@pytest.mark.parametrize("xa,na,xb,nb", [(0, 100, 0, 100), (100, 100, 100, 100), (0, 100, 100, 100),
                                         (100, 100, 0, 100), (0, 100, 5, 100)])
def test_ztest_degenerate_counts_emit_no_warnings(xa, na, xb, nb):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any divide-by-zero / invalid RuntimeWarning fails the test
        r = se.two_proportion_ztest(xa, na, xb, nb)
    assert 0.0 <= r.p_value <= 1.0
    assert 0.0 <= r.power <= 1.0
    assert np.isfinite(r.ci_low) and np.isfinite(r.ci_high)


def test_power_is_deterministic_without_sampling_noise():
    assert se.power_two_proportions(0.0, 1.0, 100) == 1.0
    assert se.power_two_proportions(0.0, 0.0, 100) == 0.0


def test_ztest_rel_lift_undefined_when_control_has_no_conversions():
    assert np.isnan(se.two_proportion_ztest(0, 100, 5, 100).rel_lift)


@pytest.mark.parametrize("args", [(0, 100, 101, 100), (-1, 100, 1, 100), (1, 0, 1, 100)])
def test_ztest_rejects_invalid_counts(args):
    with pytest.raises(ValueError):
        se.two_proportion_ztest(*args)


def test_required_sample_size_rejects_target_rate_above_one():
    with pytest.raises(ValueError):
        se.required_sample_size(0.9, 2.0)


# ------------------------------------------------------------------ Bayesian
def test_prob_b_beats_a_large_b_alpha_uses_exact_complement():
    # b_alpha > 50k used to fall back to Monte Carlo; the complement is exact and deterministic
    p = se.prob_b_beats_a(41, 961, 60_001, 940_001)
    q = se.prob_b_beats_a(41, 961, 60_001, 940_001, seed=None)
    assert p == q  # exact: does not depend on the Monte Carlo seed
    mc = se.prob_b_beats_a(41, 961, 60_001, 940_001, method="mc", draws=400_000, seed=3)
    assert p == pytest.approx(mc, abs=0.002)
    # same posteriors, swapped: must be exactly complementary
    assert se.prob_b_beats_a(60_001, 940_001, 41, 961) == pytest.approx(1 - p, abs=1e-12)


def test_prob_b_beats_a_both_paths_agree():
    for a, b in [((12, 990), (30, 970)), ((300, 9700), (330, 9670)), ((1, 1), (2, 1))]:
        direct = se.prob_b_beats_a(*a, *b, method="exact")
        via_complement = 1 - se.prob_b_beats_a(*b, *a, method="exact")
        assert direct == pytest.approx(via_complement, abs=1e-10)


def test_prob_b_beats_a_tiny_probabilities_do_not_underflow_to_garbage():
    p = se.prob_b_beats_a(500, 500, 1, 999)  # B is hopeless
    assert 0.0 <= p < 1e-100 or p == 0.0
    assert se.prob_b_beats_a(1, 999, 500, 500) == pytest.approx(1.0)


def test_prob_b_beats_a_rejects_non_positive_parameters():
    with pytest.raises(ValueError):
        se.prob_b_beats_a(0, 10, 1, 10)


def test_bayesian_lift_median_is_robust_with_zero_conversions():
    r = se.bayesian_ab_test(0, 100, 5, 100)
    # the mean of p_B / p_A - 1 explodes when A has 0 conversions; the median stays sensible
    assert r.rel_lift_mean > 10 * r.rel_lift_median
    assert r.rel_lift_ci[0] < r.rel_lift_median < r.rel_lift_ci[1]
    assert all(np.isfinite(r.rel_lift_ci))


def test_bayesian_weak_prior_stays_finite():
    # Beta(0.01, 0.01) prior: numpy's Beta sampler returns exact zeros for such shapes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = se.bayesian_ab_test(0, 50, 0, 50, prior=(0.01, 0.01))
    assert np.isfinite(r.rel_lift_median)
    assert all(np.isfinite(r.rel_lift_ci))


# ------------------------------------------------------------------ bandits
def test_bandits_validate_inputs():
    with pytest.raises(ValueError):
        se.simulate_bandits([0.1, 1.2], n_rounds=10, n_sims=2)
    with pytest.raises(ValueError):
        se.simulate_bandits([0.1, 0.2], n_rounds=0, n_sims=2)


def test_bandits_handle_zero_rate_arms_and_fewer_rounds_than_arms():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = se.simulate_bandits([0.0, 0.0, 0.0], n_rounds=2, n_sims=3)
    for r in res.values():
        assert r.cumulative_regret[-1] == 0.0
        assert r.total_revenue == 0.0
        assert len(r.cumulative_reward) == 2


def test_bandit_cumulative_reward_is_mean_running_conversions():
    res = se.simulate_bandits([0.2, 0.5], n_rounds=500, n_sims=20, value_per_conversion=1.0)
    for r in res.values():
        assert np.all(np.diff(r.cumulative_reward) >= 0)
        assert r.cumulative_reward[-1] == pytest.approx(r.total_revenue)


# ------------------------------------------------------------------ bidding
def test_beta_prior_with_no_clicks_falls_back_to_default():
    assert bo.fit_beta_prior(np.zeros(5), np.zeros(5)) == (1.0, 99.0)


def test_shrinkage_with_zero_conversions_everywhere_is_finite():
    a, b = bo.fit_beta_prior(np.zeros(10), np.full(10, 50.0))
    cvr = bo.shrunk_cvr(np.zeros(10), np.full(10, 50.0), a, b)
    assert np.all(np.isfinite(cvr)) and np.all(cvr > 0)


# ------------------------------------------------------------------ SQL runner
@pytest.mark.parametrize("name", ["../requirements", "../../etc/passwd", "/etc/passwd", "nope", ""])
def test_load_sql_only_accepts_bundled_queries(name):
    with pytest.raises(ValueError):
        load_sql(name)


def test_load_sql_accepts_names_with_and_without_suffix():
    q = list_queries()[0]
    assert load_sql(q) == load_sql(q + ".sql")


def test_sqlite_connection_is_read_only_by_default(small_dbs):
    con = connect(small_dbs["sqlite_path"])
    try:
        assert con.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM meta")
    finally:
        con.close()


def test_sqlite_read_only_does_not_create_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect(tmp_path / "missing.sqlite")
    assert not (tmp_path / "missing.sqlite").exists()
