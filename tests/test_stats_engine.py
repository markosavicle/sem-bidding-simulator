"""Statistical correctness of the A/B testing and bandit engine."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src import stats_engine as se


# ------------------------------------------------------------ frequentist
def test_ztest_matches_chi_square_without_correction():
    # two-sided pooled z-test is equivalent to a 2x2 chi-square test (z^2 = chi2)
    r = se.two_proportion_ztest(400, 10_000, 460, 10_000)
    table = np.array([[400, 9_600], [460, 9_540]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    assert r.z**2 == pytest.approx(chi2, rel=1e-9)
    assert r.p_value == pytest.approx(p, rel=1e-9)


def test_ztest_known_values():
    r = se.two_proportion_ztest(400, 10_000, 460, 10_000)
    assert r.rate_a == 0.04 and r.rate_b == 0.046
    assert r.rel_lift == pytest.approx(0.15)
    assert r.z == pytest.approx(2.0914, abs=1e-4)
    assert r.p_value == pytest.approx(0.03649, abs=1e-5)
    # unpooled Wald CI for the difference
    se1 = np.sqrt(0.04 * 0.96 / 10_000 + 0.046 * 0.954 / 10_000)
    assert r.ci_low == pytest.approx(0.006 - 1.959964 * se1, abs=1e-6)
    assert r.ci_high == pytest.approx(0.006 + 1.959964 * se1, abs=1e-6)
    assert r.significant


def test_ztest_one_sided_is_half_of_two_sided():
    two = se.two_proportion_ztest(400, 10_000, 460, 10_000)
    one = se.two_proportion_ztest(400, 10_000, 460, 10_000, alternative="larger")
    assert one.p_value == pytest.approx(two.p_value / 2)


def test_ztest_input_validation():
    with pytest.raises(ValueError):
        se.two_proportion_ztest(10, 0, 5, 100)
    with pytest.raises(ValueError):
        se.two_proportion_ztest(101, 100, 5, 100)


def test_required_sample_size_reference_value():
    # 10% -> 12%, alpha 0.05 two-sided, power 0.8: textbook answer 3,841 per arm
    assert se.required_sample_size(0.10, 0.02, relative=False) == 3841
    # relative MDE form gives the same answer
    assert se.required_sample_size(0.10, 0.20, relative=True) == 3841


def test_sample_size_monotonicity():
    n_small = se.required_sample_size(0.05, 0.05)
    n_big = se.required_sample_size(0.05, 0.20)
    assert n_small > n_big  # smaller effects need more traffic
    assert se.required_sample_size(0.05, 0.1, power=0.9) > se.required_sample_size(0.05, 0.1, power=0.8)


def test_power_at_required_n_is_target_power():
    n = se.required_sample_size(0.04, 0.10, power=0.80)
    pw = se.power_two_proportions(0.04, 0.044, n_a=n)
    assert pw == pytest.approx(0.80, abs=0.005)


def test_ci_coverage_by_simulation():
    rng = np.random.default_rng(1)
    pa, pb, n = 0.05, 0.06, 5_000
    covered = 0
    sims = 2_000
    for _ in range(sims):
        r = se.two_proportion_ztest(rng.binomial(n, pa), n, rng.binomial(n, pb), n)
        covered += r.ci_low <= (pb - pa) <= r.ci_high
    assert 0.935 <= covered / sims <= 0.965


def test_type_one_error_rate_under_null():
    rng = np.random.default_rng(2)
    n, p, sims = 4_000, 0.03, 3_000
    rejections = sum(
        se.two_proportion_ztest(rng.binomial(n, p), n, rng.binomial(n, p), n).p_value < 0.05 for _ in range(sims)
    )
    assert 0.035 <= rejections / sims <= 0.065


def test_empirical_power_matches_formula():
    rng = np.random.default_rng(3)
    n = se.required_sample_size(0.05, 0.20)
    sims = 1_500
    hits = sum(
        se.two_proportion_ztest(rng.binomial(n, 0.05), n, rng.binomial(n, 0.06), n).p_value < 0.05
        for _ in range(sims)
    )
    assert hits / sims == pytest.approx(0.80, abs=0.04)


def test_srm_check():
    assert not se.srm_check(10_000, 10_050)["srm_detected"]
    assert se.srm_check(10_000, 11_000)["srm_detected"]


# --------------------------------------------------------------- bayesian
def test_beta_posterior_update():
    assert se.beta_posterior(30, 100) == (31.0, 71.0)
    assert se.beta_posterior(30, 100, 2, 8) == (32.0, 78.0)


def test_prob_b_beats_a_exact_matches_monte_carlo():
    exact = se.prob_b_beats_a(41, 961, 56, 946, method="exact")
    mc = se.prob_b_beats_a(41, 961, 56, 946, method="mc", draws=400_000, seed=5)
    assert exact == pytest.approx(mc, abs=0.005)


def test_prob_b_beats_a_symmetry_and_identity():
    assert se.prob_b_beats_a(11, 91, 11, 91) == pytest.approx(0.5, abs=1e-9)
    p = se.prob_b_beats_a(20, 80, 30, 70)
    q = se.prob_b_beats_a(30, 70, 20, 80)
    assert p + q == pytest.approx(1.0, abs=1e-9)
    assert p > 0.9


def test_prob_b_beats_a_non_integer_falls_back_to_mc():
    p = se.prob_b_beats_a(10.5, 90.5, 15.5, 85.5)
    assert 0.5 < p < 1.0


def test_bayesian_ab_test_consistency():
    r = se.bayesian_ab_test(400, 10_000, 460, 10_000)
    assert r.posterior_a == (401.0, 9_601.0)
    assert r.mean_b > r.mean_a
    assert r.ci_a[0] < r.mean_a < r.ci_a[1]
    assert r.prob_b_beats_a == pytest.approx(0.982, abs=0.003)
    # expected loss is small for the winner and ~ the gap for the loser
    assert r.expected_loss_choose_b < 1e-4
    assert r.expected_loss_choose_a == pytest.approx(0.006, abs=0.0005)
    assert r.rel_lift_ci[0] < r.rel_lift_mean < r.rel_lift_ci[1]


def test_bayes_and_frequentist_agree_on_direction():
    z = se.two_proportion_ztest(400, 10_000, 460, 10_000, alternative="larger")
    b = se.bayesian_ab_test(400, 10_000, 460, 10_000)
    # with a flat prior, P(B>A) ~ 1 - one-sided p-value for large samples
    assert b.prob_b_beats_a == pytest.approx(1 - z.p_value, abs=0.01)


# ---------------------------------------------------------------- bandits
@pytest.fixture(scope="module")
def bandit_results():
    return se.simulate_bandits([0.03, 0.035, 0.045], n_rounds=8_000, n_sims=60, seed=11)


def test_static_split_regret_is_exactly_linear(bandit_results):
    r = bandit_results["static_split"]
    gap = np.mean([0.045 - 0.03, 0.045 - 0.035, 0.0])
    assert r.cumulative_regret[-1] == pytest.approx(gap * 8_000, rel=0.01)
    np.testing.assert_allclose(r.arm_share, [1 / 3] * 3, atol=1e-3)


def test_thompson_beats_static_and_ucb1(bandit_results):
    ts, ucb, st = (bandit_results[k] for k in ("thompson", "ucb1", "static_split"))
    assert ts.cumulative_regret[-1] < 0.6 * st.cumulative_regret[-1]
    assert ts.cumulative_regret[-1] < ucb.cumulative_regret[-1]
    assert ts.total_revenue > st.total_revenue
    assert ts.arm_share[2] > 0.6  # most traffic goes to the best arm


def test_ucb1_regret_is_sublinear_and_beats_static(bandit_results):
    ucb, st = bandit_results["ucb1"], bandit_results["static_split"]
    assert ucb.cumulative_regret[-1] < st.cumulative_regret[-1]
    first_half = ucb.cumulative_regret[3_999]
    second_half = ucb.cumulative_regret[-1] - first_half
    assert second_half < first_half  # regret growth slows down


def test_regret_is_monotone_and_bands_ordered(bandit_results):
    for r in bandit_results.values():
        assert np.all(np.diff(r.cumulative_regret) >= -1e-12)
        assert np.all(r.regret_p05 <= r.cumulative_regret + 1e-9)
        assert np.all(r.cumulative_regret <= r.regret_p95 + 1e-9)


def test_bandit_summary_lift(bandit_results):
    rows = {r["policy"]: r for r in se.bandit_summary(bandit_results)}
    assert rows["static_split"]["revenue_lift_vs_static"] == 0
    assert rows["thompson"]["revenue_lift_vs_static"] > 0.05
