"""Statistical engine: frequentist & Bayesian A/B testing + multi-armed bandits.

Frequentist (two-proportion tests on CTR / CVR)
    * ``two_proportion_ztest``   z statistic, p-value, Wald CI of the difference,
                                 relative lift and achieved power
    * ``required_sample_size``   minimum n per arm for a baseline rate + MDE
    * ``power_two_proportions``  power of a test at a given n per arm
    * ``srm_check``              sample-ratio-mismatch chi-square guard

Bayesian (Beta-Binomial)
    * ``prob_b_beats_a``         exact P(p_B > p_A) (closed form) or Monte Carlo
    * ``bayesian_ab_test``       posteriors, credible intervals, P(B>A),
                                 expected loss and relative-lift distribution

Bandits
    * ``simulate_bandits``       Thompson Sampling, UCB1 and a static equal split
                                 (50/50 for two arms) over many replications,
                                 returning cumulative regret & revenue curves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from scipy.special import betaln

# --------------------------------------------------------------------------- #
# Frequentist
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ZTestResult:
    rate_a: float
    rate_b: float
    abs_lift: float
    rel_lift: float
    z: float
    p_value: float
    ci_low: float  # CI of (rate_b - rate_a)
    ci_high: float
    alpha: float
    significant: bool
    power: float  # power to detect the observed effect at these n (post-hoc)

    def as_dict(self) -> dict:
        return asdict(self)


def _z(alpha: float, two_sided: bool) -> float:
    return stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)


def two_proportion_ztest(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> ZTestResult:
    """Two-proportion z-test (B vs control A).

    Test statistic uses the *pooled* rate under H0:
        z = (p_b - p_a) / sqrt(p(1-p) (1/n_a + 1/n_b))
    The confidence interval uses the *unpooled* (Wald) standard error:
        (p_b - p_a) +/- z_{1-a/2} sqrt(p_a(1-p_a)/n_a + p_b(1-p_b)/n_b)

    alternative: "two-sided" | "larger" (H1: p_b > p_a) | "smaller".
    """
    if min(trials_a, trials_b) <= 0:
        raise ValueError("trials must be positive")
    if not (0 <= successes_a <= trials_a and 0 <= successes_b <= trials_b):
        raise ValueError("successes must be within [0, trials]")
    pa, pb = successes_a / trials_a, successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / trials_a + 1 / trials_b))
    diff = pb - pa
    z = diff / se0 if se0 > 0 else 0.0
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "larger":
        p = stats.norm.sf(z)
    elif alternative == "smaller":
        p = stats.norm.cdf(z)
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    se1 = np.sqrt(pa * (1 - pa) / trials_a + pb * (1 - pb) / trials_b)
    zc = _z(alpha, two_sided=True)
    power = power_two_proportions(pa, pb, n_a=trials_a, n_b=trials_b, alpha=alpha) if pa != pb else alpha
    return ZTestResult(
        rate_a=pa,
        rate_b=pb,
        abs_lift=diff,
        rel_lift=diff / pa if pa > 0 else float("nan"),
        z=float(z),
        p_value=float(p),
        ci_low=float(diff - zc * se1),
        ci_high=float(diff + zc * se1),
        alpha=alpha,
        significant=bool(p < alpha),
        power=float(power),
    )


def required_sample_size(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    relative: bool = True,
    two_sided: bool = True,
) -> int:
    """Minimum sample size **per arm** for a two-proportion test.

        n = [ z_{1-a/2} sqrt(2 p_bar (1-p_bar)) + z_{1-b} sqrt(p1(1-p1) + p2(1-p2)) ]^2
            / (p2 - p1)^2

    ``mde`` is relative (0.10 = +10%) when ``relative`` else absolute (0.01 = +1pp).
    """
    p1 = baseline
    p2 = baseline * (1 + mde) if relative else baseline + mde
    if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
        raise ValueError("baseline and baseline+MDE must be distinct rates in (0, 1)")
    p_bar = (p1 + p2) / 2
    za, zb = _z(alpha, two_sided), stats.norm.ppf(power)
    num = (za * np.sqrt(2 * p_bar * (1 - p_bar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def power_two_proportions(
    p1: float,
    p2: float,
    n_a: int,
    n_b: int | None = None,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of the two-proportion z-test (normal approximation).

        power = Phi( (|p2-p1| - z_{1-a/2} * SE0) / SE1 )
    with SE0 the pooled (H0) and SE1 the unpooled (H1) standard errors.
    The negligible opposite-tail rejection probability is ignored.
    """
    n_b = n_a if n_b is None else n_b
    p_bar = (p1 * n_a + p2 * n_b) / (n_a + n_b)
    se0 = np.sqrt(p_bar * (1 - p_bar) * (1 / n_a + 1 / n_b))
    se1 = np.sqrt(p1 * (1 - p1) / n_a + p2 * (1 - p2) / n_b)
    za = _z(alpha, two_sided)
    return float(stats.norm.cdf((abs(p2 - p1) - za * se0) / se1))


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5, threshold: float = 0.001) -> dict:
    """Sample Ratio Mismatch: chi-square goodness-of-fit on the traffic split.
    A p-value below ``threshold`` means the randomisation is broken and the
    experiment result should not be trusted."""
    total = n_a + n_b
    expected = np.array([expected_share_a, 1 - expected_share_a]) * total
    chi2, p = stats.chisquare([n_a, n_b], expected)
    return {"chi2": float(chi2), "p_value": float(p), "srm_detected": bool(p < threshold)}


# --------------------------------------------------------------------------- #
# Bayesian Beta-Binomial
# --------------------------------------------------------------------------- #


def beta_posterior(successes: int, trials: int, prior_a: float = 1.0, prior_b: float = 1.0) -> tuple[float, float]:
    """Conjugate update: Beta(a, b) prior + Binomial data -> Beta(a + s, b + n - s)."""
    return prior_a + successes, prior_b + trials - successes


def prob_b_beats_a(
    a_alpha: float, a_beta: float, b_alpha: float, b_beta: float,
    method: str = "auto", draws: int = 200_000, seed: int | None = 0,
) -> float:
    """P(p_B > p_A) for p_A ~ Beta(a_alpha, a_beta), p_B ~ Beta(b_alpha, b_beta).

    Exact closed form (valid for integer b_alpha), computed in log space:

        P = sum_{i=0}^{b_alpha-1}  B(a_alpha + i, a_beta + b_beta)
                                  / ((b_beta + i) B(1 + i, b_beta) B(a_alpha, a_beta))

    Falls back to Monte Carlo for non-integer parameters or method="mc".
    """
    integer = float(b_alpha).is_integer()
    if method == "exact" or (method == "auto" and integer and b_alpha <= 50_000):
        if not integer:
            raise ValueError("exact method needs an integer b_alpha")
        i = np.arange(int(b_alpha))
        log_terms = (
            betaln(a_alpha + i, a_beta + b_beta)
            - np.log(b_beta + i)
            - betaln(1 + i, b_beta)
            - betaln(a_alpha, a_beta)
        )
        return float(np.clip(np.exp(log_terms).sum(), 0.0, 1.0))
    rng = np.random.default_rng(seed)
    pa = rng.beta(a_alpha, a_beta, draws)
    pb = rng.beta(b_alpha, b_beta, draws)
    return float((pb > pa).mean())


@dataclass(frozen=True)
class BayesResult:
    posterior_a: tuple[float, float]
    posterior_b: tuple[float, float]
    mean_a: float
    mean_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    prob_b_beats_a: float
    expected_loss_choose_b: float  # E[max(p_a - p_b, 0)]  (risk of shipping B)
    expected_loss_choose_a: float
    rel_lift_mean: float
    rel_lift_ci: tuple[float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def bayesian_ab_test(
    successes_a: int, trials_a: int, successes_b: int, trials_b: int,
    prior: tuple[float, float] = (1.0, 1.0), credible: float = 0.95,
    draws: int = 200_000, seed: int | None = 0,
) -> BayesResult:
    """Beta-Binomial A/B test (CTR or CVR). Uniform Beta(1,1) prior by default;
    pass an informative prior (e.g. from historical campaign data) to stabilise
    low-traffic tests."""
    a = beta_posterior(successes_a, trials_a, *prior)
    b = beta_posterior(successes_b, trials_b, *prior)
    rng = np.random.default_rng(seed)
    pa = rng.beta(*a, draws)
    pb = rng.beta(*b, draws)
    q = ((1 - credible) / 2, 1 - (1 - credible) / 2)
    lift = (pb - pa) / pa
    return BayesResult(
        posterior_a=a,
        posterior_b=b,
        mean_a=a[0] / (a[0] + a[1]),
        mean_b=b[0] / (b[0] + b[1]),
        ci_a=tuple(stats.beta.ppf(q, *a)),
        ci_b=tuple(stats.beta.ppf(q, *b)),
        prob_b_beats_a=prob_b_beats_a(*a, *b),
        expected_loss_choose_b=float(np.maximum(pa - pb, 0).mean()),
        expected_loss_choose_a=float(np.maximum(pb - pa, 0).mean()),
        rel_lift_mean=float(lift.mean()),
        rel_lift_ci=tuple(np.quantile(lift, q)),
    )


# --------------------------------------------------------------------------- #
# Multi-armed bandits
# --------------------------------------------------------------------------- #


@dataclass
class BanditResult:
    policy: str
    cumulative_regret: np.ndarray  # mean over replications, shape (n_rounds,)
    regret_p05: np.ndarray
    regret_p95: np.ndarray
    cumulative_reward: np.ndarray  # mean conversions, shape (n_rounds,)
    total_revenue: float  # mean over replications
    total_revenue_sd: float
    arm_share: np.ndarray  # mean share of traffic per arm, shape (n_arms,)
    best_arm: int = 0


def _choose_static(t, s, f, rng, n_sims, n_arms):
    # round-robin equal split (deterministic 50/50 for two arms)
    return np.full(n_sims, t % n_arms)


def _choose_thompson(t, s, f, rng, n_sims, n_arms):
    # sample a plausible rate per arm from Beta(1+s, 1+f), play the best sample
    return np.argmax(rng.beta(1.0 + s, 1.0 + f), axis=1)


def _choose_ucb1(t, s, f, rng, n_sims, n_arms):
    n = s + f
    if t < n_arms:  # play each arm once
        return np.full(n_sims, t)
    mean = s / n
    bonus = np.sqrt(2.0 * np.log(t) / n)
    return np.argmax(mean + bonus, axis=1)


POLICIES = {"static_split": _choose_static, "thompson": _choose_thompson, "ucb1": _choose_ucb1}


def simulate_bandits(
    true_rates: list[float] | np.ndarray,
    n_rounds: int = 20_000,
    n_sims: int = 200,
    value_per_conversion: float = 50.0,
    policies: tuple[str, ...] = ("static_split", "thompson", "ucb1"),
    seed: int = 7,
) -> dict[str, BanditResult]:
    """Monte-Carlo comparison of allocation policies on Bernoulli arms.

    Each round = one visitor/click shown one ad variant; reward = conversion.
    Pseudo-regret_t = sum_{s<=t} (p* - p_{a_s})  (expected conversions lost vs.
    always showing the best variant). All replications run vectorised.
    """
    p = np.asarray(true_rates, dtype=float)
    n_arms = len(p)
    best = p.max()
    out: dict[str, BanditResult] = {}
    for name in policies:
        choose = POLICIES[name]
        rng = np.random.default_rng(seed)  # same seed -> common random numbers
        s = np.zeros((n_sims, n_arms))
        f = np.zeros((n_sims, n_arms))
        regret = np.zeros((n_sims, n_rounds))
        reward = np.zeros((n_sims, n_rounds))
        rows = np.arange(n_sims)
        for t in range(n_rounds):
            arm = choose(t, s, f, rng, n_sims, n_arms)
            r = rng.random(n_sims) < p[arm]
            s[rows, arm] += r
            f[rows, arm] += ~r
            regret[:, t] = best - p[arm]
            reward[:, t] = r
        cum_regret = regret.cumsum(axis=1)
        cum_reward = reward.cumsum(axis=1)
        pulls = s + f
        revenue = cum_reward[:, -1] * value_per_conversion
        out[name] = BanditResult(
            policy=name,
            cumulative_regret=cum_regret.mean(axis=0),
            regret_p05=np.quantile(cum_regret, 0.05, axis=0),
            regret_p95=np.quantile(cum_regret, 0.95, axis=0),
            cumulative_reward=cum_reward.mean(axis=0),
            total_revenue=float(revenue.mean()),
            total_revenue_sd=float(revenue.std(ddof=1)) if n_sims > 1 else 0.0,
            arm_share=(pulls / pulls.sum(axis=1, keepdims=True)).mean(axis=0),
            best_arm=int(np.argmax(p)),
        )
    return out


def bandit_summary(results: dict[str, BanditResult], baseline: str = "static_split") -> list[dict]:
    """Table: final regret, revenue and revenue lift vs. the static split."""
    base = results[baseline].total_revenue if baseline in results else None
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "policy": name,
                "final_cum_regret": float(r.cumulative_regret[-1]),
                "total_revenue": r.total_revenue,
                "revenue_sd": r.total_revenue_sd,
                "revenue_lift_vs_static": (r.total_revenue / base - 1) if base else None,
                "best_arm_share": float(r.arm_share[r.best_arm]),
                "arm_share": [round(float(x), 4) for x in r.arm_share],
            }
        )
    return rows
