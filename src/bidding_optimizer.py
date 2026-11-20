"""Bid optimisation & forecasting.

1. **Expected-value bidding with Bayesian shrinkage**

       Bid_k = pCVR_k x Target_CPA_c,      Target_CPA_c = AOV_c / Target_ROAS

   where pCVR_k is the posterior mean of a Beta-Binomial model with a
   two-level empirical-Bayes prior: prior mean m_c = the keyword's campaign
   CVR, prior strength kappa = shared across the account and fitted by
   maximising the Beta-Binomial marginal likelihood:

       pCVR_k = (conv_k + m_c kappa) / (clicks_k + kappa)

   A keyword with 40 clicks and 0 conversions is no longer bid to the floor,
   and a keyword with 30 clicks and 3 lucky conversions is no longer bid to
   the moon: both are pulled toward the campaign mean in proportion to how
   little evidence they have (prior strength kappa = alpha + beta clicks).

2. **Naive rule-based bidding** (what many accounts run as automated rules):
   every week, using the trailing 14 days -
   * >= 20 clicks and 0 conversions  -> bid x 0.70
   * CPA > 1.2 x target              -> bid x 0.85
   * CPA < 0.8 x target              -> bid x 1.15

3. **Forward simulation** (``compare_strategies``): every strategy starts from
   the same history and bids, then re-bids weekly for N weeks against the
   *same* ground-truth world (``SemWorld``) that generated the history, with
   common random numbers across strategies. Compared on ROAS, CPA,
   conversions, revenue and profit.

4. **Bid landscape forecasting** (``BidLandscape``): fit diminishing-returns
   curves  clicks(b) = a (1 - exp(-c b))  and  CPC(b) = u (1 - exp(-v b))  on the
   weekly (bid, clicks, cost) history of a keyword, then forecast clicks, cost,
   conversions, value and profit for any candidate bid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import betaln

from .data_generator import DEVICE_AOV, DEVICE_CTR, DEVICE_CVR, DEVICE_SHARE, SemWorld

MIN_BID, MAX_BID = 0.05, 8.00

# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

DAILY_SQL = """
SELECT
    CAST(date AS VARCHAR)  AS date,
    keyword_id,
    campaign_id,
    AVG(max_cpc)           AS max_cpc,
    SUM(impressions)       AS impressions,
    SUM(clicks)            AS clicks,
    SUM(cost)              AS cost,
    SUM(conversions)       AS conversions,
    SUM(conv_value)        AS conv_value
FROM fact_keyword_hourly
GROUP BY date, keyword_id, campaign_id
ORDER BY date, keyword_id
"""


def load_daily_history(con) -> pd.DataFrame:
    """Keyword x day history from DuckDB (or SQLite)."""
    import sqlite3

    sql = DAILY_SQL if not isinstance(con, sqlite3.Connection) else DAILY_SQL.replace("VARCHAR", "TEXT")
    df = pd.read_sql_query(sql, con) if isinstance(con, sqlite3.Connection) else con.execute(sql).df()
    for c in ("impressions", "clicks", "conversions"):
        df[c] = df[c].astype(float)
    return df


def load_meta(con) -> dict:
    import sqlite3

    sql = "SELECT * FROM meta"
    df = pd.read_sql_query(sql, con) if isinstance(con, sqlite3.Connection) else con.execute(sql).df()
    return df.iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# Empirical-Bayes shrinkage
# --------------------------------------------------------------------------- #


def fit_beta_prior(
    conversions: np.ndarray,
    clicks: np.ndarray,
    method: str = "mle",
    kappa_bounds: tuple[float, float] = (5.0, 5_000.0),
) -> tuple[float, float]:
    """Empirical-Bayes Beta(alpha, beta) prior for keyword CVRs.

    Parametrised as mean m and strength kappa = alpha + beta ("prior clicks"):
    m is the pooled CVR sum(conv)/sum(clicks), and kappa is estimated by

    * ``method="mle"`` (default) - maximising the Beta-Binomial marginal likelihood

          L(kappa) = prod_k  B(conv_k + m kappa, clicks_k - conv_k + (1-m) kappa)
                             / B(m kappa, (1-m) kappa)

    * ``method="moments"`` - method of moments:
          v_true = click-weighted var(rates) - m(1-m) mean(1/clicks)
          kappa  = m(1-m) / v_true - 1
      (cheap, but collapses to "no heterogeneity" -> full pooling when data
      are sparse, which is exactly when shrinkage matters most).

    kappa is clipped to ``kappa_bounds``.
    """
    conversions = np.asarray(conversions, float)
    clicks = np.asarray(clicks, float)
    mask = clicks > 0
    conversions, clicks = conversions[mask], clicks[mask]
    if clicks.sum() == 0:
        return 1.0, 99.0
    m = float(np.clip(conversions.sum() / clicks.sum(), 1e-4, 0.5))
    lo, hi = kappa_bounds
    if len(clicks) < 2:
        kappa = hi
    elif method == "mle":
        def nll(log_k: float) -> float:
            k = np.exp(log_k)
            a, b = m * k, (1 - m) * k
            return -float(np.sum(betaln(conversions + a, clicks - conversions + b) - betaln(a, b)))

        res = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded")
        kappa = float(np.exp(res.x))
    elif method == "moments":
        rates = conversions / clicks
        w = clicks / clicks.sum()
        v_obs = float(np.sum(w * (rates - m) ** 2))
        v_noise = m * (1 - m) * float(np.mean(1.0 / clicks))
        v_true = max(v_obs - v_noise, 1e-12)
        kappa = m * (1 - m) / v_true - 1
    else:
        raise ValueError(f"unknown method {method!r}")
    kappa = float(np.clip(kappa, lo, hi))
    return m * kappa, (1 - m) * kappa


def shrunk_cvr(conversions, clicks, alpha: float, beta: float) -> np.ndarray:
    """Posterior mean of Beta(alpha + conv, beta + clicks - conv)."""
    return (np.asarray(conversions, float) + alpha) / (np.asarray(clicks, float) + alpha + beta)


def fit_shared_kappa(
    conversions: np.ndarray,
    clicks: np.ndarray,
    prior_means: np.ndarray,
    kappa_bounds: tuple[float, float] = (5.0, 5_000.0),
) -> float:
    """Hierarchical variant: every keyword k has its own prior mean m_k (its
    campaign's pooled CVR) and all keywords share one prior strength kappa,
    fitted by maximising the Beta-Binomial marginal likelihood

        sum_k  log B(conv_k + m_k kappa, clicks_k - conv_k + (1 - m_k) kappa)
               - log B(m_k kappa, (1 - m_k) kappa)

    Campaign means are data-rich and kappa borrows strength across the whole
    account, which is far more stable than a separate kappa per campaign when a
    campaign only has a handful of keywords.
    """
    conv = np.asarray(conversions, float)
    clk = np.asarray(clicks, float)
    m = np.clip(np.asarray(prior_means, float), 1e-4, 0.5)
    mask = clk > 0
    conv, clk, m = conv[mask], clk[mask], m[mask]
    lo, hi = kappa_bounds
    if len(clk) < 2:
        return hi

    def nll(log_k: float) -> float:
        k = np.exp(log_k)
        return -float(np.sum(betaln(conv + m * k, clk - conv + (1 - m) * k) - betaln(m * k, (1 - m) * k)))

    res = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded")
    return float(np.clip(np.exp(res.x), lo, hi))


def keyword_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Per-keyword totals + raw and shrunk CVR.

    Two-level partial pooling: prior mean = the keyword's campaign CVR,
    prior strength kappa = shared across the account (``fit_shared_kappa``).
    """
    kw = (
        history.groupby(["keyword_id", "campaign_id"], as_index=False)[
            ["clicks", "cost", "conversions", "conv_value"]
        ].sum()
    )
    camp = kw.groupby("campaign_id")[["conversions", "clicks"]].sum()
    account_mean = kw.conversions.sum() / max(kw.clicks.sum(), 1.0)
    # campaign mean, lightly smoothed toward the account mean for tiny campaigns
    camp_mean = (camp.conversions + 2.0 * account_mean) / (camp.clicks + 2.0)
    m = kw.campaign_id.map(camp_mean).to_numpy()
    kappa = fit_shared_kappa(kw.conversions, kw.clicks, m)
    kw["prior_alpha"], kw["prior_beta"] = m * kappa, (1 - m) * kappa
    kw["raw_cvr"] = np.where(kw.clicks > 0, kw.conversions / kw.clicks.clip(lower=1), 0.0)
    kw["pcvr"] = shrunk_cvr(kw.conversions, kw.clicks, kw.prior_alpha, kw.prior_beta)
    kw["shrinkage_weight"] = kappa / (kw.clicks + kappa)  # 1 = all prior, 0 = all data
    return kw


def campaign_target_cpa(history: pd.DataFrame, target_roas: float) -> pd.Series:
    """Target CPA per campaign = historical AOV / target ROAS."""
    c = history.groupby("campaign_id")[["conversions", "conv_value"]].sum()
    aov = c.conv_value / c.conversions.replace(0, np.nan)
    aov = aov.fillna(aov.median())
    return aov / target_roas


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


@dataclass
class Strategy:
    name: str

    def propose(self, history: pd.DataFrame, current_bids: np.ndarray, keyword_ids: np.ndarray,
                target_cpa: pd.Series) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class ExpectedValueBidding(Strategy):
    """Bid = pCVR x Target_CPA (pCVR shrunk with empirical-Bayes priors)."""

    name: str = "ev_shrinkage"
    shrink: bool = True

    def propose(self, history, current_bids, keyword_ids, target_cpa):
        kw = keyword_summary(history).set_index("keyword_id").reindex(keyword_ids)
        cvr = kw.pcvr if self.shrink else kw.raw_cvr
        bids = cvr.to_numpy() * target_cpa.reindex(kw.campaign_id).to_numpy()
        return np.clip(np.nan_to_num(bids, nan=MIN_BID), MIN_BID, MAX_BID)


@dataclass
class RuleBasedBidding(Strategy):
    """Classic weekly automated rules on a trailing window, no statistics."""

    name: str = "rule_based"
    window_days: int = 14
    min_clicks_no_conv: int = 20
    up: float = 1.15
    down: float = 0.85
    cut: float = 0.70

    def propose(self, history, current_bids, keyword_ids, target_cpa):
        dates = sorted(history.date.unique())[-self.window_days:]
        recent = history[history.date.isin(dates)]
        kw = (
            recent.groupby(["keyword_id", "campaign_id"])[["clicks", "cost", "conversions"]]
            .sum()
            .reset_index()
            .set_index("keyword_id")
            .reindex(keyword_ids)
        )
        tgt = target_cpa.reindex(kw.campaign_id).to_numpy()
        cpa = kw.cost.to_numpy() / np.where(kw.conversions > 0, kw.conversions, np.nan)
        mult = np.ones(len(kw))
        no_conv = (kw.conversions.to_numpy() == 0) & (kw.clicks.to_numpy() >= self.min_clicks_no_conv)
        mult[no_conv] = self.cut
        has_conv = kw.conversions.to_numpy() > 0
        mult[has_conv & (cpa > 1.2 * tgt)] = self.down
        mult[has_conv & (cpa < 0.8 * tgt)] = self.up
        return np.clip(current_bids * mult, MIN_BID, MAX_BID)


@dataclass
class StaticBidding(Strategy):
    """Do nothing: keep the last historical bids (control)."""

    name: str = "static_bids"

    def propose(self, history, current_bids, keyword_ids, target_cpa):
        return current_bids.copy()


@dataclass
class OracleBidding(Strategy):
    """Upper reference with perfect information: bid = TRUE value per click /
    target ROAS (true CVR x true AOV from the simulator). Not achievable in
    practice; shows how much headroom estimation error leaves."""

    name: str = "oracle_true_value"
    true_value_per_click: pd.Series = field(default_factory=pd.Series)
    target_roas: float = 4.0

    def propose(self, history, current_bids, keyword_ids, target_cpa):
        vpc = self.true_value_per_click.reindex(keyword_ids).to_numpy()
        return np.clip(vpc / self.target_roas, MIN_BID, MAX_BID)


# --------------------------------------------------------------------------- #
# Forward simulation
# --------------------------------------------------------------------------- #


@dataclass
class SimulationResult:
    summary: pd.DataFrame  # one row per strategy (mean over seeds)
    per_seed: pd.DataFrame  # one row per strategy x seed
    daily: pd.DataFrame  # strategy x seed x date totals
    final_bids: pd.DataFrame  # keyword x strategy (seed 0)


def compare_strategies(
    world: SemWorld,
    history: pd.DataFrame,
    strategies: list[Strategy],
    target_roas: float = 4.0,
    n_weeks: int = 4,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    margin: float = 0.35,
) -> SimulationResult:
    """Walk-forward simulation, re-bidding weekly. Profit = value x margin - cost."""
    keyword_ids = world.keywords.keyword_id.to_numpy()
    last_date = pd.Timestamp(history.date.max())
    last_bids = (
        history[history.date == history.date.max()].set_index("keyword_id").max_cpc.reindex(keyword_ids).to_numpy()
    )
    target_cpa = campaign_target_cpa(history, target_roas)

    rows, daily_rows, final_bids = [], [], {}
    for strat in strategies:
        for seed in seeds:
            hist = history.copy()
            bids = last_bids.copy()
            for w in range(n_weeks):
                bids = strat.propose(hist, bids, keyword_ids, target_cpa)
                dates = pd.date_range(last_date + pd.Timedelta(days=1 + 7 * w), periods=7, freq="D")
                rng = np.random.default_rng([seed, w, 2024])  # common random numbers
                kf, _ = world.simulate(dates, np.repeat(bids[None, :], 7, axis=0), rng, grain="daily")
                kf["date"] = kf.date.dt.strftime("%Y-%m-%d")
                kf["strategy"], kf["seed"] = strat.name, seed
                daily_rows.append(kf)
                hist = pd.concat(
                    [hist, kf[["date", "keyword_id", "campaign_id", "max_cpc", "impressions", "clicks",
                               "cost", "conversions", "conv_value"]]],
                    ignore_index=True,
                )
            if seed == seeds[0]:
                final_bids[strat.name] = bids

    daily = pd.concat(daily_rows, ignore_index=True)
    agg = daily.groupby(["strategy", "seed"])[["impressions", "clicks", "cost", "conversions", "conv_value"]].sum()
    agg["roas"] = agg.conv_value / agg.cost
    agg["cpa"] = agg.cost / agg.conversions
    agg["profit"] = agg.conv_value * margin - agg.cost
    per_seed = agg.reset_index()
    summary = per_seed.groupby("strategy").agg(
        cost=("cost", "mean"), conversions=("conversions", "mean"), conv_value=("conv_value", "mean"),
        roas=("roas", "mean"), roas_sd=("roas", "std"), cpa=("cpa", "mean"),
        profit=("profit", "mean"), profit_sd=("profit", "std"), clicks=("clicks", "mean"),
    )
    order = [s.name for s in strategies]
    summary = summary.reindex(order).reset_index()
    daily_tot = (
        daily.groupby(["strategy", "seed", "date"])[["cost", "conversions", "conv_value"]].sum().reset_index()
    )
    fb = pd.DataFrame(final_bids, index=keyword_ids)
    fb.insert(0, "last_historical_bid", last_bids)
    fb.index.name = "keyword_id"
    return SimulationResult(summary, per_seed, daily_tot, fb.reset_index())


def default_strategies(world: SemWorld, target_roas: float) -> list[Strategy]:
    kw = world.keywords.set_index("keyword_id")
    dev_aov = float((DEVICE_SHARE * DEVICE_CTR * DEVICE_CVR * DEVICE_AOV).sum()
                    / (DEVICE_SHARE * DEVICE_CTR * DEVICE_CVR).sum())
    return [
        StaticBidding(),
        RuleBasedBidding(),
        ExpectedValueBidding(name="ev_raw_cvr", shrink=False),
        ExpectedValueBidding(name="ev_shrinkage", shrink=True),
        OracleBidding(true_value_per_click=kw.eff_cvr * kw.true_aov * dev_aov, target_roas=target_roas),
    ]


# --------------------------------------------------------------------------- #
# Bid landscape forecasting
# --------------------------------------------------------------------------- #


def _saturating(b, a, c):
    return a * (1.0 - np.exp(-c * b))


@dataclass
class BidLandscape:
    """Per-keyword response curves fitted on weekly history:

        clicks(b) = a (1 - exp(-c b))        traffic saturates with bid
        CPC(b)    = u (1 - exp(-v b))        GSP price rises, capped by rivals' ranks
    """

    keyword_id: int
    a: float
    c: float
    u: float
    v: float
    pcvr: float
    aov: float
    bid_min_observed: float = MIN_BID
    bid_max_observed: float = MAX_BID
    period_days: int = 7

    @staticmethod
    def _fit_saturating(b: np.ndarray, y: np.ndarray, scale_hint: float) -> tuple[float, float]:
        a0 = max(float(y.max()) * 1.5, 1e-3)
        try:
            (a, c), _ = curve_fit(_saturating, b, y, p0=(a0, 1.0 / max(scale_hint, 1e-3)),
                                  bounds=([0.0, 1e-3], [a0 * 50, 100.0]), maxfev=20_000)
        except (RuntimeError, ValueError):  # degenerate data
            a, c = a0, 1.0 / max(scale_hint, 1e-3)
        return float(a), float(c)

    @classmethod
    def fit(cls, weekly: pd.DataFrame, keyword_id: int, pcvr: float, aov: float) -> BidLandscape:
        """``weekly`` needs columns keyword_id, max_cpc, clicks, cost (e.g. sql/07)."""
        d = weekly[(weekly.keyword_id == keyword_id) & (weekly.max_cpc > 0)]
        b, y = d.max_cpc.to_numpy(float), d.clicks.to_numpy(float)
        a, c = cls._fit_saturating(b, y, b.mean())
        m = d.clicks > 0
        if m.sum() >= 3:
            u, v = cls._fit_saturating(b[m], (d.cost[m] / d.clicks[m]).to_numpy(float), b.mean())
        else:
            u, v = float(b.max()), 1.0
        return cls(int(keyword_id), a, c, u, v, float(pcvr), float(aov),
                   float(b.min()), float(b.max()))

    def forecast(self, bids) -> pd.DataFrame:
        b = np.asarray(bids, float)
        clicks = _saturating(b, self.a, self.c)
        cpc = np.minimum(_saturating(b, self.u, self.v), b)
        cost = clicks * cpc
        conv = clicks * self.pcvr
        value = conv * self.aov
        out = pd.DataFrame({"bid": b, "clicks": clicks, "avg_cpc": cpc, "cost": cost,
                            "conversions": conv, "value": value})
        out["roas"] = out.value / out.cost.replace(0, np.nan)
        if len(b) > 1:  # cost of the *next* click: d cost / d clicks
            out["marginal_cpc"] = np.gradient(cost, b) / np.maximum(np.gradient(clicks, b), 1e-9)
        return out

    def best_bid(self, margin: float = 0.35, grid=None) -> float:
        """Bid that maximises forecast profit = value x margin - cost.

        The default grid stops at 1.5x the highest bid ever observed: beyond
        that the curve is pure extrapolation and should not drive decisions.
        """
        if grid is None:
            grid = np.linspace(MIN_BID, min(MAX_BID, 1.5 * self.bid_max_observed), 400)
        f = self.forecast(grid)
        return float(grid[np.argmax(f.value * margin - f.cost)])


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def cvr_estimation_error(history: pd.DataFrame, world: SemWorld,
                         bins: tuple[float, ...] = (0, 300, 1_000, 5_000, np.inf)) -> pd.DataFrame:
    """Absolute error of raw vs shrunk CVR against the simulator's true CVR,
    by keyword click-volume bucket. Shows *where* shrinkage pays off."""
    kw = keyword_summary(history).merge(world.keywords[["keyword_id", "eff_cvr"]], on="keyword_id")
    kw["abs_err_raw"] = (kw.raw_cvr - kw.eff_cvr).abs()
    kw["abs_err_shrunk"] = (kw.pcvr - kw.eff_cvr).abs()
    labels = [f"{int(lo)}-{int(hi) if np.isfinite(hi) else 'inf'} clicks" for lo, hi in zip(bins[:-1], bins[1:])]
    kw["volume_bucket"] = pd.cut(kw.clicks, bins=list(bins), labels=labels, right=False)
    out = kw.groupby("volume_bucket", observed=True).agg(
        keywords=("keyword_id", "count"),
        mae_raw=("abs_err_raw", "mean"),
        mae_shrunk=("abs_err_shrunk", "mean"),
        mean_true_cvr=("eff_cvr", "mean"),
    )
    out["error_reduction"] = 1 - out.mae_shrunk / out.mae_raw
    total = pd.DataFrame(
        {"keywords": [len(kw)], "mae_raw": [kw.abs_err_raw.mean()], "mae_shrunk": [kw.abs_err_shrunk.mean()],
         "mean_true_cvr": [kw.eff_cvr.mean()]},
        index=pd.Index(["all keywords"], name="volume_bucket"),
    )
    total["error_reduction"] = 1 - total.mae_shrunk / total.mae_raw
    return pd.concat([out, total]).reset_index()
