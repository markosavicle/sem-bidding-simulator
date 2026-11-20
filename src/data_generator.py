"""Synthetic SEM auction & campaign data generator.

The generator is built around a small, explicit *world model* (``SemWorld``):

* every keyword has latent ("true") parameters: search volume, CTR, CVR,
  average order value, Quality Score and a competitor ad-rank level;
* every keyword expands into one or more **search terms** (exact = 1 term,
  phrase/broad = several, including irrelevant "junk" queries that burn money);
* an analytic **GSP auction response** maps (bid, Quality Score, competition)
  to impression share, a position-driven CTR multiplier and an expected CPC
  (competitor ad ranks ~ Weibull, you pay the next rank / your QS).
  Impression share saturates (1 - exp(-x^a)), so extra bid buys less and less
  traffic while marginal CPC keeps rising -> **diminishing returns on bids**;
* seasonality (hour-of-day, day-of-week, trend, a Black-Friday style promo
  window), device effects and Gamma-Poisson / Binomial noise are layered on top.

The exact same ``SemWorld.simulate`` call is used by the historical data
generator *and* by the forward bidding simulation in ``bidding_optimizer``,
so strategies are evaluated against the same ground truth that produced the
training data.

Output: a DuckDB database (default) and optionally an identical SQLite copy.

    python -m src.data_generator --days 120 --seed 42 \
        --db data/sem.duckdb --sqlite data/sem.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc

# --------------------------------------------------------------------------- #
# Static world configuration
# --------------------------------------------------------------------------- #

DEVICES = ("desktop", "mobile", "tablet")
DEVICE_SHARE = np.array([0.36, 0.56, 0.08])  # share of search queries
DEVICE_CTR = np.array([1.00, 1.08, 0.92])
DEVICE_CVR = np.array([1.30, 0.80, 0.95])
DEVICE_AOV = np.array([1.10, 0.95, 1.00])

_H = np.arange(24)
_hour_raw = (
    0.18
    + np.exp(-(((_H - 11.0) / 3.2) ** 2))
    + 1.15 * np.exp(-(((_H - 20.5) / 2.6) ** 2))
)
HOUR_VOLUME = _hour_raw / _hour_raw.sum()
HOUR_CVR = (
    0.88
    + 0.30 * np.exp(-(((_H - 21.0) / 3.0) ** 2))
    - 0.20 * np.exp(-(((_H - 4.0) / 2.5) ** 2))
)

DOW_VOLUME = np.array([1.00, 1.03, 1.01, 0.98, 0.93, 0.82, 0.90])  # Mon..Sun
DOW_CVR = np.array([1.00, 1.00, 0.99, 0.98, 0.96, 1.04, 1.07])

MATCH_VOLUME = {"exact": 1.0, "phrase": 1.8, "broad": 3.0}

GOOD_MODIFIERS = ("best {kw}", "{kw} price", "{kw} sale", "cheap {kw}", "{kw} deals", "buy {kw}")
JUNK_MODIFIERS = (
    "{kw} repair",
    "free {kw}",
    "{kw} jobs",
    "{kw} manual pdf",
    "how to fix {kw}",
    "{kw} driver download",
    "used {kw} forum",
    "{kw} wallpaper",
)


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    ctr: float  # baseline CTR at QS ~8 and full impression share
    cvr: float  # baseline click -> conversion rate
    aov: float  # average order value (EUR)
    comp_cpc: float  # typical competitor CPC level (EUR) -> competitor ad rank
    qs_range: tuple[int, int]
    volume: float  # median daily queries for an exact keyword
    keywords: tuple[str, ...]


CAMPAIGNS: tuple[CampaignSpec, ...] = (
    CampaignSpec(
        "Brand", 0.14, 0.050, 150.0, 0.22, (8, 10), 260.0,
        ("techhub", "techhub shop", "techhub laptops", "techhub phones", "techhub coupon"),
    ),
    CampaignSpec(
        "Generic - Laptops", 0.040, 0.016, 480.0, 2.00, (4, 9), 300.0,
        (
            "gaming laptop", "business laptop", "ultrabook", "laptop 16gb ram", "student laptop",
            "laptop i7", "2 in 1 laptop", "laptop deals", "lightweight laptop", "laptop 17 inch",
            "oled laptop", "laptop for programming", "chromebook", "macbook alternative",
        ),
    ),
    CampaignSpec(
        "Generic - Phones", 0.045, 0.018, 310.0, 1.70, (4, 9), 340.0,
        (
            "android phone", "5g smartphone", "phone 256gb", "unlocked phone", "dual sim phone",
            "phone with best camera", "budget smartphone", "foldable phone", "rugged phone",
            "phone 120hz", "compact phone", "refurbished phone",
        ),
    ),
    CampaignSpec(
        "Accessories", 0.050, 0.028, 32.0, 0.30, (5, 9), 380.0,
        (
            "usb c charger", "laptop sleeve", "wireless mouse", "phone case", "screen protector",
            "usb c hub", "bluetooth earbuds", "laptop stand", "power bank", "hdmi cable",
            "mechanical keyboard", "webcam 1080p",
        ),
    ),
    CampaignSpec(
        "Competitor", 0.018, 0.007, 300.0, 1.40, (2, 5), 220.0,
        ("bigbox electronics", "gigamart laptops", "phonezone shop", "eshop tech outlet"),
    ),
)

MATCH_PLAN = ("exact", "phrase", "broad")  # cycled over keywords within a campaign


# --------------------------------------------------------------------------- #
# Auction model
# --------------------------------------------------------------------------- #


def auction_response(
    bid: np.ndarray | float,
    quality_score: np.ndarray | float,
    competitor_rank: np.ndarray | float,
    alpha: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected generalized-second-price (GSP) outcome for a max-CPC bid.

    The strongest competing ad rank in each auction is R ~ Weibull(scale=C, shape=a),
    with C = ``competitor_rank``. Our ad rank is r = bid x QS, x = r / C.

    * impression share   s = P(R < r) = 1 - exp(-x^a)          (saturating)
    * CTR multiplier     m = 0.55 + 0.45 s                      (better position)
    * expected CPC       CPC = E[R | R < r] / QS + 0.01, capped at the bid
                             = C Gamma(1+1/a) P(1+1/a, x^a) / (s QS) + 0.01

      where P is the regularised lower incomplete gamma function. You pay the
      rank of the ad below you divided by your QS, so a higher Quality Score
      directly lowers CPC, and CPC is always <= bid.

    Diminishing returns: share saturates while the *marginal* CPC of the extra
    clicks keeps rising (you start winning auctions against stronger rivals).

    Returns (impression_share, ctr_multiplier, expected_cpc).
    """
    bid = np.maximum(np.asarray(bid, dtype=float), 0.0)
    qs = np.asarray(quality_score, dtype=float)
    comp = np.asarray(competitor_rank, dtype=float)
    xa = np.power(bid * qs / comp, alpha)
    share = 1.0 - np.exp(-xa)
    k = 1.0 + 1.0 / alpha
    with np.errstate(invalid="ignore", divide="ignore"):
        paid_rank = comp * gamma_fn(k) * gammainc(k, xa) / share
    paid_rank = np.where(share > 1e-12, paid_rank, bid * qs * alpha / (alpha + 1.0))
    cpc = np.minimum(paid_rank / qs + 0.01, bid)
    ctr_mult = 0.55 + 0.45 * share
    return share, ctr_mult, cpc


def day_effects(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Calendar multipliers: day-of-week, slow trend, yearly wave, promo window."""
    dow = dates.day_of_week.to_numpy()
    t = (dates - pd.Timestamp("2025-01-01")).days.to_numpy().astype(float)
    trend = 1.0 + 0.0008 * (t - 200.0)
    yearly = 1.0 + 0.08 * np.sin(2 * np.pi * (dates.day_of_year.to_numpy() - 80) / 365.25)

    # Black-Friday / Cyber-Monday window: the Friday after the 4th Thursday of Nov.
    promo = np.zeros(len(dates))
    for i, d in enumerate(dates):
        nov1 = pd.Timestamp(year=d.year, month=11, day=1)
        thanksgiving = nov1 + pd.Timedelta(days=(3 - nov1.day_of_week) % 7 + 21)
        bf = thanksgiving + pd.Timedelta(days=1)
        delta = (d - bf).days
        if -7 <= delta <= 3:  # BF week + Cyber Monday
            promo[i] = 1.0 if delta >= -1 else 0.5
    return {
        "volume": DOW_VOLUME[dow] * trend * yearly * (1.0 + 0.75 * promo),
        "cvr": DOW_CVR[dow] * (1.0 + 0.25 * promo),
        "competition": 1.0 + 0.25 * promo,
    }


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #


@dataclass
class SemWorld:
    """Latent ground truth of the synthetic search account."""

    campaigns: pd.DataFrame
    keywords: pd.DataFrame  # one row per keyword, incl. true_* parameters
    terms: pd.DataFrame  # one row per search term (maps to keyword_id)

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, seed: int = 42, keyword_scale: float = 1.0) -> SemWorld:
        """Sample latent keyword/search-term parameters.

        ``keyword_scale`` < 1 keeps only a fraction of each campaign's keywords
        (used by the fast unit tests).
        """
        rng = np.random.default_rng(seed)
        camp_rows, kw_rows, term_rows = [], [], []
        kw_id = 0
        for c_id, spec in enumerate(CAMPAIGNS, start=1):
            camp_rows.append({"campaign_id": c_id, "campaign_name": spec.name})
            n_kw = max(2, int(round(len(spec.keywords) * keyword_scale)))
            for j, text in enumerate(spec.keywords[:n_kw]):
                kw_id += 1
                match = "exact" if spec.name == "Brand" and j == 0 else MATCH_PLAN[j % 3]
                qs = int(rng.integers(spec.qs_range[0], spec.qs_range[1] + 1))
                true_ctr = spec.ctr * (qs / 8.0) ** 0.8 * rng.lognormal(0.0, 0.20)
                true_cvr = spec.cvr * rng.lognormal(0.0, 0.45)
                true_aov = spec.aov * rng.lognormal(0.0, 0.15)
                # heavy-tailed volume: many low-volume keywords (where shrinkage matters)
                volume = spec.volume * rng.lognormal(0.0, 0.9) * MATCH_VOLUME[match]
                kw_rows.append(
                    dict(
                        keyword_id=kw_id, campaign_id=c_id, keyword_text=text, match_type=match,
                        quality_score=qs, true_ctr=true_ctr, true_cvr=true_cvr, true_aov=true_aov,
                        base_volume=volume, competitor_rank=spec.comp_cpc * 7.0,
                    )
                )
                # ---- search-term expansion
                terms = [(text, 1.0, 1.0, 1.0, False)]
                if match != "exact":
                    n_good = 2 if match == "phrase" else 3
                    n_junk = int(rng.integers(1, 3)) if match == "phrase" else int(rng.integers(2, 4))
                    for m in rng.choice(GOOD_MODIFIERS, n_good, replace=False):
                        terms.append((m.format(kw=text), rng.uniform(0.25, 0.7),
                                      rng.uniform(0.8, 1.2), rng.uniform(0.7, 1.3), False))
                    junk_share = (0.25, 0.6) if match == "phrase" else (0.4, 0.9)
                    for m in rng.choice(JUNK_MODIFIERS, n_junk, replace=False):
                        terms.append((m.format(kw=text), rng.uniform(*junk_share),
                                      rng.uniform(0.6, 1.1), rng.uniform(0.0, 0.12), True))
                shares = np.array([t[1] for t in terms])
                shares /= shares.sum()
                for (term, _, ctr_m, cvr_m, junk), share in zip(terms, shares):
                    term_rows.append(dict(keyword_id=kw_id, search_term=term, share=share,
                                          ctr_mult=ctr_m, cvr_mult=cvr_m, is_junk=junk))

        keywords = pd.DataFrame(kw_rows)
        terms = pd.DataFrame(term_rows)
        terms.insert(0, "term_id", np.arange(1, len(terms) + 1))
        # effective (term-mix weighted) keyword CTR / CVR -> what an oracle would know
        t = terms.merge(keywords[["keyword_id", "true_ctr", "true_cvr"]], on="keyword_id")
        t["ctr"] = t.true_ctr * t.ctr_mult
        t["clk"] = t.share * t.ctr
        t["conv"] = t.clk * t.true_cvr * t.cvr_mult
        eff = t.groupby("keyword_id").agg(clk=("clk", "sum"), conv=("conv", "sum"))
        dev_ctr = (DEVICE_SHARE * DEVICE_CTR).sum()
        dev_cvr = (DEVICE_SHARE * DEVICE_CTR * DEVICE_CVR).sum() / dev_ctr
        hour_cvr = (HOUR_VOLUME * HOUR_CVR).sum()
        keywords["eff_ctr"] = (eff.clk * dev_ctr).to_numpy()
        keywords["eff_cvr"] = (eff.conv / eff.clk * dev_cvr * hour_cvr).to_numpy()
        return cls(pd.DataFrame(camp_rows), keywords, terms)

    # ------------------------------------------------------------- simulate
    def simulate(
        self,
        dates: pd.DatetimeIndex,
        bids: np.ndarray,
        rng: np.random.Generator,
        grain: str = "hourly",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the auction for ``dates`` x keywords with ``bids[day, keyword]``.

        grain="hourly": returns (keyword x date x hour x device, search term x date)
        grain="daily":  returns (keyword x date, search term x date)  -- faster
        """
        kw = self.keywords
        terms = self.terms
        n_d, n_kw = len(dates), len(kw)
        assert bids.shape == (n_d, n_kw), f"bids shape {bids.shape} != {(n_d, n_kw)}"

        kw_index = {k: i for i, k in enumerate(kw.keyword_id)}
        t_kw = terms.keyword_id.map(kw_index).to_numpy()  # term -> keyword position
        eff = day_effects(dates)

        # (day, keyword) level: demand noise (Gamma-Poisson) + competition noise
        demand_noise = rng.gamma(25.0, 1.0 / 25.0, size=(n_d, n_kw))
        comp = (
            kw.competitor_rank.to_numpy()[None, :]
            * eff["competition"][:, None]
            * rng.lognormal(0.0, 0.12, size=(n_d, n_kw))
        )
        share, pos_mult, cpc = auction_response(bids, kw.quality_score.to_numpy()[None, :], comp)

        # broadcast to (day, term, hour, device)
        queries = (
            kw.base_volume.to_numpy()[None, t_kw]
            * eff["volume"][:, None]
            * demand_noise[:, t_kw]
            * terms.share.to_numpy()[None, :]
        )[:, :, None, None] * (HOUR_VOLUME[:, None] * DEVICE_SHARE[None, :])[None, None]
        impressions = rng.poisson(queries * share[:, t_kw][:, :, None, None])

        ctr = (
            kw.true_ctr.to_numpy()[t_kw] * terms.ctr_mult.to_numpy()
        )[None, :, None, None] * pos_mult[:, t_kw][:, :, None, None] * DEVICE_CTR[None, None, None, :]
        clicks = rng.binomial(impressions, np.clip(ctr, 0.0, 0.9))

        cvr = (
            (kw.true_cvr.to_numpy()[t_kw] * terms.cvr_mult.to_numpy())[None, :, None, None]
            * eff["cvr"][:, None, None, None]
            * HOUR_CVR[None, None, :, None]
            * DEVICE_CVR[None, None, None, :]
        )
        conversions = rng.binomial(clicks, np.clip(cvr, 0.0, 0.95))

        aov = kw.true_aov.to_numpy()[t_kw][None, :, None, None] * DEVICE_AOV[None, None, None, :]
        shape = np.where(conversions > 0, conversions * 4.0, 1.0)
        conv_value = np.where(conversions > 0, rng.gamma(shape, aov / 4.0), 0.0)

        unit_cpc = cpc[:, t_kw][:, :, None, None] * rng.lognormal(0.0, 0.05, size=clicks.shape)
        unit_cpc = np.minimum(unit_cpc, bids[:, t_kw][:, :, None, None])
        cost = clicks * unit_cpc

        # ---------- flatten to long format
        D, T, H, V = clicks.shape
        idx = np.indices((D, T, H, V)).reshape(4, -1)
        frame = pd.DataFrame(
            {
                "d": idx[0], "t": idx[1], "hour": idx[2], "v": idx[3],
                "impressions": impressions.ravel(), "clicks": clicks.ravel(),
                "cost": cost.ravel(), "conversions": conversions.ravel(),
                "conv_value": conv_value.ravel(),
            }
        )
        frame["k"] = t_kw[frame.t.to_numpy()]
        metrics = ["impressions", "clicks", "cost", "conversions", "conv_value"]

        # search-term x day
        st = frame.groupby(["d", "t"], sort=True)[metrics].sum().reset_index()
        st["date"] = dates[st.d.to_numpy()]
        st["term_id"] = terms.term_id.to_numpy()[st.t.to_numpy()]
        st["keyword_id"] = terms.keyword_id.to_numpy()[st.t.to_numpy()]
        st["search_term"] = terms.search_term.to_numpy()[st.t.to_numpy()]

        keys = ["d", "k"] if grain == "daily" else ["d", "k", "hour", "v"]
        kf = frame.groupby(keys, sort=True)[metrics].sum().reset_index()
        kf["date"] = dates[kf.d.to_numpy()]
        kf["keyword_id"] = kw.keyword_id.to_numpy()[kf.k.to_numpy()]
        kf["campaign_id"] = kw.campaign_id.to_numpy()[kf.k.to_numpy()]
        kf["max_cpc"] = bids[kf.d.to_numpy(), kf.k.to_numpy()]
        if grain != "daily":
            kf["device"] = np.array(DEVICES)[kf.v.to_numpy()]
        st["campaign_id"] = st.keyword_id.map(dict(zip(kw.keyword_id, kw.campaign_id)))
        return kf.drop(columns=[c for c in ("d", "k", "v") if c in kf]), st.drop(columns=["d", "t"])


# --------------------------------------------------------------------------- #
# Historical dataset
# --------------------------------------------------------------------------- #


def historical_bids(world: SemWorld, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Plausible human-managed bid history: start near the competitor CPC,
    then change bids weekly (mean-reverting AR(1) in log space; gives the bid-landscape
    variation that forecasting needs)."""
    kw = world.keywords
    comp_cpc = kw.competitor_rank.to_numpy() / 7.0
    start = comp_cpc * rng.uniform(0.6, 1.5, size=len(kw))
    n_weeks = int(np.ceil(n_days / 7))
    # mean-reverting AR(1) in log-space: managers nudge bids up/down weekly
    # but stay around their "intended" level instead of drifting away.
    log_dev = np.zeros((n_weeks, len(kw)))
    for w in range(1, n_weeks):
        log_dev[w] = 0.6 * log_dev[w - 1] + rng.normal(0.0, 0.22, size=len(kw))
    weekly = start[None, :] * np.exp(log_dev)
    weekly = np.clip(weekly, 0.05, 6.0)
    return np.round(np.repeat(weekly, 7, axis=0)[:n_days], 2)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-computed calendar columns keep the SQL layer dialect-neutral
    (no DuckDB-vs-SQLite date function differences)."""
    d = pd.to_datetime(df["date"])
    df["date"] = d.dt.strftime("%Y-%m-%d")
    df["week_start"] = (d - pd.to_timedelta(d.dt.day_of_week, unit="D")).dt.strftime("%Y-%m-%d")
    df["day_of_week"] = d.dt.day_of_week + 1  # 1 = Monday
    return df


def generate(
    n_days: int = 120,
    start: str = "2025-08-04",
    seed: int = 42,
    keyword_scale: float = 1.0,
    chunk_days: int = 14,
) -> dict[str, pd.DataFrame]:
    """Generate all tables in memory. Returns {table_name: DataFrame}."""
    rng = np.random.default_rng(seed)
    world = SemWorld.build(seed=seed, keyword_scale=keyword_scale)
    dates = pd.date_range(start, periods=n_days, freq="D")
    bids = historical_bids(world, n_days, rng)

    hourly, terms_daily = [], []
    for s in range(0, n_days, chunk_days):
        sl = slice(s, min(s + chunk_days, n_days))
        kf, st = world.simulate(dates[sl], bids[sl], rng, grain="hourly")
        hourly.append(kf)
        terms_daily.append(st)

    hourly_df = _add_calendar(pd.concat(hourly, ignore_index=True))
    hourly_df["cost"] = hourly_df["cost"].round(2)
    hourly_df["conv_value"] = hourly_df["conv_value"].round(2)
    hourly_df = hourly_df[
        ["date", "week_start", "day_of_week", "hour", "device", "campaign_id", "keyword_id",
         "max_cpc", "impressions", "clicks", "cost", "conversions", "conv_value"]
    ]
    st_df = _add_calendar(pd.concat(terms_daily, ignore_index=True))
    st_df["cost"] = st_df["cost"].round(2)
    st_df["conv_value"] = st_df["conv_value"].round(2)
    st_df = st_df[
        ["date", "week_start", "campaign_id", "keyword_id", "term_id", "search_term",
         "impressions", "clicks", "cost", "conversions", "conv_value"]
    ]

    kw = world.keywords
    meta = pd.DataFrame(
        [{"seed": seed, "start_date": str(dates[0].date()), "n_days": n_days,
          "keyword_scale": keyword_scale, "generator_version": "1.0"}]
    )
    return {
        "meta": meta,
        "dim_campaign": world.campaigns,
        "dim_keyword": kw[["keyword_id", "campaign_id", "keyword_text", "match_type", "quality_score"]],
        "fact_keyword_hourly": hourly_df,
        "fact_search_term_daily": st_df,
        # Ground truth: used ONLY by the simulator / evaluation, never by strategies.
        "truth_keyword": kw[["keyword_id", "true_ctr", "true_cvr", "true_aov", "eff_ctr", "eff_cvr",
                             "base_volume", "competitor_rank"]],
        "truth_search_term": world.terms,
    }


def write_duckdb(tables: dict[str, pd.DataFrame], path: str | Path) -> None:
    import duckdb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        for name, df in tables.items():
            con.register("_tmp", df)
            if "date" in df.columns:
                cols = ", ".join(
                    f"CAST({c} AS DATE) AS {c}" if c in ("date", "week_start") else c for c in df.columns
                )
                con.execute(f"CREATE TABLE {name} AS SELECT {cols} FROM _tmp")
            else:
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp")
            con.unregister("_tmp")
    finally:
        con.close()


def write_sqlite(tables: dict[str, pd.DataFrame], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        for name, df in tables.items():
            df.to_sql(name, con, index=False, chunksize=50_000)
        con.execute("CREATE INDEX ix_hourly_kw ON fact_keyword_hourly(keyword_id, date)")
        con.execute("CREATE INDEX ix_terms_kw ON fact_search_term_daily(keyword_id, date)")
        con.commit()
    finally:
        con.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--start", default="2025-08-04")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db", default="data/sem.duckdb")
    p.add_argument("--sqlite", default=None, help="optional SQLite copy, e.g. data/sem.sqlite")
    a = p.parse_args(argv)

    tables = generate(n_days=a.days, start=a.start, seed=a.seed)
    write_duckdb(tables, a.db)
    print(f"DuckDB  -> {a.db}")
    if a.sqlite:
        write_sqlite(tables, a.sqlite)
        print(f"SQLite  -> {a.sqlite}")
    for name, df in tables.items():
        print(f"  {name:<24} {len(df):>9,} rows")


if __name__ == "__main__":
    main()
