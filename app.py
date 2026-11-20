"""Interactive Streamlit dashboard.

    streamlit run app.py

Falls back gracefully: if Streamlit is not installed, run
    python -m src.pipeline --report outputs/report.html
to get the same visuals as a self-contained HTML report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("Streamlit is not installed. Use: python -m src.pipeline --report outputs/report.html")

from src import bidding_optimizer as bo
from src import charts
from src import data_generator as dg
from src import stats_engine as se
from src.sql_runner import connect, list_queries, run_query

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "sem.duckdb"

st.set_page_config(page_title="SEM Bidding Simulator", layout="wide")

# The app is public (sem.savicdev.com, mem_limit 1536m). Every distinct widget combination is a new
# cache entry, so all caches are bounded: without max_entries a few curious visitors can grow the
# process until the container is OOM-killed.
CACHE = {"max_entries": 32, "ttl": 3600}


def ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def pct(x: float, fmt: str = "+.2%") -> str:
    return "n/a" if not np.isfinite(x) else format(x, fmt)


@st.cache_resource(show_spinner="Generating synthetic SEM data ...")
def ensure_db(seed: int) -> str:
    if not DB.exists():
        dg.write_duckdb(dg.generate(seed=seed), DB)
    return str(DB)


@st.cache_data(show_spinner=False, max_entries=64)
def query(name: str, db: str) -> pd.DataFrame:
    con = connect(db)
    try:
        return run_query(name, con)
    finally:
        con.close()


@st.cache_data(show_spinner=False, max_entries=2)
def history(db: str) -> tuple[pd.DataFrame, dict]:
    con = connect(db)
    try:
        return bo.load_daily_history(con), bo.load_meta(con)
    finally:
        con.close()


@st.cache_data(show_spinner="Running bandit simulation ...", **CACHE)
def bandits(rates: tuple[float, ...], rounds: int, sims: int, value: float):
    return se.simulate_bandits(list(rates), n_rounds=rounds, n_sims=sims, value_per_conversion=value)


@st.cache_data(show_spinner="Simulating the bidding strategies ...", **CACHE)
def bidding(db: str, target_roas: float, margin: float, weeks: int, n_seeds: int, history_days: int):
    hist, meta = history(db)
    days = sorted(hist.date.unique())
    hist = hist[hist.date >= days[-min(history_days, len(days))]]  # the DB may hold fewer days
    world = dg.SemWorld.build(seed=int(meta["seed"]), keyword_scale=float(meta["keyword_scale"]))
    sim = bo.compare_strategies(world, hist, bo.default_strategies(world, target_roas), target_roas=target_roas,
                                n_weeks=weeks, seeds=tuple(range(n_seeds)), margin=margin)
    return sim.summary, sim.daily, bo.cvr_estimation_error(hist, world)


db = ensure_db(42)
st.title("SEM Bidding Simulator")
st.caption("Synthetic search-ads account - Python + SQL (DuckDB) + statistics. "
           "All numbers are simulated and illustrate the methods; they are not client results.")

tab_sql, tab_ab, tab_bandit, tab_bid = st.tabs(
    ["Campaign analytics (SQL)", "A/B testing", "Multi-armed bandits", "Bid optimisation"]
)

# --------------------------------------------------------------------- SQL
with tab_sql:
    kpi = query("01_campaign_kpis", db)
    waste = query("05_wasted_spend_search_terms", db)
    c = st.columns(4)
    c[0].metric("Spend", f"{kpi.cost.sum():,.0f} EUR")
    c[1].metric("Conversion value", f"{kpi.conv_value.sum():,.0f} EUR")
    c[2].metric("ROAS", f"{ratio(kpi.conv_value.sum(), kpi.cost.sum()):.2f}")
    c[3].metric("Wasted spend flagged", f"{waste.wasted_cost.sum():,.0f} EUR",
                f"{pct(ratio(waste.wasted_cost.sum(), kpi.cost.sum()), '.1%')} of spend", delta_color="off")
    st.dataframe(kpi, hide_index=True, width="stretch")
    l, r = st.columns(2)
    l.plotly_chart(charts.rolling_roas(query("02_daily_rolling_trends", db)), width="stretch")
    r.plotly_chart(charts.roas_quartiles(query("03_keyword_roas_cohorts", db)), width="stretch")
    st.plotly_chart(charts.wasted_spend_bar(waste), width="stretch")
    st.plotly_chart(charts.hour_device_heatmap(query("06_hour_device_bid_modifiers", db)), width="stretch")
    with st.expander("Run any query from sql/"):
        name = st.selectbox("Query", list_queries())
        st.code((ROOT / "sql" / f"{name}.sql").read_text(), language="sql")
        st.dataframe(query(name, db), hide_index=True, width="stretch")

# --------------------------------------------------------------------- A/B
with tab_ab:
    st.subheader("Ad copy / landing page experiment")
    c = st.columns(4)
    n_a = c[0].number_input("Visitors A", 100, 10_000_000, 40_000, step=1_000)
    x_a = c[1].number_input("Conversions A", 0, 10_000_000, 1_580)
    n_b = c[2].number_input("Visitors B", 100, 10_000_000, 40_000, step=1_000)
    x_b = c[3].number_input("Conversions B", 0, 10_000_000, 1_720)
    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
    if x_a > n_a or x_b > n_b:
        st.error("Conversions cannot exceed visitors.")
    else:
        z = se.two_proportion_ztest(int(x_a), int(n_a), int(x_b), int(n_b), alpha=alpha)
        b = se.bayesian_ab_test(int(x_a), int(n_a), int(x_b), int(n_b))
        srm = se.srm_check(int(n_a), int(n_b))
        m = st.columns(5)
        m[0].metric("Relative lift", pct(z.rel_lift),
                    None if np.isfinite(z.rel_lift) else "A has 0 conversions", delta_color="off")
        m[1].metric("p-value", f"{z.p_value:.4f}", "significant" if z.significant else "not significant",
                    delta_color="normal" if z.significant else "off")
        m[2].metric(f"{1 - alpha:.0%} CI (abs. diff)", f"[{z.ci_low:+.4f}, {z.ci_high:+.4f}]")
        m[3].metric("P(B beats A)", f"{b.prob_b_beats_a:.1%}")
        m[4].metric("Expected loss if ship B", f"{b.expected_loss_choose_b:.5f}")
        if srm["srm_detected"]:
            st.warning(f"Sample ratio mismatch (p = {srm['p_value']:.2g}): check randomisation before trusting this test.")
        l, r = st.columns(2)
        l.plotly_chart(charts.posterior_plot(b.posterior_a, b.posterior_b, metric="CVR"), width="stretch")
        r.plotly_chart(charts.lift_distribution(b.posterior_a, b.posterior_b), width="stretch")

    st.subheader("Sample size planner")
    c = st.columns(3)
    base = c[0].number_input("Baseline rate", 0.001, 0.9, 0.04, 0.005, format="%.3f")
    mde = c[1].number_input("Minimum detectable effect (relative)", 0.01, 2.0, 0.10, 0.01)
    power = c[2].slider("Power", 0.5, 0.99, 0.80, 0.01)
    if base * (1 + mde) >= 1:
        st.error(f"Baseline x (1 + MDE) = {base * (1 + mde):.0%}: the target rate must stay below 100%.")
    else:
        n_req = se.required_sample_size(base, mde, alpha=alpha, power=power)
        st.info(f"**{n_req:,} visitors per arm** ({2 * n_req:,} total) to detect {base:.2%} -> "
                f"{base * (1 + mde):.2%} with alpha={alpha}, power={power:.0%}.")

# --------------------------------------------------------------------- bandit
with tab_bandit:
    st.subheader("Thompson Sampling & UCB1 vs static equal split")
    c = st.columns(4)
    rates_txt = c[0].text_input("True CVR per variant", "0.030, 0.034, 0.040")
    rounds = c[1].select_slider("Visitors", [2_000, 5_000, 10_000, 20_000, 50_000], 20_000)
    sims = c[2].select_slider("Replications", [20, 50, 100, 200], 100)
    value = c[3].number_input("Value per conversion (EUR)", 1.0, 10_000.0, 60.0)
    try:
        rates = tuple(float(x) for x in rates_txt.split(","))
        if not (2 <= len(rates) <= 10 and all(0 < p < 1 for p in rates)):
            raise ValueError
    except ValueError:  # (not assert: asserts vanish under python -O)
        st.error("Enter 2 to 10 comma-separated rates between 0 and 1.")
    else:
        res = bandits(rates, rounds, sims, value)
        st.dataframe(pd.DataFrame(se.bandit_summary(res)), hide_index=True, width="stretch")
        l, r = st.columns(2)
        l.plotly_chart(charts.regret_curves(res, every=max(1, rounds // 400)), width="stretch")
        r.plotly_chart(charts.arm_allocation(res, rates), width="stretch")
        st.caption("UCB1's exploration bonus sqrt(2 ln t / n) is large relative to conversion rates of a few "
                   "percent, so it keeps exploring for a long time; Thompson Sampling adapts to the actual "
                   "posterior uncertainty and concentrates traffic much faster.")

# --------------------------------------------------------------------- bidding
with tab_bid:
    st.subheader("Expected-value bidding with Bayesian shrinkage vs rule-based bidding")
    c = st.columns(5)
    margin = c[0].slider("Gross margin", 0.10, 0.80, 0.35, 0.05)
    target = c[1].number_input("Target ROAS", 1.0, 20.0, round(1 / margin, 2), 0.25,
                               help="Default = break-even ROAS (1 / margin), profit-maximising in a second-price auction.")
    weeks = c[2].slider("Weeks simulated", 1, 8, 4)
    n_seeds = c[3].slider("Seeds", 1, 8, 3)
    hist_days = c[4].select_slider("History used (days)", [14, 28, 56, 120], 120)
    summary, daily, err = bidding(db, float(target), float(margin), int(weeks), int(n_seeds), int(hist_days))
    st.dataframe(summary, hide_index=True, width="stretch")
    l, r = st.columns(2)
    l.plotly_chart(charts.strategy_bars(summary, "roas", "ROAS by strategy", ".2f"), width="stretch")
    r.plotly_chart(charts.strategy_bars(summary, "profit", "Profit by strategy (EUR)"), width="stretch")
    st.plotly_chart(charts.strategy_daily(daily), width="stretch")
    st.markdown("**CVR estimation error vs ground truth** (where shrinkage pays off)")
    st.dataframe(err, hide_index=True, width="stretch")

    st.subheader("Bid landscape forecast")
    weekly = query("07_keyword_bid_landscape", db)
    hist, _ = history(db)
    ks = bo.keyword_summary(hist).set_index("keyword_id")
    names = weekly.drop_duplicates("keyword_id").set_index("keyword_id").keyword_text
    kid = st.selectbox("Keyword", names.index, format_func=lambda k: f"{names[k]} (#{k})")
    land = bo.BidLandscape.fit(weekly, kid, ks.pcvr[kid], ks.conv_value[kid] / max(ks.conversions[kid], 1))
    fc = land.forecast(np.linspace(0.05, 1.5 * land.bid_max_observed, 200))
    cur = float(weekly[weekly.keyword_id == kid].max_cpc.iloc[-1])
    best = land.best_bid(margin=margin)
    l, r = st.columns(2)
    l.plotly_chart(charts.landscape_plot(fc, cur, best), width="stretch")
    r.plotly_chart(charts.landscape_profit(fc, margin), width="stretch")
    st.caption(f"Observed bids {land.bid_min_observed:.2f}-{land.bid_max_observed:.2f} EUR. "
               "Forecasts outside that range are extrapolations.")
