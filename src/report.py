"""Self-contained HTML report (no Streamlit needed).

    python -m src.pipeline --report outputs/report.html
    python -m src.report                       # runs the pipeline, writes outputs/report.html
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline

from . import bidding_optimizer as bo
from . import charts

CSS = """
:root{--bg:#f7f7f5;--surface:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--line:#e6e5e0;--accent:#2a78d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
p.lead{color:var(--muted);margin:0 0 20px}.note{color:var(--muted);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:24px;font-weight:650;margin-top:2px}.tile .s{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px;margin:12px 0}
.grid2>*{min-width:0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:6px;overflow:hidden;min-width:0;margin:12px 0}
.grid2 .card{margin:0}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface)}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin:8px 0}
code{background:#efeee9;padding:1px 5px;border-radius:4px;font-size:13px}
"""


RESIZE_JS = """<script>
function _fit(){document.querySelectorAll('.js-plotly-plot').forEach(function(d){Plotly.Plots.resize(d)})}
window.addEventListener('load',_fit);window.addEventListener('resize',_fit);
</script>"""


def _tile(k: str, v: str, s: str = "") -> str:
    return f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>' \
           f'<div class="s">{html.escape(s)}</div></div>'


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: "" if pd.isna(v) else format(v, f))
    return '<div class="tbl">' + d.to_html(index=False, escape=True, border=0, na_rep="") + "</div>"


def _fig(fig) -> str:
    return '<div class="card">' + fig.to_html(full_html=False, include_plotlyjs=False,
                                               config={"displaylogo": False, "responsive": True}) + "</div>"


def pick_landscape_keyword(weekly: pd.DataFrame) -> int:
    """Keyword with the most bid variation among the higher-spend ones."""
    g = weekly.groupby("keyword_id").agg(cost=("cost", "sum"), bid_cv=("max_cpc", lambda s: s.std() / s.mean()))
    g = g[g.cost >= g.cost.quantile(0.6)]
    return int(g.bid_cv.idxmax())


def build_report(results: dict, path: str | Path, inline_js: bool = True) -> Path:
    s = results["summary"]
    sql = results["sql"]
    sim = results["bidding"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kpi = sql["01_campaign_kpis"]
    tot_cost, tot_val = kpi.cost.sum(), kpi.conv_value.sum()
    ab_f, ab_b = s["ab_test"]["frequentist"], s["ab_test"]["bayesian"]
    bsum = sim.summary.set_index("strategy")
    rb, ev = bsum.loc["rule_based"], bsum.loc["ev_shrinkage"]
    bt = {r["policy"]: r for r in s["bandits"]["table"]}
    paired = s["bidding"]["paired_ev_shrinkage_vs_rules"]

    # landscape example
    weekly = sql["07_keyword_bid_landscape"]
    ks = bo.keyword_summary(results["history"]).set_index("keyword_id")
    kid = pick_landscape_keyword(weekly)
    aov = ks.conv_value[kid] / max(ks.conversions[kid], 1)
    land = bo.BidLandscape.fit(weekly, kid, ks.pcvr[kid], aov)
    grid = np.linspace(0.05, 1.5 * land.bid_max_observed, 200)
    fc = land.forecast(grid)
    cur_bid = float(weekly[weekly.keyword_id == kid].max_cpc.iloc[-1])
    best = land.best_bid(margin=s["bidding"]["margin"])
    kw_text = weekly.loc[weekly.keyword_id == kid, "keyword_text"].iloc[0]

    parts = [
        "<h1>SEM Bidding Simulator - sample run</h1>",
        f'<p class="lead">Synthetic search account, {s["data"]["keywords"]} keywords, '
        f'{s["data"]["date_from"]} to {s["data"]["date_to"]}. Generated {s["generated_at"]}. '
        "<b>All data are simulated</b>: the numbers demonstrate the methods, they are not client results.</p>",
        '<div class="tiles">',
        _tile("Spend (history)", f"{tot_cost:,.0f} EUR", f"ROAS {tot_val / tot_cost:.2f}"),
        _tile("Wasted spend found", f"{s['sql']['wasted_spend_eur']:,.0f} EUR",
              f"{s['sql']['wasted_spend_share']:.1%} of spend, {s['sql']['wasted_terms']} search terms"),
        _tile("A/B: P(B beats A)", f"{ab_b['prob_b_beats_a']:.1%}",
              f"p = {ab_f['p_value']:.4f}, lift {ab_f['rel_lift']:+.1%}"),
        _tile("Thompson vs static split", f"{bt['thompson']['revenue_lift_vs_static']:+.1%}",
              f"revenue; regret {bt['thompson']['final_cum_regret']:.0f} vs {bt['static_split']['final_cum_regret']:.0f}"),
        _tile("EV+shrinkage vs rules", f"{ev.roas:.2f} vs {rb.roas:.2f}",
              f"ROAS; profit {ev.profit / rb.profit - 1:+.1%} ({paired['seeds_ev_wins_profit']}/{paired['n_seeds']} seeds)"),
        "</div>",
        # ---------------- SQL
        "<h2>1. Campaign analytics (SQL layer)</h2>",
        '<p class="note">Every chart and table here is the output of a file in <code>sql/</code>, '
        "run unchanged on DuckDB (and verified to return identical results on SQLite).</p>",
        _table(kpi, {"impressions": ",.0f", "clicks": ",.0f", "cost": ",.0f", "conversions": ",.0f",
                     "conv_value": ",.0f", "ctr": ".2%", "cpc": ".2f", "cvr": ".2%", "cpa": ".2f",
                     "roas": ".2f", "share_of_spend": ".1%"}),
        '<div class="grid2">',
        _fig(charts.rolling_roas(sql["02_daily_rolling_trends"])),
        _fig(charts.roas_quartiles(sql["03_keyword_roas_cohorts"])),
        "</div>",
        _fig(charts.wasted_spend_bar(sql["05_wasted_spend_search_terms"])),
        _fig(charts.hour_device_heatmap(sql["06_hour_device_bid_modifiers"])),
        # ---------------- A/B
        "<h2>2. A/B test - ad copy CTR (frequentist + Bayesian)</h2>",
        f'<p class="note">Simulated experiment: true CTR {s["ab_test"]["true_ctr"][0]:.2%} vs '
        f'{s["ab_test"]["true_ctr"][1]:.2%}, {s["ab_test"]["data"]["imps_a"]:,} impressions per arm. '
        f'Minimum sample size for this {s["ab_test"]["true_ctr"][1] / s["ab_test"]["true_ctr"][0] - 1:.0%} '
        f'relative MDE (alpha 0.05, power 0.80): <b>{s["ab_test"]["required_n_per_arm"]:,} per arm</b>. '
        f'SRM check p = {s["ab_test"]["srm"]["p_value"]:.3f}.</p>',
        _table(pd.DataFrame([{
            "CTR A": ab_f["rate_a"], "CTR B": ab_f["rate_b"], "rel. lift": ab_f["rel_lift"], "z": ab_f["z"],
            "p-value": ab_f["p_value"], "95% CI (B-A)": f"[{ab_f['ci_low']:+.4f}, {ab_f['ci_high']:+.4f}]",
            "post-hoc power": ab_f["power"], "P(B>A)": ab_b["prob_b_beats_a"],
            "exp. loss if ship B": ab_b["expected_loss_choose_b"],
        }]), {"CTR A": ".3%", "CTR B": ".3%", "rel. lift": "+.1%", "z": ".2f", "p-value": ".4f",
              "post-hoc power": ".2f", "P(B>A)": ".3f", "exp. loss if ship B": ".6f"}),
        '<div class="grid2">',
        _fig(charts.posterior_plot(tuple(ab_b["posterior_a"]), tuple(ab_b["posterior_b"]))),
        _fig(charts.lift_distribution(tuple(ab_b["posterior_a"]), tuple(ab_b["posterior_b"]))),
        "</div>",
        # ---------------- bandits
        "<h2>3. Multi-armed bandits - dynamic budget allocation</h2>",
        f'<p class="note">{len(s["bandits"]["true_rates"])} ad variants with true CVR '
        f'{", ".join(f"{p:.1%}" for p in s["bandits"]["true_rates"])}; {s["bandits"]["rounds"]:,} visitors, '
        f'{s["bandits"]["sims"]} Monte-Carlo replications.</p>',
        _table(pd.DataFrame(s["bandits"]["table"]).drop(columns=["arm_share"]),
               {"final_cum_regret": ",.1f", "total_revenue": ",.0f", "revenue_sd": ",.0f",
                "revenue_lift_vs_static": "+.1%", "best_arm_share": ".1%"}),
        '<div class="grid2">',
        _fig(charts.regret_curves(results["bandits"])),
        _fig(charts.arm_allocation(results["bandits"], s["bandits"]["true_rates"])),
        "</div>",
        # ---------------- bidding
        "<h2>4. Bid optimisation - EV bidding with Bayesian shrinkage vs rules</h2>",
        f'<p class="note">Walk-forward simulation: {s["bidding"]["forward_weeks"]} weeks, weekly re-bidding, '
        f'{len(s["bidding"]["seeds"])} seeds with common random numbers. Target ROAS '
        f'{s["bidding"]["target_roas"]:.2f} (= break-even at {s["bidding"]["margin"]:.0%} margin); '
        "profit = value x margin - cost. Error bars = SD across seeds.</p>",
        _table(sim.summary, {"cost": ",.0f", "conversions": ",.1f", "conv_value": ",.0f", "roas": ".2f",
                             "roas_sd": ".2f", "cpa": ".2f", "profit": ",.0f", "profit_sd": ",.0f",
                             "clicks": ",.0f"}),
        '<div class="grid2">',
        _fig(charts.strategy_bars(sim.summary, "roas", "ROAS by strategy", ".2f")),
        _fig(charts.strategy_bars(sim.summary, "profit", "Profit by strategy (EUR)")),
        "</div>",
        _fig(charts.strategy_daily(sim.daily)),
        "<p class='note'>Where shrinkage helps: mean absolute error of the CVR estimate vs the simulator's "
        "true CVR, using only the last 28 days of history.</p>",
        _table(results["est_err_28"], {"mae_raw": ".4f", "mae_shrunk": ".4f", "mean_true_cvr": ".4f",
                                       "error_reduction": "+.1%"}),
        f"<h2>5. Bid landscape forecast - \"{html.escape(kw_text)}\"</h2>",
        f'<p class="note">Fitted on {int((weekly.keyword_id == kid).sum())} weekly observations '
        f"(bids {land.bid_min_observed:.2f}-{land.bid_max_observed:.2f} EUR). clicks(b) = {land.a:.0f}(1-e^(-{land.c:.2f}b)); "
        f"pCVR {land.pcvr:.2%}, AOV {land.aov:,.0f} EUR. Current bid {cur_bid:.2f}, forecast profit-max bid "
        f"<b>{best:.2f}</b>"
        + (" - below the observed bid range, so treat it as a direction (bid down) and test it in steps."
           if best < land.bid_min_observed else ".") + "</p>",
        '<div class="grid2">',
        _fig(charts.landscape_plot(fc, cur_bid, best)),
        _fig(charts.landscape_profit(fc, s["bidding"]["margin"])),
        "</div>",
        f'<p class="note">Runtime {s["runtime_seconds"]}s. Source: sem-bidding-simulator.</p>',
    ]
    js = (f"<script>{plotly.offline.get_plotlyjs()}</script>" if inline_js
          else '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>')
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>SEM Bidding Simulator</title><style>{CSS}</style>{js}</head>"
        f"<body><main>{''.join(parts)}</main>{RESIZE_JS}</body></html>"
    )
    path.write_text(doc, encoding="utf-8")
    return path


if __name__ == "__main__":
    from .pipeline import RunConfig, run

    res = run(RunConfig())
    print(build_report(res, RunConfig().out_dir / "report.html"))
