"""Plotly figure builders shared by the Streamlit app and the HTML report.

Colour rules: categorical hues in a fixed order (colour follows the entity,
never its rank), one diverging blue<->red scale with a neutral grey midpoint
for indices around 1.0, thin 2px lines, one y-axis per chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]

STRATEGY_COLORS = {
    "static_bids": "#8a8984",
    "rule_based": SERIES[1],
    "ev_raw_cvr": SERIES[3],
    "ev_shrinkage": SERIES[0],
    "oracle_true_value": SERIES[2],
}
STRATEGY_LABELS = {
    "static_bids": "Static bids (control)",
    "rule_based": "Rule-based",
    "ev_raw_cvr": "EV bidding, raw CVR",
    "ev_shrinkage": "EV bidding + Bayesian shrinkage",
    "oracle_true_value": "Oracle (true value/click)",
}
CAMPAIGN_COLORS = {"Generic - Laptops": SERIES[0], "Generic - Phones": SERIES[1], "Accessories": SERIES[2],
                   "Competitor": SERIES[3], "Brand": SERIES[4]}
POLICY_LABELS = {"static_split": "Static equal split", "thompson": "Thompson Sampling", "ucb1": "UCB1"}
POLICY_COLORS = {"static_split": "#8a8984", "thompson": SERIES[0], "ucb1": SERIES[1]}


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    margin = kw.pop("margin", {"l": 56, "r": 16, "t": 56, "b": 48})
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": TEXT}},
        template="plotly_white",
        height=height,
        margin=margin,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Inter, system-ui, -apple-system, Segoe UI, sans-serif", "color": MUTED, "size": 12},
        legend=kw.pop("legend", {"orientation": "h", "y": -0.18, "x": 0}),
        hoverlabel={"bgcolor": "white", "font": {"color": TEXT}},
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# ----------------------------------------------------------------- SQL views
def rolling_roas(daily: pd.DataFrame, exclude: tuple[str, ...] = ("Brand",)) -> go.Figure:
    """Non-brand campaigns (brand ROAS is an order of magnitude higher and
    would flatten everything else on a shared linear axis)."""
    fig = go.Figure()
    for i, (name, g) in enumerate(daily.groupby("campaign_name", sort=True)):
        if name in exclude:
            continue
        g = g.dropna(subset=["rolling_roas_7d"])
        fig.add_trace(go.Scatter(x=g.date, y=g.rolling_roas_7d, name=name, mode="lines",
                                 line={"width": 2, "color": CAMPAIGN_COLORS.get(name, SERIES[6])},
                                 hovertemplate="%{x}<br>7d ROAS %{y:.2f}<extra>" + name + "</extra>"))
    fig.update_yaxes(title="7-day rolling ROAS", rangemode="tozero")
    title = "7-day rolling ROAS by campaign" + (f" (excl. {', '.join(exclude)})" if exclude else "")
    return _layout(fig, title, hovermode="x unified")


def wasted_spend_bar(waste: pd.DataFrame, top: int = 12) -> go.Figure:
    d = waste.head(top).iloc[::-1]
    color = np.where(d.waste_flag == "ZERO_CONV", SERIES[7], SERIES[1])
    fig = go.Figure(go.Bar(
        x=d.wasted_cost, y=d.search_term, orientation="h", marker={"color": color},
        customdata=np.stack([d.waste_flag, d.cost, d.conversions, d.campaign_name], axis=1),
        hovertemplate="%{y}<br>%{customdata[3]}<br>%{customdata[0]}: wasted %{x:,.0f} EUR"
                      "<br>cost %{customdata[1]:,.0f} EUR, %{customdata[2]} conv<extra></extra>",
    ))
    fig.update_xaxes(title="Wasted spend (EUR)")
    return _layout(fig, f"Top {top} wasted-spend search terms (red = zero conversions, orange = CPA > 3x)",
                   height=440, margin={"l": 260, "r": 16, "t": 56, "b": 48})


def hour_device_heatmap(mods: pd.DataFrame) -> go.Figure:
    pv = mods.pivot(index="device", columns="hour", values="vpc_index")
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=pv.columns, y=pv.index, colorscale=DIVERGING, zmid=1.0, zmin=0.4, zmax=1.6,
        xgap=2, ygap=2, colorbar={"title": "VPC index"},
        hovertemplate="%{y}, %{x}:00<br>value/click index %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title="Hour of day", dtick=2)
    return _layout(fig, "Value-per-click index by hour x device (1.0 = account average)", height=300)


def roas_quartiles(q: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=q.roas_quartile, y=q.share_of_spend, name="Share of spend",
                         marker={"color": SERIES[1]}, hovertemplate="%{x}: %{y:.1%} of spend<extra></extra>"))
    fig.add_trace(go.Bar(x=q.roas_quartile, y=q.share_of_value, name="Share of conversion value",
                         marker={"color": SERIES[0]}, hovertemplate="%{x}: %{y:.1%} of value<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title="Keyword ROAS quartile (Q1 = worst)")
    return _layout(fig, "Spend vs value by keyword ROAS quartile", barmode="group", bargap=0.3, bargroupgap=0.08)


# ----------------------------------------------------------------- A/B test
def posterior_plot(posterior_a: tuple[float, float], posterior_b: tuple[float, float],
                   labels=("A (control)", "B (variant)"), metric: str = "CTR") -> go.Figure:
    lo = min(stats.beta.ppf(0.0005, *posterior_a), stats.beta.ppf(0.0005, *posterior_b))
    hi = max(stats.beta.ppf(0.9995, *posterior_a), stats.beta.ppf(0.9995, *posterior_b))
    x = np.linspace(lo, hi, 500)
    fig = go.Figure()
    for (a, b), lab, col in zip((posterior_a, posterior_b), labels, (SERIES[1], SERIES[0])):
        fig.add_trace(go.Scatter(x=x, y=stats.beta.pdf(x, a, b), name=lab, mode="lines",
                                 line={"width": 2, "color": col}, fill="tozeroy",
                                 fillcolor=col + "22",
                                 hovertemplate=metric + " %{x:.4f}<br>density %{y:.0f}<extra>" + lab + "</extra>"))
    fig.update_xaxes(title=f"{metric}", tickformat=".2%")
    fig.update_yaxes(title="Posterior density")
    return _layout(fig, f"Beta posteriors of {metric}")


def lift_distribution(posterior_a, posterior_b, draws: int = 100_000, seed: int = 0) -> go.Figure:
    rng = np.random.default_rng(seed)
    lift = rng.beta(*posterior_b, draws) / rng.beta(*posterior_a, draws) - 1
    fig = go.Figure(go.Histogram(x=lift, nbinsx=80, marker={"color": SERIES[0]},
                                 hovertemplate="lift %{x:.1%}<br>%{y} draws<extra></extra>"))
    fig.add_vline(x=0, line={"color": MUTED, "width": 1, "dash": "dot"})
    fig.update_xaxes(title="Relative lift of B over A", tickformat=".0%")
    fig.update_yaxes(title="Posterior draws")
    return _layout(fig, f"Posterior of relative lift - P(B > A) = {(lift > 0).mean():.1%}", bargap=0.02)


# ----------------------------------------------------------------- bandits
def regret_curves(results: dict, every: int = 50) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        x = np.arange(1, len(r.cumulative_regret) + 1)[::every]
        col = POLICY_COLORS.get(name, SERIES[2])
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[r.regret_p95[::every], r.regret_p05[::every][::-1]],
                                 fill="toself", fillcolor=col + "1f", line={"width": 0}, hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=r.cumulative_regret[::every], name=POLICY_LABELS.get(name, name),
                                 mode="lines", line={"width": 2, "color": col},
                                 hovertemplate="round %{x:,}<br>regret %{y:.1f} conv<extra>"
                                               + POLICY_LABELS.get(name, name) + "</extra>"))
    fig.update_xaxes(title="Visitors (rounds)")
    fig.update_yaxes(title="Cumulative regret (lost conversions)")
    return _layout(fig, "Cumulative regret - mean and 5-95% band", hovermode="x unified")


def arm_allocation(results: dict, rates) -> go.Figure:
    fig = go.Figure()
    arms = [f"Variant {chr(65 + i)} ({p:.1%})" for i, p in enumerate(rates)]
    for i, arm in enumerate(arms):
        fig.add_trace(go.Bar(
            y=[POLICY_LABELS.get(n, n) for n in results], x=[r.arm_share[i] for r in results.values()],
            name=arm, orientation="h", marker={"color": SERIES[i], "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{y}<br>" + arm + ": %{x:.1%} of traffic<extra></extra>",
        ))
    fig.update_xaxes(tickformat=".0%", title="Share of traffic")
    return _layout(fig, "Traffic allocation per variant (true CVR in brackets)", barmode="stack", height=340,
                   margin={"l": 140, "r": 16, "t": 56, "b": 96}, legend={"orientation": "h", "y": -0.32, "x": 0,
                                                                        "traceorder": "normal"})


# ----------------------------------------------------------------- bidding
def strategy_bars(summary: pd.DataFrame, metric: str, title: str, fmt: str = ",.0f") -> go.Figure:
    s = summary.copy()
    err = s.get(f"{metric}_sd")
    fig = go.Figure(go.Bar(
        x=[STRATEGY_LABELS.get(n, n) for n in s.strategy], y=s[metric],
        marker={"color": [STRATEGY_COLORS.get(n, SERIES[6]) for n in s.strategy]},
        error_y={"type": "data", "array": err, "color": MUTED, "thickness": 1} if err is not None else None,
        text=[format(v, fmt) for v in s[metric]], textposition="inside", insidetextanchor="start",
        textfont={"color": "white"},
        hovertemplate="%{x}<br>" + metric + " %{y:" + fmt + "}<extra></extra>",
    ))
    return _layout(fig, title, height=360, bargap=0.35)


def strategy_daily(daily: pd.DataFrame) -> go.Figure:
    d = daily.groupby(["strategy", "date"])[["cost", "conv_value"]].mean().reset_index()
    fig = go.Figure()
    for name in ["rule_based", "ev_shrinkage", "oracle_true_value", "static_bids"]:
        g = d[d.strategy == name]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g.date, y=g.conv_value / g.cost, name=STRATEGY_LABELS[name], mode="lines",
                                 line={"width": 2, "color": STRATEGY_COLORS[name]},
                                 hovertemplate="%{x}<br>ROAS %{y:.2f}<extra>" + STRATEGY_LABELS[name] + "</extra>"))
    fig.update_yaxes(title="Daily ROAS (mean over seeds)")
    return _layout(fig, "Forward simulation - daily ROAS", hovermode="x unified")


def landscape_plot(fc: pd.DataFrame, current_bid: float | None = None, best_bid: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.bid, y=fc.clicks, name="Forecast clicks / week", mode="lines",
                             line={"width": 2, "color": SERIES[0]},
                             hovertemplate="bid %{x:.2f}<br>clicks %{y:.0f}<extra></extra>"))
    for b, lab in ((current_bid, "current"), (best_bid, "profit-max")):
        if b is not None:
            fig.add_vline(x=b, line={"color": MUTED, "width": 1, "dash": "dot"},
                          annotation_text=f"{lab} {b:.2f}", annotation_position="top")
    fig.update_xaxes(title="Max CPC bid (EUR)")
    fig.update_yaxes(title="Clicks per week")
    return _layout(fig, "Bid landscape: diminishing returns on clicks")


def landscape_profit(fc: pd.DataFrame, margin: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=fc.bid, y=fc.value * margin - fc.cost, mode="lines", name="Profit / week",
                               line={"width": 2, "color": SERIES[2]},
                               hovertemplate="bid %{x:.2f}<br>profit %{y:,.0f} EUR<extra></extra>"))
    fig.update_xaxes(title="Max CPC bid (EUR)")
    fig.update_yaxes(title=f"Forecast profit / week (EUR, {margin:.0%} margin)")
    return _layout(fig, "Bid landscape: forecast profit")
