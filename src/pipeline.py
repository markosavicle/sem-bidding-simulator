"""End-to-end sample run: data -> SQL -> A/B -> bandits -> bidding -> outputs.

    python -m src.pipeline                      # uses data/sem.duckdb (generates it if missing)
    python -m src.pipeline --regenerate --seeds 5 --report outputs/report.html

Writes:
    outputs/summary.json          headline numbers of every module
    outputs/sql/*.csv             result of every query in sql/
    outputs/bidding_*.csv         strategy comparison tables
    outputs/report.html           self-contained HTML report (see src/report.py)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import bidding_optimizer as bo
from . import data_generator as dg
from . import stats_engine as se
from .sql_runner import connect, run_all

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunConfig:
    db: Path = ROOT / "data" / "sem.duckdb"
    sqlite: Path | None = ROOT / "data" / "sem.sqlite"
    out_dir: Path = ROOT / "outputs"
    regenerate: bool = False
    seed: int = 42
    days: int = 120
    # A/B experiment (ad copy CTR test) - synthetic ground truth
    ab_true_ctr: tuple[float, float] = (0.040, 0.044)
    ab_impressions_per_arm: int = 40_000
    # bandit
    bandit_rates: tuple[float, ...] = (0.030, 0.034, 0.040)
    bandit_rounds: int = 20_000
    bandit_sims: int = 200
    value_per_conversion: float = 60.0
    # bidding
    margin: float = 0.35
    target_roas: float | None = None  # None -> break-even ROAS = 1 / margin
    forward_weeks: int = 4
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    extra: dict = field(default_factory=dict)


def simulate_ab_experiment(true_a: float, true_b: float, n: int, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    return {"clicks_a": int(rng.binomial(n, true_a)), "imps_a": n,
            "clicks_b": int(rng.binomial(n, true_b)), "imps_b": n}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run(cfg: RunConfig, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) data ------------------------------------------------------------
    if cfg.regenerate or not cfg.db.exists():
        log(f"[data] generating {cfg.days} days (seed={cfg.seed}) ...")
        tables = dg.generate(n_days=cfg.days, seed=cfg.seed)
        dg.write_duckdb(tables, cfg.db)
        if cfg.sqlite:
            dg.write_sqlite(tables, cfg.sqlite)
    con = connect(cfg.db)
    meta = bo.load_meta(con)
    world = dg.SemWorld.build(seed=int(meta["seed"]), keyword_scale=float(meta["keyword_scale"]))
    history = bo.load_daily_history(con)
    log(f"[data] {len(history):,} keyword-days, {history.date.min()} .. {history.date.max()}")

    # 2) SQL ---------------------------------------------------------------
    sql = run_all(con)
    (cfg.out_dir / "sql").mkdir(exist_ok=True)
    for name, df in sql.items():
        df.to_csv(cfg.out_dir / "sql" / f"{name}.csv", index=False)
    waste = sql["05_wasted_spend_search_terms"]
    total_cost = float(sql["01_campaign_kpis"].cost.sum())
    log(f"[sql] {len(sql)} queries OK; wasted spend flagged: {waste.wasted_cost.sum():,.0f} EUR "
        f"({waste.wasted_cost.sum() / total_cost:.1%} of spend) in {len(waste)} search terms")

    # 3) A/B ---------------------------------------------------------------
    ab = simulate_ab_experiment(*cfg.ab_true_ctr, cfg.ab_impressions_per_arm)
    z = se.two_proportion_ztest(ab["clicks_a"], ab["imps_a"], ab["clicks_b"], ab["imps_b"])
    bayes = se.bayesian_ab_test(ab["clicks_a"], ab["imps_a"], ab["clicks_b"], ab["imps_b"])
    mde = cfg.ab_true_ctr[1] / cfg.ab_true_ctr[0] - 1
    n_req = se.required_sample_size(cfg.ab_true_ctr[0], mde)
    srm = se.srm_check(ab["imps_a"], ab["imps_b"])
    log(f"[ab] CTR {z.rate_a:.4f} -> {z.rate_b:.4f} (lift {z.rel_lift:+.1%}), p={z.p_value:.4f}, "
        f"P(B>A)={bayes.prob_b_beats_a:.3f}; n required for {mde:.0%} MDE: {n_req:,}/arm")

    # 4) bandits -----------------------------------------------------------
    bandits = se.simulate_bandits(cfg.bandit_rates, n_rounds=cfg.bandit_rounds, n_sims=cfg.bandit_sims,
                                  value_per_conversion=cfg.value_per_conversion)
    bandit_table = se.bandit_summary(bandits)
    for row in bandit_table:
        log(f"[bandit] {row['policy']:<13} regret={row['final_cum_regret']:7.1f}  "
            f"revenue={row['total_revenue']:9,.0f}  lift={row['revenue_lift_vs_static']:+.1%}")

    # 5) bidding -----------------------------------------------------------
    target_roas = cfg.target_roas or 1.0 / cfg.margin
    sim = bo.compare_strategies(world, history, bo.default_strategies(world, target_roas), target_roas=target_roas,
                                n_weeks=cfg.forward_weeks, seeds=cfg.seeds, margin=cfg.margin)
    sim.summary.to_csv(cfg.out_dir / "bidding_summary.csv", index=False)
    sim.per_seed.to_csv(cfg.out_dir / "bidding_per_seed.csv", index=False)
    sim.final_bids.to_csv(cfg.out_dir / "bidding_final_bids.csv", index=False)
    s = sim.summary.set_index("strategy")
    for name, r in s.iterrows():
        log(f"[bidding] {name:<16} ROAS={r.roas:5.2f}  CPA={r.cpa:6.2f}  conv={r.conversions:7.1f}  "
            f"profit={r.profit:10,.0f}")
    recent = history[history.date >= sorted(history.date.unique())[-28]]
    est_err_full = bo.cvr_estimation_error(history, world)
    est_err_28 = bo.cvr_estimation_error(recent, world)

    # paired per-seed comparison EV-shrinkage vs rules (common random numbers)
    ps = sim.per_seed.pivot(index="seed", columns="strategy", values=["roas", "profit"])
    paired = {
        "roas_diff_mean": float((ps["roas"]["ev_shrinkage"] - ps["roas"]["rule_based"]).mean()),
        "roas_diff_min": float((ps["roas"]["ev_shrinkage"] - ps["roas"]["rule_based"]).min()),
        "profit_diff_mean": float((ps["profit"]["ev_shrinkage"] - ps["profit"]["rule_based"]).mean()),
        "profit_diff_min": float((ps["profit"]["ev_shrinkage"] - ps["profit"]["rule_based"]).min()),
        "seeds_ev_wins_profit": int((ps["profit"]["ev_shrinkage"] > ps["profit"]["rule_based"]).sum()),
        "n_seeds": len(ps),
    }

    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "note": "All data are synthetic (simulated world). Numbers are not client results.",
        "data": {"meta": meta, "keyword_days": len(history), "keywords": int(history.keyword_id.nunique()),
                 "date_from": history.date.min(), "date_to": history.date.max()},
        "sql": {
            "campaign_kpis": sql["01_campaign_kpis"].to_dict(orient="records"),
            "wasted_spend_eur": float(waste.wasted_cost.sum()),
            "wasted_spend_share": float(waste.wasted_cost.sum() / total_cost),
            "wasted_terms": len(waste),
            "top_waste_terms": waste.head(5)[["search_term", "waste_flag", "cost", "conversions",
                                              "wasted_cost"]].to_dict(orient="records"),
        },
        "ab_test": {"data": ab, "true_ctr": cfg.ab_true_ctr, "frequentist": z.as_dict(),
                    "bayesian": bayes.as_dict(), "required_n_per_arm": n_req, "srm": srm},
        "bandits": {"true_rates": cfg.bandit_rates, "rounds": cfg.bandit_rounds, "sims": cfg.bandit_sims,
                    "table": bandit_table},
        "bidding": {
            "target_roas": target_roas, "margin": cfg.margin, "forward_weeks": cfg.forward_weeks,
            "seeds": list(cfg.seeds), "summary": sim.summary.to_dict(orient="records"),
            "paired_ev_shrinkage_vs_rules": paired,
            "cvr_estimation_error_full_history": est_err_full.to_dict(orient="records"),
            "cvr_estimation_error_last_28d": est_err_28.to_dict(orient="records"),
        },
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    log(f"[done] {summary['runtime_seconds']}s -> {cfg.out_dir / 'summary.json'}")
    con.close()
    return {"summary": summary, "sql": sql, "bandits": bandits, "bidding": sim, "history": history,
            "world": world, "ab": ab, "est_err_28": est_err_28, "est_err_full": est_err_full}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(RunConfig.db))
    p.add_argument("--out", default=str(RunConfig.out_dir))
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=5, help="replications of the forward bidding simulation")
    p.add_argument("--bandit-sims", type=int, default=200)
    p.add_argument("--report", default=None, help="also write the HTML report to this path")
    a = p.parse_args(argv)
    cfg = RunConfig(db=Path(a.db), out_dir=Path(a.out), regenerate=a.regenerate, seed=a.seed,
                    seeds=tuple(range(a.seeds)), bandit_sims=a.bandit_sims)
    results = run(cfg)
    if a.report:
        from .report import build_report

        path = build_report(results, a.report)
        print(f"[report] {path}")


if __name__ == "__main__":
    main()
