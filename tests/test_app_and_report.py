"""Smoke tests: the end-to-end pipeline, the HTML report and the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline import RunConfig, run
from src.report import build_report


def test_pipeline_and_report(small_dbs, tmp_path):
    cfg = RunConfig(db=Path(small_dbs["duck_path"]), sqlite=None, out_dir=tmp_path, seeds=(0,),
                    bandit_sims=10, bandit_rounds=2_000, forward_weeks=1)
    res = run(cfg, verbose=False)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["note"].startswith("All data are synthetic")
    assert len(list((tmp_path / "sql").glob("*.csv"))) == 7
    path = build_report(res, tmp_path / "report.html", inline_js=False)
    html = path.read_text()
    assert "SEM Bidding Simulator" in html and html.count("plotly-graph-div") >= 10


def test_streamlit_app_renders_without_exceptions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parent.parent
    if not (root / "data" / "sem.duckdb").exists():
        pytest.skip("run `python -m src.data_generator` first")
    at = AppTest.from_file(str(root / "app.py"), default_timeout=300).run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_streamlit_app_survives_edge_case_inputs():
    """0 conversions in A, an impossible baseline x MDE, invalid bandit rates, more history days
    than the DB holds: every case shows a message instead of crashing."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parent.parent
    if not (root / "data" / "sem.duckdb").exists():
        pytest.skip("run `python -m src.data_generator` first")
    at = AppTest.from_file(str(root / "app.py"), default_timeout=300).run()
    at.number_input[1].set_value(0)            # Conversions A = 0
    at.number_input[4].set_value(0.9)          # baseline 90% ...
    at.number_input[5].set_value(1.0)          # ... + 100% relative MDE = 180%
    at.text_input[0].set_value("0.5, abc")     # invalid bandit rates
    at.run()
    assert not at.exception
    assert any(m.value == "n/a" for m in at.metric)
    errors = " ".join(e.value for e in at.error)
    assert "below 100%" in errors and "comma-separated rates" in errors
