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
