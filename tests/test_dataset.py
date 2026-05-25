from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dataset import main
from scripts.build_showcase import main as build_showcase


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TEST_AS_OF = "2026-05-24"


def test_candidate_status_and_normalization() -> None:
    candidates = pd.read_csv(RAW_DIR / "candidates.csv")
    active = candidates[candidates["status"].eq("Active") & candidates["party"].eq("Republican")]
    assert len(active) == 9
    assert "Dan Cox" in set(active["candidate"])
    assert "Ed Hale" in set(active["candidate"])
    assert "Kurt Wedekind" in set(candidates[candidates["status"].eq("Disqualified")]["candidate"])


def test_source_metadata_complete() -> None:
    registry = pd.read_csv(RAW_DIR / "source_registry.csv")
    raw_names = {path.stem for path in RAW_DIR.glob("*.csv")} - {"source_registry"}
    assert raw_names.issubset(set(registry["dataset_name"]))
    for path in RAW_DIR.glob("*.csv"):
        if path.name in {"source_registry.csv", "wager_settings.csv"}:
            continue
        frame = pd.read_csv(path)
        assert "source_url" in frame.columns
        assert "notes" in frame.columns
        if not frame.empty:
            assert frame["source_url"].fillna("").astype(str).str.strip().ne("").all()
            assert frame["notes"].fillna("").astype(str).str.strip().ne("").all()


def test_2022_results_reconcile_to_official_totals() -> None:
    primary = pd.read_csv(RAW_DIR / "historical_2022_primary_results.csv")
    county = pd.read_csv(RAW_DIR / "historical_2022_county_results.csv")
    votes = dict(zip(primary["candidate"], primary["total_votes"]))
    assert votes["Dan Cox"] == 153_423
    assert votes["Kelly Schulz"] == 128_302
    assert votes["Robin Ficker"] == 8_268
    assert votes["Joe Werner"] == 5_075
    assert county["cox_votes"].sum() == votes["Dan Cox"]
    assert county["schulz_votes"].sum() == votes["Kelly Schulz"]
    assert county["county"].nunique() == 24


def test_pipeline_outputs_and_market_exclusion() -> None:
    result = main(TEST_AS_OF, write_outputs=False)
    scores = result["candidate_scores"]
    scenarios = result["scenarios"]
    release = result["release_status"].iloc[0]

    assert abs(scores["forecast_probability"].sum() - 1) < 1e-9
    assert len(scores) == 9
    assert scores.iloc[0]["candidate"] in {"Dan Cox", "Ed Hale"}
    assert scenarios["scenario"].nunique() >= 8
    assert release["release_status"] == "withheld"
    assert not release["forecast_publishable"]
    assert "no_credible_candidate_polling" in release["blocking_issues"]

    output = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))
    assert output["as_of"] == TEST_AS_OF
    assert output["headline_forecast"]["candidate"] == scores.iloc[0]["candidate"]
    assert "market" not in output["model_health"]
    assert "environment_context" in output


def test_pipeline_is_deterministic() -> None:
    first = main(TEST_AS_OF, write_outputs=False)["candidate_scores"]
    second = main(TEST_AS_OF, write_outputs=False)["candidate_scores"]
    pd.testing.assert_series_equal(
        first["forecast_probability"].reset_index(drop=True),
        second["forecast_probability"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        first["probability_p10"].reset_index(drop=True),
        second["probability_p10"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        first["probability_p90"].reset_index(drop=True),
        second["probability_p90"].reset_index(drop=True),
        check_names=False,
    )


def test_required_processed_outputs_exist() -> None:
    build_showcase()
    required = [
        "candidate_forecast.csv",
        "model_scenarios.csv",
        "market_comparison.csv",
        "wager_value_table.csv",
        "data_quality_report.csv",
        "candidate_poll_summary.csv",
        "environment_polls.csv",
        "environment_poll_summary.csv",
        "environment_context.csv",
        "source_manifest.csv",
        "source_registry_status.csv",
        "release_status.csv",
        "snapshot_manifest.csv",
        "model_output.json",
        "model_card.md",
        "audit_report.md",
    ]
    for filename in required:
        path = PROCESSED_DIR / filename
        assert path.exists(), filename
        assert path.stat().st_size > 0, filename
    assert (ROOT / "showcase" / "index.html").exists()


def test_wager_layer_is_safety_gated_no_bet() -> None:
    result = main(TEST_AS_OF, write_outputs=False)
    wager = result["wager_value_table"]
    assert (wager["value_flag"] == "no_bet").all()
    assert not wager["decision_eligible"].any()
    joined = ";".join(wager["block_reasons"].fillna(""))
    assert "forecast_withheld" in joined
    assert "edge_not_durable_across_scenarios" in joined


def test_processed_csvs_include_last_updated() -> None:
    for path in PROCESSED_DIR.glob("*.csv"):
        frame = pd.read_csv(path)
        assert "last_updated" in frame.columns, path.name
        if not frame.empty:
            assert set(frame["last_updated"].dropna()) == {TEST_AS_OF}


def test_probability_intervals_are_well_formed() -> None:
    scores = main(TEST_AS_OF, write_outputs=False)["candidate_scores"]
    assert (scores["probability_p10"] <= scores["forecast_probability"]).all()
    assert (scores["forecast_probability"] <= scores["probability_p90"]).all()
    assert (scores["confidence_band_width"] >= 0).all()


def test_environment_poll_layer_has_signal() -> None:
    env_context = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))["environment_context"]
    env_summary = pd.read_csv(PROCESSED_DIR / "environment_poll_summary.csv")
    assert not env_summary.empty
    assert {"job_approval", "direction_of_state", "matchup"}.issubset(set(env_summary["measure_group"]))
    assert "anti_incumbent_pressure" in env_context
    assert pd.notna(env_context["anti_incumbent_pressure"])
