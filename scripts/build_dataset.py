from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
DEFAULT_AS_OF = pd.Timestamp("2026-05-20")
AS_OF = DEFAULT_AS_OF
PRIMARY_DATE = pd.Timestamp("2026-06-23")
FORECAST_VERSION = "1.0"


def configure_run(as_of: object | None = None) -> pd.Timestamp:
    global AS_OF
    AS_OF = pd.Timestamp(as_of).normalize() if as_of is not None else DEFAULT_AS_OF
    return AS_OF


def read_raw() -> dict[str, pd.DataFrame]:
    return {
        path.stem: pd.read_csv(path)
        for path in RAW_DIR.glob("*.csv")
    }


def add_last_updated(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["last_updated"] = str(AS_OF.date())
    return out


def write_csv(name: str, frame: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    add_last_updated(frame).to_csv(PROCESSED_DIR / name, index=False)


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_source_metadata(raw: dict[str, pd.DataFrame]) -> None:
    registry = raw["source_registry"]
    require_columns(
        registry,
        [
            "dataset_name",
            "source_type",
            "retrieval_method",
            "freshness_sla_days",
            "trust_tier",
            "citation_url",
            "refresh_instructions",
            "notes",
        ],
        "source_registry",
    )
    raw_names = set(raw) - {"source_registry"}
    missing_registry = raw_names - set(registry["dataset_name"])
    if missing_registry:
        raise AssertionError(f"source_registry missing datasets: {sorted(missing_registry)}")

    for name, frame in raw.items():
        if name in {"source_registry", "wager_settings"}:
            continue
        require_columns(frame, ["source_url", "notes"], name)
        if not frame.empty:
            if frame["source_url"].fillna("").astype(str).str.strip().eq("").any():
                raise AssertionError(f"{name} has rows missing source_url")
            if frame["notes"].fillna("").astype(str).str.strip().eq("").any():
                raise AssertionError(f"{name} has rows missing notes")


def validate_candidates(candidates: pd.DataFrame) -> None:
    require_columns(
        candidates,
        ["candidate", "lieutenant_governor", "party", "status", "filed_date", "source_url", "notes"],
        "candidates",
    )
    active = candidates[candidates["party"].eq("Republican") & candidates["status"].eq("Active")]
    expected = {
        "Carl A. Brunner Jr.",
        "L. D. Burkindine",
        "Dan Cox",
        "Ed Hale",
        "Douglas Larcomb",
        "John A. Myrick",
        "Michael Oakes",
        "Nancy Jane Taylor",
        "Shannon Wright",
    }
    if set(active["candidate"]) != expected:
        raise AssertionError("Active Republican candidate set does not match the May 20 SBE list.")
    if "Kurt Wedekind" not in set(candidates.loc[candidates["status"].eq("Disqualified"), "candidate"]):
        raise AssertionError("Disqualified Wedekind ticket is missing from raw candidate status data.")


def validate_2022_results(primary: pd.DataFrame, county: pd.DataFrame) -> None:
    expected_votes = {
        "Dan Cox": 153_423,
        "Robin Ficker": 8_268,
        "Kelly Schulz": 128_302,
        "Joe Werner": 5_075,
    }
    votes = dict(zip(primary["candidate"], primary["total_votes"]))
    for candidate, expected in expected_votes.items():
        if int(votes[candidate]) != expected:
            raise AssertionError(f"{candidate} total changed: expected {expected}, got {votes[candidate]}")
    if int(county["cox_votes"].sum()) != expected_votes["Dan Cox"]:
        raise AssertionError("County Cox totals do not reconcile to statewide total.")
    if int(county["schulz_votes"].sum()) != expected_votes["Kelly Schulz"]:
        raise AssertionError("County Schulz totals do not reconcile to statewide total.")
    if county["county"].nunique() != 24:
        raise AssertionError("Expected all 24 Maryland jurisdictions in county history.")


def validate_all(raw: dict[str, pd.DataFrame]) -> None:
    validate_source_metadata(raw)
    validate_candidates(raw["candidates"])
    validate_2022_results(raw["historical_2022_primary_results"], raw["historical_2022_county_results"])


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["filed_date"] = pd.to_datetime(out["filed_date"])
    out["is_active_republican"] = out["party"].eq("Republican") & out["status"].eq("Active")
    out["has_website"] = out["website"].fillna("").astype(str).str.strip().ne("")
    out["has_social"] = (
        out[["facebook", "x"]].fillna("").astype(str).apply(lambda row: any(value.strip() for value in row), axis=1)
    )
    out["days_on_file"] = (AS_OF - out["filed_date"]).dt.days.clip(lower=0)
    return out


def jurisdiction_key(value: object) -> str:
    text = str(value).strip()
    if text in {"Baltimore City", "Baltimore County"}:
        return text
    return text.removesuffix(" County")


def process_2022_county(county: pd.DataFrame) -> pd.DataFrame:
    out = county.copy()
    numeric = ["cox_votes", "ficker_votes", "schulz_votes", "werner_votes"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out["total_2022_gop_votes"] = out[numeric].sum(axis=1)
    out["cox_share"] = out["cox_votes"] / out["total_2022_gop_votes"]
    out["schulz_share"] = out["schulz_votes"] / out["total_2022_gop_votes"]
    out["cox_margin_vs_schulz"] = out["cox_share"] - out["schulz_share"]
    out["county_lane"] = np.select(
        [out["cox_margin_vs_schulz"] >= 0.12, out["cox_margin_vs_schulz"] <= -0.05],
        ["cox_base", "establishment_lean"],
        default="competitive_gop_primary",
    )
    out["jurisdiction_key"] = out["county"].map(jurisdiction_key)
    return out


def process_finance(finance: pd.DataFrame) -> pd.DataFrame:
    out = finance.copy()
    out["report_date"] = pd.to_datetime(out["report_date"])
    for column in ["raised_usd", "spent_usd", "cash_on_hand_usd"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["days_old"] = (AS_OF - out["report_date"]).dt.days.clip(lower=0)
    out["cash_score"] = np.log1p(out["cash_on_hand_usd"].fillna(0)) / np.log1p(125_000)
    out["raised_score"] = np.log1p(out["raised_usd"].fillna(0)) / np.log1p(250_000)
    out["finance_score"] = 8.0 * out[["cash_score", "raised_score"]].mean(axis=1)
    return out


def process_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    out["impact_points"] = pd.to_numeric(out["impact_points"], errors="raise")
    out["days_old"] = (AS_OF - out["event_date"]).dt.days.clip(lower=0)
    out["event_weight"] = np.exp(-np.log(2) * out["days_old"] / 60)
    out["weighted_impact_points"] = out["impact_points"] * out["event_weight"]
    return out


def process_markets(markets: pd.DataFrame, settings: dict[str, float]) -> pd.DataFrame:
    out = markets.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"])
    for column in ["yes_price", "implied_probability", "volume_usd", "liquidity_usd"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["days_stale"] = (AS_OF - out["snapshot_date"]).dt.days.clip(lower=0)
    total = out["implied_probability"].sum()
    out["normalized_probability"] = out["implied_probability"] / total if total else np.nan
    out["stale_price_flag"] = out["days_stale"] > settings["stale_market_days"]
    out["liquidity_warning"] = np.where(
        out["liquidity_usd"].fillna(0) < settings["min_liquidity_usd"],
        "thin_liquidity",
        "ok",
    )
    out["settlement_warning"] = np.where(
        out["notes"].str.lower().str.contains("non-active|verify", regex=True),
        "verify_contract_terms",
        "ok",
    )
    return out


def settings_map(settings: pd.DataFrame) -> dict[str, float]:
    return {row["setting"]: float(row["value"]) for _, row in settings.iterrows()}


def build_candidate_scores(
    candidates: pd.DataFrame,
    primary_2022: pd.DataFrame,
    county_2022: pd.DataFrame,
    finance: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    active = candidates[candidates["is_active_republican"]].copy()
    scores = pd.DataFrame({"candidate": active["candidate"]})
    scores = scores.merge(active[["candidate", "lieutenant_governor", "home_jurisdiction", "days_on_file", "has_website", "has_social"]], on="candidate")
    scores["ballot_score"] = 3.0
    scores["infrastructure_score"] = (
        np.minimum(scores["days_on_file"] / 365, 1.0) * 2.0
        + scores["has_website"].astype(float) * 1.5
        + scores["has_social"].astype(float) * 0.75
    )

    statewide_total = float(primary_2022["total_votes"].sum())
    cox_votes = float(primary_2022.loc[primary_2022["candidate"].eq("Dan Cox"), "total_votes"].iloc[0])
    cox_2022_share = cox_votes / statewide_total
    scores["historical_score"] = np.where(scores["candidate"].eq("Dan Cox"), 18.0 * cox_2022_share, 0.0)
    scores.loc[scores["candidate"].eq("Shannon Wright"), "historical_score"] += 2.0

    finance_by_candidate = finance.groupby("candidate", as_index=False)["finance_score"].sum()
    scores = scores.merge(finance_by_candidate, on="candidate", how="left")
    scores["finance_score"] = scores["finance_score"].fillna(0.0)

    event_by_candidate = events.groupby("candidate", as_index=False)["weighted_impact_points"].sum()
    scores = scores.merge(event_by_candidate, on="candidate", how="left")
    scores["event_score"] = scores["weighted_impact_points"].fillna(0.0)
    scores = scores.drop(columns=["weighted_impact_points"])

    local_base = county_2022.groupby("county", as_index=False).agg(
        total_2022_gop_votes=("total_2022_gop_votes", "sum"),
        cox_share=("cox_share", "mean"),
        jurisdiction_key=("jurisdiction_key", "first"),
    )
    scores["jurisdiction_key"] = scores["home_jurisdiction"].map(jurisdiction_key)
    scores = scores.merge(local_base, on="jurisdiction_key", how="left")
    scores["local_geography_score"] = np.log1p(scores["total_2022_gop_votes"].fillna(0)) / np.log1p(40_000)
    scores["cox_base_local_boost"] = np.where(scores["candidate"].eq("Dan Cox"), scores["cox_share"].fillna(0.50) * 2.0, 0.0)

    known_lane_adjustments = {
        "Ed Hale": 7.0,
        "John A. Myrick": 3.0,
        "Douglas Larcomb": 1.0,
        "Shannon Wright": 1.5,
    }
    scores["lane_score"] = scores["candidate"].map(known_lane_adjustments).fillna(0.0)

    component_columns = [
        "ballot_score",
        "infrastructure_score",
        "historical_score",
        "finance_score",
        "event_score",
        "local_geography_score",
        "cox_base_local_boost",
        "lane_score",
    ]
    scores["raw_score"] = scores[component_columns].sum(axis=1)
    scores["score_floor"] = 1.0
    scores["model_score"] = scores["raw_score"] + scores["score_floor"]
    exp_score = np.exp(scores["model_score"] / 8.0)
    scores["forecast_probability"] = exp_score / exp_score.sum()
    scores = scores.sort_values("forecast_probability", ascending=False).reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1)

    diagnostics = {
        "cox_2022_share": cox_2022_share,
        "cox_2022_votes": cox_votes,
        "statewide_2022_gop_primary_votes": statewide_total,
        "poll_rows": 0.0,
        "active_candidate_count": float(len(scores)),
        "score_entropy": float(-(scores["forecast_probability"] * np.log(scores["forecast_probability"])).sum()),
    }
    return scores, diagnostics


def scenario_adjustments(base_scores: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = base_scores.copy()
    out["scenario_score"] = out["model_score"]
    if scenario == "cox_repeat_base_consolidation":
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] += 4.0
    elif scenario == "hale_money_moderate_surge":
        out.loc[out["candidate"].eq("Ed Hale"), "scenario_score"] += 4.5
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] -= 1.0
    elif scenario == "myrick_wright_larcomb_debate_surge":
        out.loc[out["candidate"].isin(["John A. Myrick", "Shannon Wright", "Douglas Larcomb"]), "scenario_score"] += 3.5
    elif scenario == "fragmented_field_plurality":
        out["scenario_score"] = out["scenario_score"] * 0.90 + 2.0
    elif scenario == "late_trump_or_major_gop_endorsement":
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] += 3.0
        out.loc[out["candidate"].eq("Ed Hale"), "scenario_score"] -= 1.5
    elif scenario == "mail_ballot_admin_uncertainty":
        out["scenario_score"] = out["scenario_score"] * 0.95 + 1.0
    elif scenario == "low_turnout":
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] += 2.0
    elif scenario == "high_turnout":
        out.loc[out["candidate"].isin(["Ed Hale", "John A. Myrick"]), "scenario_score"] += 2.0
    exp_score = np.exp(out["scenario_score"] / 8.0)
    out["scenario_probability"] = exp_score / exp_score.sum()
    out["scenario"] = scenario
    return out[["scenario", "candidate", "scenario_probability"]]


def build_scenarios(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        "baseline",
        "cox_repeat_base_consolidation",
        "hale_money_moderate_surge",
        "myrick_wright_larcomb_debate_surge",
        "fragmented_field_plurality",
        "late_trump_or_major_gop_endorsement",
        "mail_ballot_admin_uncertainty",
        "low_turnout",
        "high_turnout",
    ]
    rows = [scenario_adjustments(candidate_scores, scenario) for scenario in scenarios]
    return pd.concat(rows, ignore_index=True)


def build_market_comparison(candidate_scores: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    out = candidate_scores[["candidate", "forecast_probability"]].merge(
        markets[["candidate", "yes_price", "implied_probability", "normalized_probability", "stale_price_flag", "liquidity_warning", "settlement_warning"]],
        on="candidate",
        how="left",
    )
    out["market_probability"] = out["normalized_probability"].fillna(out["implied_probability"])
    out["edge_vs_market"] = out["forecast_probability"] - out["market_probability"]
    out["market_data_available"] = out["market_probability"].notna()
    return out


def build_release_status(
    raw: dict[str, pd.DataFrame],
    candidate_scores: pd.DataFrame,
    markets: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    blocking: list[str] = []
    penalties: list[str] = []
    if raw["polls"].empty:
        blocking.append("no_credible_public_polling")
        penalties.append("polling_component_unavailable")
    if set(candidate_scores["candidate"]) - set(raw["finance"]["candidate"]):
        blocking.append("finance_incomplete_for_full_field")
        penalties.append("finance_missing_for_some_active_candidates")
    if markets["stale_price_flag"].any():
        blocking.append("stale_market_price")
    if (markets["settlement_warning"] != "ok").any():
        blocking.append("verify_contract_terms")
    if candidate_scores["forecast_probability"].max() < 0.50:
        penalties.append("plurality_field_high_uncertainty")
    leader = candidate_scores.iloc[0]["candidate"]
    leader_scenarios = scenarios[scenarios["candidate"].eq(leader)]
    if leader_scenarios["scenario_probability"].min() < 0.52:
        blocking.append("edge_not_durable_across_scenarios")

    score = 100
    score -= 22 if "no_credible_public_polling" in blocking else 0
    score -= 16 if "finance_incomplete_for_full_field" in blocking else 0
    score -= 12 if "stale_market_price" in blocking else 0
    score -= 10 if "edge_not_durable_across_scenarios" in blocking else 0
    score -= 8 if "verify_contract_terms" in blocking else 0
    score -= 6 * len(set(penalties) - {"polling_component_unavailable", "finance_missing_for_some_active_candidates"})
    score = max(0, min(100, score))
    reliability_class = "high" if score >= 75 else "medium" if score >= 50 else "low"
    return pd.DataFrame(
        [
            {
                "release_status": "published" if not blocking else "withheld",
                "forecast_publishable": not blocking,
                "betting_eligible": False if blocking else True,
                "reliability_score": round(score, 1),
                "reliability_class": reliability_class,
                "blocking_issues": ";".join(dict.fromkeys(blocking)),
                "reliability_penalties": ";".join(dict.fromkeys(penalties)),
            }
        ]
    )


def build_wager_value_table(
    market_comparison: pd.DataFrame,
    release_status: pd.DataFrame,
    scenarios: pd.DataFrame,
    settings: dict[str, float],
) -> pd.DataFrame:
    publishable = bool(release_status.iloc[0]["forecast_publishable"])
    rows = []
    for _, row in market_comparison.iterrows():
        candidate = row["candidate"]
        block_reasons: list[str] = []
        if not publishable:
            block_reasons.append("forecast_withheld")
        if not bool(row["market_data_available"]):
            block_reasons.append("no_direct_market")
        if bool(row["market_data_available"]) and bool(row.get("stale_price_flag", False)):
            block_reasons.append("stale_market_price")
        if bool(row["market_data_available"]) and str(row.get("liquidity_warning", "")) != "ok" and pd.notna(row.get("liquidity_warning")):
            block_reasons.append(str(row["liquidity_warning"]))
        if bool(row["market_data_available"]) and str(row.get("settlement_warning", "")) != "ok" and pd.notna(row.get("settlement_warning")):
            block_reasons.append(str(row["settlement_warning"]))
        edge = row["edge_vs_market"]
        if pd.isna(edge) or edge < settings["min_required_edge"]:
            block_reasons.append("edge_below_required_buffer")
        worst_case = scenarios.loc[scenarios["candidate"].eq(candidate), "scenario_probability"].min()
        if pd.isna(worst_case) or worst_case < settings["scenario_durability_threshold"]:
            block_reasons.append("edge_not_durable_across_scenarios")
        decision_eligible = not block_reasons
        rows.append(
            {
                "candidate": candidate,
                "forecast_probability": row["forecast_probability"],
                "market_probability": row["market_probability"],
                "edge_vs_market": edge,
                "worst_case_scenario_probability": worst_case,
                "required_edge": settings["min_required_edge"],
                "value_flag": "candidate_value" if decision_eligible else "no_bet",
                "decision_eligible": decision_eligible,
                "capped_exposure_fraction": min(settings["max_exposure_fraction"], max(0.0, float(edge) / 2.0 if pd.notna(edge) else 0.0)) if decision_eligible else 0.0,
                "block_reasons": ";".join(dict.fromkeys(block_reasons)),
            }
        )
    return pd.DataFrame(rows)


def build_data_quality(raw: dict[str, pd.DataFrame], markets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in raw.items():
        if name == "source_registry":
            continue
        missing_sources = 0
        missing_notes = 0
        if "source_url" in frame.columns and not frame.empty:
            missing_sources = int(frame["source_url"].fillna("").astype(str).str.strip().eq("").sum())
        if "notes" in frame.columns and not frame.empty:
            missing_notes = int(frame["notes"].fillna("").astype(str).str.strip().eq("").sum())
        stale = False
        if name == "markets" and not markets.empty:
            stale = bool(markets["stale_price_flag"].any())
        rows.append(
            {
                "dataset_name": name,
                "row_count": len(frame),
                "missing_source_count": missing_sources,
                "missing_notes_count": missing_notes,
                "stale": stale,
                "required_for_publish": name in {"candidates", "historical_2022_primary_results", "finance", "polls", "markets"},
            }
        )
    return pd.DataFrame(rows)


def build_source_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"dataset_name": path.stem, "file": str(path.relative_to(ROOT)), "sha256": digest, "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def build_source_registry_status(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    registry = raw["source_registry"].copy()
    checks = raw["source_checks"].copy()
    checks["check_date"] = pd.to_datetime(checks["check_date"])
    latest_checks = checks.sort_values("check_date").groupby("dataset_name", as_index=False).tail(1)
    out = registry.merge(
        latest_checks[["dataset_name", "check_date", "check_result"]],
        on="dataset_name",
        how="left",
    )
    out["latest_source_check_date"] = out["check_date"].dt.date.astype(str).replace("NaT", "")
    out["freshness_sla_days"] = pd.to_numeric(out["freshness_sla_days"], errors="coerce")
    out["days_since_check"] = (AS_OF - out["check_date"]).dt.days
    out["freshness_pass"] = out["days_since_check"].fillna(0) <= out["freshness_sla_days"].fillna(999)
    out["parse_status"] = "manual_or_csv_ok"
    out["freshness_basis"] = np.where(out["check_date"].notna(), "source_check", "registry")
    return out.drop(columns=["check_date"])


def build_model_card(candidate_scores: pd.DataFrame, release: pd.DataFrame, diagnostics: dict[str, float]) -> str:
    leader = candidate_scores.iloc[0]
    return f"""# Model Card

## Intended Use

This package estimates the June 23, 2026 Maryland Republican gubernatorial primary as a probabilistic research forecast. It is not a guarantee or financial advice.

## Headline

- As of: `{AS_OF.date()}`
- Top candidate: `{leader['candidate']}`
- Top candidate fair probability: `{leader['forecast_probability']:.1%}`
- Release status: `{release.iloc[0]['release_status']}`
- Betting eligible: `{bool(release.iloc[0]['betting_eligible'])}`

## Inputs

- Official May 20 SBE candidate list.
- Official 2022 Maryland GOP governor primary statewide and county results.
- Current campaign-finance reporting summarized from Maryland Matters and MDCRIS source checks.
- Debate, event, and ballot-administration signals.
- Prediction markets as an external price check only.

## Known Limits

- No credible public candidate-level poll was found as of the run date.
- Finance is incomplete for several minor active candidates.
- Market prices are stale and include non-active names, so the betting layer defaults to no bet.
- Reliability score: `{release.iloc[0]['reliability_score']}/100`.
"""


def build_audit_report(candidate_scores: pd.DataFrame, release: pd.DataFrame, wager: pd.DataFrame) -> str:
    leader = candidate_scores.iloc[0]
    ranking_lines = ["| rank | candidate | forecast_probability |", "| ---: | --- | ---: |"]
    for _, row in candidate_scores.iterrows():
        ranking_lines.append(
            f"| {int(row['rank'])} | {row['candidate']} | {float(row['forecast_probability']):.3f} |"
        )
    ranking_table = "\n".join(ranking_lines)
    return f"""# Forecast Audit Report

- Run ID: `mdgopgov-{AS_OF.strftime('%Y%m%d')}`
- As of: `{AS_OF.date()}`
- Release status: `{release.iloc[0]['release_status']}`
- Forecast publishable: `{bool(release.iloc[0]['forecast_publishable'])}`
- Betting eligible: `{bool(release.iloc[0]['betting_eligible'])}`
- Reliability: `{release.iloc[0]['reliability_class']}` ({release.iloc[0]['reliability_score']}/100)

## Headline Forecast

- Leader: `{leader['candidate']}`
- Leader fair probability: `{leader['forecast_probability']:.3f}`
- Active candidates modeled: `{len(candidate_scores)}`

## Blocking Issues

{release.iloc[0]['blocking_issues'] or 'None'}

## Betting Decision Summary

All current market rows are safety-gated to no bet. Primary blockers include stale market prices, no credible public polling, incomplete full-field finance, and insufficient scenario durability.

## Candidate Ranking

{ranking_table}
"""


def build_snapshot() -> pd.DataFrame:
    snapshot_dir = SNAPSHOT_DIR / f"mdgopgov-{AS_OF.strftime('%Y%m%d')}" / "raw"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        dest = snapshot_dir / path.name
        shutil.copy2(path, dest)
        rows.append({"snapshot_file": str(dest.relative_to(ROOT)), "source_file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.DataFrame(rows)


def render_terminal_report(
    candidate_scores: pd.DataFrame,
    release: pd.DataFrame,
    wager: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> str:
    leader = candidate_scores.iloc[0]
    release_row = release.iloc[0]
    scenario_leaders = (
        scenarios.sort_values(["scenario", "scenario_probability"], ascending=[True, False])
        .groupby("scenario", as_index=False)
        .first()[["scenario", "candidate", "scenario_probability"]]
    )
    lines = [
        f"As of {AS_OF.date()}",
        f"Leader: {leader['candidate']} ({leader['forecast_probability']:.1%})",
        f"Release status: {release_row['release_status']}",
        f"Betting eligible: {bool(release_row['betting_eligible'])}",
        f"Reliability: {release_row['reliability_class']} ({release_row['reliability_score']}/100)",
        "",
        "Candidate ranking:",
    ]
    for _, row in candidate_scores.iterrows():
        lines.append(f"  {int(row['rank'])}. {row['candidate']} - {row['forecast_probability']:.1%}")
    lines.extend(["", "Scenario leaders:"])
    for _, row in scenario_leaders.iterrows():
        lines.append(f"  {row['scenario']}: {row['candidate']} ({row['scenario_probability']:.1%})")
    lines.extend(["", "Wager gate:"])
    for _, row in wager.iterrows():
        lines.append(f"  {row['candidate']}: {row['value_flag']} [{row['block_reasons']}]")
    if release_row["blocking_issues"]:
        lines.extend(["", f"Blocking issues: {release_row['blocking_issues']}"])
    return "\n".join(lines)


def main(as_of: object | None = None, write_outputs: bool = True) -> dict[str, Any]:
    configure_run(as_of)
    raw = read_raw()
    validate_all(raw)

    settings = settings_map(raw["wager_settings"])
    candidates = normalize_candidates(raw["candidates"])
    county_2022 = process_2022_county(raw["historical_2022_county_results"])
    finance = process_finance(raw["finance"])
    events = process_events(raw["events"])
    markets = process_markets(raw["markets"], settings)
    candidate_scores, diagnostics = build_candidate_scores(
        candidates,
        raw["historical_2022_primary_results"],
        county_2022,
        finance,
        events,
    )
    scenarios = build_scenarios(candidate_scores)
    market_comparison = build_market_comparison(candidate_scores, markets)
    release = build_release_status(raw, candidate_scores, markets, scenarios)
    wager = build_wager_value_table(market_comparison, release, scenarios, settings)
    quality = build_data_quality(raw, markets)
    source_manifest = build_source_manifest()
    source_status = build_source_registry_status(raw)
    model_output = {
        "version": FORECAST_VERSION,
        "as_of": str(AS_OF.date()),
        "primary_date": str(PRIMARY_DATE.date()),
        "headline_forecast": {
            "candidate": candidate_scores.iloc[0]["candidate"],
            "probability": round(float(candidate_scores.iloc[0]["forecast_probability"]), 4),
        },
        "candidate_probabilities": {
            row["candidate"]: round(float(row["forecast_probability"]), 4)
            for _, row in candidate_scores.iterrows()
        },
        "release_status": release.iloc[0].to_dict(),
        "model_health": diagnostics,
        "betting_decision_summary": {
            row["candidate"]: row["value_flag"]
            for _, row in wager.iterrows()
        },
    }

    if write_outputs:
        snapshot = build_snapshot()
        outputs = {
            "candidates.csv": candidates,
            "historical_2022_primary_results.csv": raw["historical_2022_primary_results"],
            "historical_2022_county_results.csv": county_2022,
            "finance.csv": finance,
            "events.csv": events,
            "markets.csv": markets,
            "polls.csv": raw["polls"],
            "election_admin.csv": raw["election_admin"],
            "candidate_forecast.csv": candidate_scores,
            "model_scenarios.csv": scenarios,
            "market_comparison.csv": market_comparison,
            "wager_value_table.csv": wager,
            "data_quality_report.csv": quality,
            "source_manifest.csv": source_manifest,
            "source_registry_status.csv": source_status,
            "release_status.csv": release,
            "snapshot_manifest.csv": snapshot,
        }
        for filename, frame in outputs.items():
            write_csv(filename, frame)
        (PROCESSED_DIR / "model_output.json").write_text(json.dumps(model_output, indent=2), encoding="utf-8")
        (PROCESSED_DIR / "model_card.md").write_text(build_model_card(candidate_scores, release, diagnostics), encoding="utf-8")
        (PROCESSED_DIR / "audit_report.md").write_text(build_audit_report(candidate_scores, release, wager), encoding="utf-8")

    return {
        "candidate_scores": candidate_scores,
        "scenarios": scenarios,
        "release_status": release,
        "wager_value_table": wager,
        "terminal_report": render_terminal_report(candidate_scores, release, wager, scenarios),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=str(DEFAULT_AS_OF.date()))
    parser.add_argument(
        "--write-output",
        action="store_true",
        help="Write processed CSV/JSON/Markdown outputs to disk instead of terminal-only mode.",
    )
    args = parser.parse_args()
    result = main(args.as_of, write_outputs=args.write_output)
    print(result["terminal_report"])
