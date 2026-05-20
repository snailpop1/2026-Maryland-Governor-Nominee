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
FORECAST_VERSION = "2.0"
SIMULATION_COUNT = 5000
MODEL_TEMPERATURE = 2.35


PRIOR_PROBABILITIES = {
    "Dan Cox": 0.29,
    "Ed Hale": 0.22,
    "John A. Myrick": 0.14,
    "Shannon Wright": 0.08,
    "Carl A. Brunner Jr.": 0.08,
    "Douglas Larcomb": 0.06,
    "L. D. Burkindine": 0.04,
    "Nancy Jane Taylor": 0.04,
    "Michael Oakes": 0.03,
}


FEATURE_WEIGHTS = {
    "organization_score": 0.95,
    "finance_score": 1.15,
    "event_score": 0.90,
    "geography_score": 0.70,
    "market_score": 0.20,
    "environment_score": 0.55,
}


SCENARIOS = [
    "baseline",
    "cox_repeat_base_consolidation",
    "hale_money_moderate_surge",
    "myrick_organization_surge",
    "wright_debate_surge",
    "fragmented_field_plurality",
    "late_trump_or_major_gop_endorsement",
    "mail_ballot_admin_uncertainty",
    "low_turnout",
    "high_turnout",
]


def configure_run(as_of: object | None = None) -> pd.Timestamp:
    global AS_OF
    AS_OF = pd.Timestamp(as_of).normalize() if as_of is not None else DEFAULT_AS_OF
    return AS_OF


def read_raw() -> dict[str, pd.DataFrame]:
    return {path.stem: pd.read_csv(path) for path in RAW_DIR.glob("*.csv")}


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


def safe_log1p(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.log1p(max(float(value), 0.0)))


def softmax(values: pd.Series, temperature: float = 1.0) -> pd.Series:
    scaled = values.astype(float) / max(temperature, 1e-9)
    scaled = scaled - scaled.max()
    exp_values = np.exp(scaled)
    total = exp_values.sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(np.repeat(1.0 / len(values), len(values)), index=values.index)
    return pd.Series(exp_values / total, index=values.index)


def zscore(series: pd.Series, clip: float = 2.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    scaled = (values - mean) / std
    return scaled.clip(lower=-clip, upper=clip).fillna(0.0)


def settings_map(settings: pd.DataFrame) -> dict[str, float]:
    return {row["setting"]: float(row["value"]) for _, row in settings.iterrows()}


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
    out["filed_age_log"] = np.log1p(out["days_on_file"].clip(lower=0))
    out["ticket_freshness_flag"] = np.where(out["days_on_file"] >= 30, "mature", "newer")
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
    out["county_weight"] = out["total_2022_gop_votes"] / out["total_2022_gop_votes"].sum()
    return out


def process_finance(finance: pd.DataFrame) -> pd.DataFrame:
    out = finance.copy()
    out["report_date"] = pd.to_datetime(out["report_date"])
    for column in ["raised_usd", "spent_usd", "cash_on_hand_usd"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["days_old"] = (AS_OF - out["report_date"]).dt.days.clip(lower=0)
    out["freshness_weight"] = np.exp(-np.log(2) * out["days_old"] / 14)
    out["cash_score"] = np.log1p(out["cash_on_hand_usd"].fillna(0)) / np.log1p(125_000)
    out["raised_score"] = np.log1p(out["raised_usd"].fillna(0)) / np.log1p(250_000)
    out["spend_penalty"] = np.log1p(out["spent_usd"].fillna(0)) / np.log1p(200_000)
    out["finance_signal"] = (
        0.60 * out["cash_score"] + 0.45 * out["raised_score"] - 0.20 * out["spend_penalty"]
    ) * out["freshness_weight"]
    out["finance_signal"] = out["finance_signal"].fillna(0.0)
    return out


def process_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    out["impact_points"] = pd.to_numeric(out["impact_points"], errors="raise")
    out["days_old"] = (AS_OF - out["event_date"]).dt.days.clip(lower=0)
    out["event_weight"] = np.exp(-np.log(2) * out["days_old"] / 60)
    out["weighted_impact_points"] = out["impact_points"] * out["event_weight"]
    out["positive_signal"] = np.where(out["weighted_impact_points"] > 0, out["weighted_impact_points"], 0.0)
    out["negative_signal"] = np.where(out["weighted_impact_points"] < 0, -out["weighted_impact_points"], 0.0)
    return out


def process_markets(markets: pd.DataFrame, settings: dict[str, float]) -> pd.DataFrame:
    out = markets.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"])
    for column in ["yes_price", "implied_probability", "volume_usd", "liquidity_usd"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["days_stale"] = (AS_OF - out["snapshot_date"]).dt.days.clip(lower=0)
    out["is_active_candidate"] = out["candidate"].isin(PRIOR_PROBABILITIES)
    active = out[out["is_active_candidate"]].copy()
    total = active["implied_probability"].sum()
    active["normalized_probability"] = active["implied_probability"] / total if total else np.nan
    out = out.merge(
        active[["candidate", "normalized_probability"]],
        on="candidate",
        how="left",
    )
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
    out["market_signal"] = np.where(
        out["normalized_probability"].notna(),
        np.log(out["normalized_probability"].clip(lower=1e-6)),
        np.nan,
    )
    return out


def process_polls(polls: pd.DataFrame) -> pd.DataFrame:
    out = polls.copy()
    if out.empty:
        return out.assign(
            release_date=pd.Series(dtype="datetime64[ns]"),
            sample_size=pd.Series(dtype="float"),
            pct=pd.Series(dtype="float"),
            poll_signal=pd.Series(dtype="float"),
            poll_weight=pd.Series(dtype="float"),
        )
    out["release_date"] = pd.to_datetime(out["release_date"])
    out["sample_size"] = pd.to_numeric(out["sample_size"], errors="coerce")
    out["pct"] = pd.to_numeric(out["pct"], errors="coerce")
    out["poll_signal"] = out["pct"] / 100.0
    out["days_old"] = (AS_OF - out["release_date"]).dt.days.clip(lower=0)
    out["poll_weight"] = np.exp(-np.log(2) * out["days_old"] / 45) * np.sqrt(out["sample_size"].fillna(0).clip(lower=0) / 800)
    return out


def process_environment_polls(environment_polls: pd.DataFrame) -> pd.DataFrame:
    out = environment_polls.copy()
    if out.empty:
        return out.assign(
            release_date=pd.Series(dtype="datetime64[ns]"),
            sample_size=pd.Series(dtype="float"),
            pct=pd.Series(dtype="float"),
            poll_weight=pd.Series(dtype="float"),
            signed_pct=pd.Series(dtype="float"),
            weighted_signed_pct=pd.Series(dtype="float"),
            measure_group=pd.Series(dtype="object"),
        )
    out["release_date"] = pd.to_datetime(out["release_date"])
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out["sample_size"] = pd.to_numeric(out["sample_size"], errors="coerce")
    out["pct"] = pd.to_numeric(out["pct"], errors="coerce")
    out["days_old"] = (AS_OF - out["release_date"]).dt.days.clip(lower=0)
    out["poll_weight"] = np.exp(-np.log(2) * out["days_old"] / 120) * np.sqrt(out["sample_size"].fillna(0).clip(lower=0) / 800)
    out["measure_group"] = out["measure"].astype(str).str.lower().str.strip()

    def option_sign(measure: str, option: str) -> float:
        m = str(measure).lower().strip()
        o = str(option).lower().strip()
        if m == "job_approval":
            if o in {"approve", "strongly approve", "somewhat approve"}:
                return 1.0
            if o in {"disapprove", "strongly disapprove", "somewhat disapprove"}:
                return -1.0
            return 0.0
        if m == "direction_of_state":
            if o in {"right_direction", "right track", "right"}:
                return 1.0
            if o in {"wrong_direction", "wrong track", "wrong"}:
                return -1.0
            return 0.0
        if m == "matchup":
            if o in {"reelect_moore", "moore", "democratic incumbent"}:
                return 1.0
            if o in {"republican_challenger", "republican candidate", "challenger"}:
                return -1.0
            return 0.0
        return 0.0

    out["sign"] = out.apply(lambda row: option_sign(row["measure_group"], row["option"]), axis=1)
    out["signed_pct"] = out["pct"] * out["sign"]
    out["weighted_signed_pct"] = out["signed_pct"] * out["poll_weight"]
    return out


def build_candidate_poll_summary(candidate_polls: pd.DataFrame) -> pd.DataFrame:
    if candidate_polls.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "poll_count",
                "weighted_poll_pct",
                "latest_poll_pct",
                "latest_release_date",
                "total_sample_size",
                "avg_poll_weight",
                "total_poll_weight",
            ]
        )
    out = candidate_polls.copy()
    out["weighted_pct"] = out["pct"] * out["poll_weight"]
    summary = out.groupby("candidate", as_index=False).agg(
        poll_count=("pct", "count"),
        weighted_pct_sum=("weighted_pct", "sum"),
        total_poll_weight=("poll_weight", "sum"),
        latest_poll_pct=("pct", "last"),
        latest_release_date=("release_date", "max"),
        total_sample_size=("sample_size", "sum"),
        avg_poll_weight=("poll_weight", "mean"),
    )
    summary["weighted_poll_pct"] = np.where(
        summary["total_poll_weight"].gt(0),
        summary["weighted_pct_sum"] / summary["total_poll_weight"],
        np.nan,
    )
    summary = summary.drop(columns=["weighted_pct_sum"])
    return summary


def build_environment_poll_summary(environment_polls: pd.DataFrame) -> pd.DataFrame:
    if environment_polls.empty:
        return pd.DataFrame(
            columns=[
                "measure_group",
                "poll_count",
                "weighted_gap_pct",
                "weighted_positive_pct",
                "weighted_negative_pct",
                "latest_release_date",
                "total_sample_size",
                "total_poll_weight",
            ]
        )
    rows = []
    for measure, group in environment_polls.groupby("measure_group"):
        total_weight = float(group["poll_weight"].sum())
        if measure == "job_approval":
            positive = group.loc[group["sign"].gt(0), "weighted_signed_pct"].sum()
            negative = -group.loc[group["sign"].lt(0), "weighted_signed_pct"].sum()
        elif measure == "direction_of_state":
            positive = group.loc[group["sign"].gt(0), "weighted_signed_pct"].sum()
            negative = -group.loc[group["sign"].lt(0), "weighted_signed_pct"].sum()
        elif measure == "matchup":
            positive = group.loc[group["sign"].gt(0), "weighted_signed_pct"].sum()
            negative = -group.loc[group["sign"].lt(0), "weighted_signed_pct"].sum()
        else:
            positive = group.loc[group["sign"].gt(0), "weighted_signed_pct"].sum()
            negative = -group.loc[group["sign"].lt(0), "weighted_signed_pct"].sum()
        gap = (positive - negative) / total_weight if total_weight else 0.0
        rows.append(
            {
                "measure_group": measure,
                "poll_count": int(len(group)),
                "weighted_positive_pct": float(positive / total_weight) if total_weight else 0.0,
                "weighted_negative_pct": float(negative / total_weight) if total_weight else 0.0,
                "weighted_gap_pct": float(gap),
                "latest_release_date": str(group["release_date"].max().date()),
                "total_sample_size": float(group["sample_size"].sum()),
                "total_poll_weight": total_weight,
            }
        )
    return pd.DataFrame(rows)


def build_environment_context(environment_summary: pd.DataFrame) -> dict[str, float]:
    if environment_summary.empty:
        return {
            "approval_gap": 0.0,
            "direction_gap": 0.0,
            "matchup_gap": 0.0,
            "anti_incumbent_pressure": 0.0,
        }
    summary = {row["measure_group"]: row for _, row in environment_summary.iterrows()}
    approval_gap = float(summary.get("job_approval", {}).get("weighted_gap_pct", 0.0))
    direction_gap = float(summary.get("direction_of_state", {}).get("weighted_gap_pct", 0.0))
    matchup_gap = float(summary.get("matchup", {}).get("weighted_gap_pct", 0.0))
    anti_incumbent_pressure = (-0.45 * approval_gap - 0.35 * direction_gap - 0.20 * matchup_gap) / 100.0
    return {
        "approval_gap": approval_gap,
        "direction_gap": direction_gap,
        "matchup_gap": matchup_gap,
        "anti_incumbent_pressure": anti_incumbent_pressure,
    }


def process_election_admin(election_admin: pd.DataFrame) -> pd.DataFrame:
    out = election_admin.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["days_to_event"] = (out["event_date"] - AS_OF).dt.days
    out["is_future_event"] = out["days_to_event"].ge(0)
    return out


def candidate_prior_probability(candidate: str) -> float:
    return PRIOR_PROBABILITIES.get(candidate, 0.01)


def build_candidate_model(
    candidates: pd.DataFrame,
    primary_2022: pd.DataFrame,
    county_2022: pd.DataFrame,
    finance: pd.DataFrame,
    events: pd.DataFrame,
    markets: pd.DataFrame,
    polls: pd.DataFrame,
    environment_context: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    active = candidates[candidates["is_active_republican"]].copy()
    out = active[["candidate", "lieutenant_governor", "home_jurisdiction", "days_on_file", "has_website", "has_social", "filed_age_log"]].copy()

    statewide_total = float(primary_2022["total_votes"].sum())
    cox_votes = float(primary_2022.loc[primary_2022["candidate"].eq("Dan Cox"), "total_votes"].iloc[0])
    cox_2022_share = cox_votes / statewide_total
    schulz_votes = float(primary_2022.loc[primary_2022["candidate"].eq("Kelly Schulz"), "total_votes"].iloc[0])
    schulz_2022_share = schulz_votes / statewide_total

    county_strength = county_2022[["county", "jurisdiction_key", "total_2022_gop_votes", "cox_share"]].copy()
    county_strength["county_strength"] = (
        np.log1p(county_strength["total_2022_gop_votes"]) * county_strength["cox_share"]
    )
    county_lookup = county_strength.groupby("jurisdiction_key", as_index=False).agg(
        county_strength=("county_strength", "mean"),
        total_2022_gop_votes=("total_2022_gop_votes", "sum"),
        cox_share=("cox_share", "mean"),
    )
    out["jurisdiction_key"] = out["home_jurisdiction"].map(jurisdiction_key)
    out = out.merge(county_lookup, on="jurisdiction_key", how="left")

    finance_summary = finance.groupby("candidate", as_index=False).agg(
        finance_signal=("finance_signal", "sum"),
        finance_cash=("cash_on_hand_usd", "sum"),
        finance_raised=("raised_usd", "sum"),
        finance_spent=("spent_usd", "sum"),
        finance_days_old=("days_old", "min"),
    )
    out = out.merge(finance_summary, on="candidate", how="left")

    event_summary = events.groupby("candidate", as_index=False).agg(
        weighted_impact_points=("weighted_impact_points", "sum"),
        positive_signal=("positive_signal", "sum"),
        negative_signal=("negative_signal", "sum"),
        event_count=("event_type", "count"),
    )
    out = out.merge(event_summary, on="candidate", how="left")

    market_summary = markets[markets["is_active_candidate"]].groupby("candidate", as_index=False).agg(
        normalized_probability=("normalized_probability", "first"),
        yes_price=("yes_price", "first"),
        market_days_stale=("days_stale", "first"),
        liquidity_usd=("liquidity_usd", "first"),
        market_signal=("market_signal", "first"),
    )
    out = out.merge(market_summary, on="candidate", how="left")

    poll_summary = polls.groupby("candidate", as_index=False).agg(
        poll_signal=("poll_signal", "mean"),
        poll_sample_size=("sample_size", "sum"),
    )
    out = out.merge(poll_summary, on="candidate", how="left")

    out["finance_signal"] = out["finance_signal"].fillna(0.0)
    out["finance_cash"] = out["finance_cash"].fillna(0.0)
    out["finance_raised"] = out["finance_raised"].fillna(0.0)
    out["finance_spent"] = out["finance_spent"].fillna(0.0)
    out["weighted_impact_points"] = out["weighted_impact_points"].fillna(0.0)
    out["positive_signal"] = out["positive_signal"].fillna(0.0)
    out["negative_signal"] = out["negative_signal"].fillna(0.0)
    out["event_count"] = out["event_count"].fillna(0.0)
    out["county_strength"] = out["county_strength"].fillna(out["county_strength"].median())
    out["total_2022_gop_votes"] = out["total_2022_gop_votes"].fillna(out["total_2022_gop_votes"].median())
    out["cox_share"] = out["cox_share"].fillna(out["cox_share"].median())
    out["normalized_probability"] = out["normalized_probability"].fillna(0.0)
    out["poll_signal"] = out["poll_signal"].fillna(0.0)
    out["poll_sample_size"] = out["poll_sample_size"].fillna(0.0)
    out["market_signal"] = out["market_signal"].fillna(np.log(0.01))
    out["market_days_stale"] = out["market_days_stale"].fillna(999.0)

    out["base_prior_probability"] = out["candidate"].map(candidate_prior_probability)
    out["base_prior_log_score"] = np.log(out["base_prior_probability"])

    out["organization_score"] = (
        0.55 * zscore(out["filed_age_log"])
        + 0.30 * out["has_website"].astype(float)
        + 0.15 * out["has_social"].astype(float)
    )

    out["finance_score"] = zscore(out["finance_signal"])
    out["event_score"] = zscore(out["weighted_impact_points"] + 0.5 * out["positive_signal"] - 0.4 * out["negative_signal"])
    out["geography_score"] = zscore(np.log1p(out["county_strength"]) + 0.25 * np.log1p(out["total_2022_gop_votes"]))

    market_signal = out["normalized_probability"].replace(0, np.nan)
    market_bridge = np.where(market_signal.notna(), np.log(market_signal.clip(lower=1e-6)), out["market_signal"])
    out["market_score"] = zscore(pd.Series(market_bridge, index=out.index))
    out["poll_score"] = zscore(out["poll_signal"])

    environment_fit = {
        "Dan Cox": -0.10,
        "Ed Hale": 0.22,
        "John A. Myrick": 0.14,
        "Shannon Wright": 0.05,
        "Carl A. Brunner Jr.": 0.04,
        "Douglas Larcomb": 0.03,
        "L. D. Burkindine": 0.03,
        "Nancy Jane Taylor": 0.03,
        "Michael Oakes": 0.03,
    }
    out["environment_fit"] = out["candidate"].map(environment_fit).fillna(0.0)
    out["environment_pressure"] = float(environment_context.get("anti_incumbent_pressure", 0.0))
    out["environment_score"] = out["environment_fit"] * out["environment_pressure"]

    out["history_bonus"] = np.where(out["candidate"].eq("Dan Cox"), 0.65, 0.0)
    out["history_bonus"] += np.where(out["candidate"].eq("Shannon Wright"), 0.15, 0.0)
    out["history_bonus"] += np.where(out["candidate"].eq("Ed Hale"), 0.15, 0.0)
    out["history_bonus"] += np.where(out["candidate"].eq("John A. Myrick"), 0.08, 0.0)

    out["data_completeness_penalty"] = (
        np.where(out["has_website"], 0.0, -0.10)
        + np.where(out["has_social"], 0.0, -0.07)
        + np.where(out["finance_cash"].gt(0), 0.0, -0.18)
    )

    out["feature_total"] = (
        out["base_prior_log_score"]
        + 0.55 * out["organization_score"]
        + FEATURE_WEIGHTS["finance_score"] * out["finance_score"]
        + FEATURE_WEIGHTS["event_score"] * out["event_score"]
        + FEATURE_WEIGHTS["geography_score"] * out["geography_score"]
        + FEATURE_WEIGHTS["market_score"] * out["market_score"]
        + FEATURE_WEIGHTS["environment_score"] * out["environment_score"]
        + 0.80 * out["poll_score"]
        + out["history_bonus"]
        + out["data_completeness_penalty"]
    )

    out["model_score"] = out["feature_total"] / MODEL_TEMPERATURE
    out["baseline_probability"] = softmax(out["model_score"])

    rng = np.random.default_rng(
        int(hashlib.sha256(f"{AS_OF.date()}|{FORECAST_VERSION}".encode("utf-8")).hexdigest()[:16], 16)
    )
    sims = simulate_probability_draws(out, rng)
    out["forecast_probability"] = sims.mean(axis=0).to_numpy()
    out["probability_p10"] = sims.quantile(0.10, axis=0).to_numpy()
    out["probability_p90"] = sims.quantile(0.90, axis=0).to_numpy()
    out["probability_std"] = sims.std(axis=0).to_numpy()
    out["confidence_band_width"] = out["probability_p90"] - out["probability_p10"]
    out = out.sort_values("forecast_probability", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["margin_to_leader"] = out["forecast_probability"].iloc[0] - out["forecast_probability"]

    diagnostics = {
        "cox_2022_share": cox_2022_share,
        "schulz_2022_share": schulz_2022_share,
        "statewide_2022_gop_primary_votes": statewide_total,
        "poll_rows": float(len(polls)),
        "active_candidate_count": float(len(out)),
        "score_entropy": float(-(out["forecast_probability"] * np.log(out["forecast_probability"])).sum()),
        "top_margin": float(out.iloc[0]["forecast_probability"] - out.iloc[1]["forecast_probability"]) if len(out) > 1 else 1.0,
        "temperature": MODEL_TEMPERATURE,
        "environment_pressure": float(environment_context.get("anti_incumbent_pressure", 0.0)),
    }
    return out, diagnostics


def simulate_probability_draws(features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    std_map = {
        "organization_score": 0.18,
        "finance_score": np.where(features["finance_cash"].gt(0), 0.15, 0.35),
        "event_score": np.where(features["event_count"].gt(0), 0.16, 0.28),
        "geography_score": 0.12,
        "market_score": np.where(features["market_days_stale"].lt(5), 0.22, 0.45),
        "poll_score": np.where(features["poll_sample_size"].gt(0), 0.10, 0.0),
    }
    draws: list[pd.Series] = []
    base = features["feature_total"].to_numpy(dtype=float)
    for _ in range(SIMULATION_COUNT):
        noise = (
            rng.normal(0, std_map["organization_score"], size=len(features))
            + rng.normal(0, std_map["finance_score"])
            + rng.normal(0, std_map["event_score"])
            + rng.normal(0, std_map["geography_score"], size=len(features))
            + rng.normal(0, std_map["market_score"])
            + rng.normal(0, std_map["poll_score"])
            + rng.normal(0, 0.10, size=len(features))
        )
        candidate_shift = rng.normal(0, np.where(features["candidate"].eq("Dan Cox"), 0.08, 0.12))
        latent = base + noise + candidate_shift
        probabilities = softmax(pd.Series(latent, index=features.index), temperature=MODEL_TEMPERATURE)
        draws.append(probabilities)
    return pd.concat(draws, axis=1).T


def scenario_adjustments(base_scores: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = base_scores.copy()
    out["scenario_score"] = out["feature_total"]
    if scenario == "cox_repeat_base_consolidation":
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] += 1.05
        out.loc[out["candidate"].isin(["Ed Hale", "John A. Myrick"]), "scenario_score"] -= 0.35
    elif scenario == "hale_money_moderate_surge":
        out.loc[out["candidate"].eq("Ed Hale"), "scenario_score"] += 2.00
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] -= 0.40
    elif scenario == "myrick_organization_surge":
        out.loc[out["candidate"].eq("John A. Myrick"), "scenario_score"] += 1.60
        out.loc[out["candidate"].eq("Shannon Wright"), "scenario_score"] += 0.30
    elif scenario == "wright_debate_surge":
        out.loc[out["candidate"].eq("Shannon Wright"), "scenario_score"] += 1.35
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] -= 0.20
    elif scenario == "fragmented_field_plurality":
        out["scenario_score"] = out["scenario_score"] * 0.88 + 0.05
    elif scenario == "late_trump_or_major_gop_endorsement":
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] += 1.10
        out.loc[out["candidate"].eq("Ed Hale"), "scenario_score"] -= 0.30
    elif scenario == "mail_ballot_admin_uncertainty":
        out["scenario_score"] = out["scenario_score"] * 0.94 - 0.06
    elif scenario == "low_turnout":
        out.loc[out["candidate"].eq("Dan Cox"), "scenario_score"] += 0.90
        out.loc[out["candidate"].eq("Shannon Wright"), "scenario_score"] += 0.25
    elif scenario == "high_turnout":
        out.loc[out["candidate"].isin(["Ed Hale", "John A. Myrick"]), "scenario_score"] += 0.90
    exp_score = np.exp(out["scenario_score"] / MODEL_TEMPERATURE)
    out["scenario_probability"] = exp_score / exp_score.sum()
    out["scenario"] = scenario
    return out[["scenario", "candidate", "scenario_probability"]]


def build_scenarios(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows = [scenario_adjustments(candidate_scores, scenario) for scenario in SCENARIOS]
    return pd.concat(rows, ignore_index=True)


def build_market_comparison(candidate_scores: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    out = candidate_scores[["candidate", "forecast_probability", "probability_p10", "probability_p90"]].merge(
        markets[
            [
                "candidate",
                "yes_price",
                "implied_probability",
                "normalized_probability",
                "stale_price_flag",
                "liquidity_warning",
                "settlement_warning",
                "days_stale",
                "is_active_candidate",
            ]
        ],
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
        blocking.append("no_credible_candidate_polling")
        penalties.append("candidate_polling_component_unavailable")
    if set(candidate_scores["candidate"]) - set(raw["finance"]["candidate"]):
        blocking.append("finance_incomplete_for_full_field")
        penalties.append("finance_missing_for_some_active_candidates")
    if markets["stale_price_flag"].any():
        blocking.append("stale_market_price")
    if (markets["settlement_warning"] != "ok").any():
        blocking.append("verify_contract_terms")
    if candidate_scores["forecast_probability"].max() < 0.45:
        penalties.append("plurality_field_high_uncertainty")
    leader = candidate_scores.iloc[0]["candidate"]
    leader_scenarios = scenarios[scenarios["candidate"].eq(leader)]
    if leader_scenarios["scenario_probability"].min() < 0.50:
        blocking.append("edge_not_durable_across_scenarios")
    if candidate_scores.iloc[0]["confidence_band_width"] > 0.30:
        penalties.append("leader_probability_interval_wide")

    score = 100
    score -= 24 if "no_credible_candidate_polling" in blocking else 0
    score -= 16 if "finance_incomplete_for_full_field" in blocking else 0
    score -= 12 if "stale_market_price" in blocking else 0
    score -= 10 if "edge_not_durable_across_scenarios" in blocking else 0
    score -= 8 if "verify_contract_terms" in blocking else 0
    score -= 6 * len(set(penalties) - {"candidate_polling_component_unavailable", "finance_missing_for_some_active_candidates"})
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
                "forecast_probability_p10": row["probability_p10"],
                "forecast_probability_p90": row["probability_p90"],
                "market_probability": row["market_probability"],
                "edge_vs_market": edge,
                "worst_case_scenario_probability": worst_case,
                "required_edge": settings["min_required_edge"],
                "value_flag": "candidate_value" if decision_eligible else "no_bet",
                "decision_eligible": decision_eligible,
                "capped_exposure_fraction": min(settings["max_exposure_fraction"], max(0.0, float(edge) / 2.0 if pd.notna(edge) else 0.0))
                if decision_eligible
                else 0.0,
                "block_reasons": ";".join(dict.fromkeys(block_reasons)),
            }
        )
    return pd.DataFrame(rows)


def build_data_quality(raw: dict[str, pd.DataFrame], markets: pd.DataFrame, polls: pd.DataFrame) -> pd.DataFrame:
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
        freshness_days = None
        if name == "markets" and not markets.empty:
            stale = bool(markets["stale_price_flag"].any())
            freshness_days = int(markets["days_stale"].min())
        if name == "polls":
            freshness_days = 0 if polls.empty else int((AS_OF - polls["release_date"].max()).days)
        completeness = 1.0
        if name in {"candidates", "finance", "events", "markets"}:
            completeness -= 0.15 * int(missing_sources > 0)
            completeness -= 0.10 * int(missing_notes > 0)
        rows.append(
            {
                "dataset_name": name,
                "row_count": len(frame),
                "missing_source_count": missing_sources,
                "missing_notes_count": missing_notes,
                "stale": stale,
                "freshness_days": freshness_days if freshness_days is not None else "",
                "required_for_publish": name in {"candidates", "historical_2022_primary_results", "finance", "polls", "markets", "environment_polls"},
                "completeness_score": round(max(0.0, completeness), 2),
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
    interval = f"{leader['probability_p10']:.1%} to {leader['probability_p90']:.1%}"
    return f"""# Model Card

## Intended Use

This package estimates the June 23, 2026 Maryland Republican gubernatorial primary as a probabilistic research forecast. It is not a guarantee or financial advice.

## Headline

- As of: `{AS_OF.date()}`
- Top candidate: `{leader['candidate']}`
- Top candidate fair probability: `{leader['forecast_probability']:.1%}`
- Top candidate interval: `{interval}`
- Release status: `{release.iloc[0]['release_status']}`
- Betting eligible: `{bool(release.iloc[0]['betting_eligible'])}`

## Inputs

- Official May 2026 SBE ballot and candidate list.
- Official 2022 Maryland GOP governor primary statewide and county results.
- Current campaign-finance reporting summarized from MDCRIS and reporting captures.
- Public Maryland environment polling on Wes Moore approval, state direction, and generic governor matchups.
- Debate, endorsement, ballot-administration, and organizational activity.
- Prediction markets as a weak external price check only.

## Known Limits

- No credible public candidate-level poll was found as of the run date.
- Finance is incomplete for several minor active candidates.
- Market prices are stale and include non-active names.
- Reliability score: `{release.iloc[0]['reliability_score']}/100`.
"""


def build_audit_report(candidate_scores: pd.DataFrame, release: pd.DataFrame, wager: pd.DataFrame) -> str:
    leader = candidate_scores.iloc[0]
    ranking_lines = ["| rank | candidate | forecast_probability | p10 | p90 |", "| ---: | --- | ---: | ---: | ---: |"]
    for _, row in candidate_scores.iterrows():
        ranking_lines.append(
            f"| {int(row['rank'])} | {row['candidate']} | {float(row['forecast_probability']):.3f} | {float(row['probability_p10']):.3f} | {float(row['probability_p90']):.3f} |"
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
- Leader interval: `{leader['probability_p10']:.3f}` to `{leader['probability_p90']:.3f}`
- Active candidates modeled: `{len(candidate_scores)}`

## Blocking Issues

{release.iloc[0]['blocking_issues'] or 'None'}

## Betting Decision Summary

All current market rows are safety-gated to no bet unless data freshness, liquidity, settlement terms, and scenario durability all pass. The current blocker remains the absence of credible candidate-level polling.

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
        f"Interval: {leader['probability_p10']:.1%} to {leader['probability_p90']:.1%}",
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
    polls = process_polls(raw["polls"])
    environment_polls = process_environment_polls(raw["environment_polls"])
    candidate_poll_summary = build_candidate_poll_summary(polls)
    environment_poll_summary = build_environment_poll_summary(environment_polls)
    environment_context = build_environment_context(environment_poll_summary)
    election_admin = process_election_admin(raw["election_admin"])
    candidate_scores, diagnostics = build_candidate_model(
        candidates,
        raw["historical_2022_primary_results"],
        county_2022,
        finance,
        events,
        markets,
        polls,
        environment_context,
    )
    scenarios = build_scenarios(candidate_scores)
    market_comparison = build_market_comparison(candidate_scores, markets)
    release = build_release_status(raw, candidate_scores, markets, scenarios)
    wager = build_wager_value_table(market_comparison, release, scenarios, settings)
    quality = build_data_quality(raw, markets, polls)
    source_manifest = build_source_manifest()
    source_status = build_source_registry_status(raw)
    model_output = {
        "version": FORECAST_VERSION,
        "as_of": str(AS_OF.date()),
        "primary_date": str(PRIMARY_DATE.date()),
        "headline_forecast": {
            "candidate": candidate_scores.iloc[0]["candidate"],
            "probability": round(float(candidate_scores.iloc[0]["forecast_probability"]), 4),
            "interval": [
                round(float(candidate_scores.iloc[0]["probability_p10"]), 4),
                round(float(candidate_scores.iloc[0]["probability_p90"]), 4),
            ],
        },
        "candidate_probabilities": {
            row["candidate"]: round(float(row["forecast_probability"]), 4)
            for _, row in candidate_scores.iterrows()
        },
        "candidate_intervals": {
            row["candidate"]: {
                "p10": round(float(row["probability_p10"]), 4),
                "p90": round(float(row["probability_p90"]), 4),
            }
            for _, row in candidate_scores.iterrows()
        },
        "release_status": release.iloc[0].to_dict(),
        "model_health": diagnostics,
        "environment_context": environment_context,
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
            "polls.csv": polls,
            "environment_polls.csv": environment_polls,
            "election_admin.csv": election_admin,
            "candidate_poll_summary.csv": candidate_poll_summary,
            "environment_poll_summary.csv": environment_poll_summary,
            "environment_context.csv": pd.DataFrame([environment_context]),
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
