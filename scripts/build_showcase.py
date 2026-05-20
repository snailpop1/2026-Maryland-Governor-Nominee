from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SHOWCASE_DIR = ROOT / "showcase"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    scores = pd.read_csv(PROCESSED_DIR / "candidate_forecast.csv")
    release = pd.read_csv(PROCESSED_DIR / "release_status.csv").iloc[0]
    wager = pd.read_csv(PROCESSED_DIR / "wager_value_table.csv")
    scenarios = pd.read_csv(PROCESSED_DIR / "model_scenarios.csv")
    quality = pd.read_csv(PROCESSED_DIR / "data_quality_report.csv")
    market = pd.read_csv(PROCESSED_DIR / "market_comparison.csv")
    candidate_polls = pd.read_csv(PROCESSED_DIR / "candidate_poll_summary.csv")
    environment_polls = pd.read_csv(PROCESSED_DIR / "environment_polls.csv")
    environment_summary = pd.read_csv(PROCESSED_DIR / "environment_poll_summary.csv")
    environment_context = pd.read_csv(PROCESSED_DIR / "environment_context.csv").iloc[0]
    leader = scores.iloc[0]

    ranking_rows = "\n".join(
        f"<tr><td>{int(row.rank)}</td><td>{escape(row.candidate)}</td><td>{pct(row.forecast_probability)}</td><td>{pct(row.probability_p10)} - {pct(row.probability_p90)}</td><td>{row.model_score:.2f}</td></tr>"
        for row in scores.itertuples(index=False)
    )
    wager_rows = "\n".join(
        f"<tr><td>{escape(row.candidate)}</td><td>{escape(row.value_flag)}</td><td>{pct(row.forecast_probability)}</td><td>{escape(str(row.block_reasons))}</td></tr>"
        for row in wager.itertuples(index=False)
    )
    scenario_rows = "\n".join(
        f"<tr><td>{escape(row.scenario)}</td><td>{escape(row.candidate)}</td><td>{pct(row.scenario_probability)}</td></tr>"
        for row in scenarios.sort_values(["scenario", "scenario_probability"], ascending=[True, False]).groupby("scenario").head(3).itertuples(index=False)
    )
    quality_rows = "\n".join(
        f"<tr><td>{escape(row.dataset_name)}</td><td>{int(row.row_count)}</td><td>{escape(str(row.stale))}</td><td>{int(row.missing_source_count)}</td><td>{float(row.completeness_score):.2f}</td></tr>"
        for row in quality.itertuples(index=False)
    )
    candidate_poll_rows = "\n".join(
        f"<tr><td>{escape(row.candidate)}</td><td>{int(row.poll_count)}</td><td>{pct(row.weighted_poll_pct / 100) if pd.notna(row.weighted_poll_pct) else 'n/a'}</td><td>{pct(row.latest_poll_pct / 100) if pd.notna(row.latest_poll_pct) else 'n/a'}</td><td>{escape(str(row.latest_release_date))}</td></tr>"
        for row in candidate_polls.itertuples(index=False)
    )
    environment_summary_rows = "\n".join(
        f"<tr><td>{escape(row.measure_group)}</td><td>{int(row.poll_count)}</td><td>{float(row.weighted_gap_pct):+.2f}</td><td>{escape(str(row.latest_release_date))}</td></tr>"
        for row in environment_summary.itertuples(index=False)
    )
    market_rows = "\n".join(
        f"<tr><td>{escape(row.candidate)}</td><td>{pct(row.forecast_probability)}</td><td>{pct(row.market_probability) if pd.notna(row.market_probability) else 'n/a'}</td><td>{'' if pd.isna(row.edge_vs_market) else f'{row.edge_vs_market:+.3f}'}</td></tr>"
        for row in market.itertuples(index=False)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Maryland GOP Governor Nominee Forecast</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: linear-gradient(180deg, #eef2f7 0%, #ffffff 32%, #eef2f7 100%); color: #17202a; }}
    header {{ background: #16213a; color: white; padding: 28px 36px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 48px; }}
    .headline {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .panel {{ background: white; border: 1px solid #d9dee8; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 6px 22px rgba(17, 24, 39, 0.06); }}
    .metric {{ font-size: 30px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e5e8ef; padding: 10px 8px; vertical-align: top; }}
    th {{ color: #465366; font-size: 12px; text-transform: uppercase; }}
    .status {{ display: inline-block; padding: 5px 9px; border-radius: 999px; background: #fff1c2; color: #5a4100; font-weight: 700; }}
    .subtle {{ color: #5f6b7a; font-size: 13px; }}
    @media (max-width: 920px) {{ .headline {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 640px) {{ .headline {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Maryland GOP Governor Nominee Forecast</h1>
    <p>As-of {escape(str(leader.last_updated))}. Hybrid forecast package with conservative wagering gate.</p>
  </header>
  <main>
    <section class="headline">
      <div class="panel"><div>Leader</div><div class="metric">{escape(leader.candidate)}</div></div>
      <div class="panel"><div>Fair Probability</div><div class="metric">{pct(float(leader.forecast_probability))}</div></div>
      <div class="panel"><div>Interval</div><div class="metric">{pct(float(leader.probability_p10))} - {pct(float(leader.probability_p90))}</div></div>
      <div class="panel"><div>Official Bet Status</div><div class="metric"><span class="status">{escape(str(release.release_status))} / no bet</span></div></div>
    </section>
    <section class="panel">
      <h2>Candidate Ranking</h2>
      <table><thead><tr><th>Rank</th><th>Candidate</th><th>Probability</th><th>Interval</th><th>Score</th></tr></thead><tbody>{ranking_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Scenario Leaders</h2>
      <table><thead><tr><th>Scenario</th><th>Candidate</th><th>Probability</th></tr></thead><tbody>{scenario_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Wager Gate</h2>
      <table><thead><tr><th>Candidate</th><th>Decision</th><th>Forecast</th><th>Block Reasons</th></tr></thead><tbody>{wager_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Data Quality</h2>
      <table><thead><tr><th>Dataset</th><th>Rows</th><th>Stale</th><th>Missing Sources</th><th>Completeness</th></tr></thead><tbody>{quality_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Polling Layer</h2>
      <p class="subtle">Candidate polling is empty right now; environment polling is compiled from Maryland public poll reports and linked PDFs.</p>
      <table><thead><tr><th>Candidate Polls</th><th>Count</th><th>Weighted</th><th>Latest</th><th>Last Release</th></tr></thead><tbody>{candidate_poll_rows or ''}</tbody></table>
      <div style="height:12px"></div>
      <table><thead><tr><th>Environment Measure</th><th>Polls</th><th>Weighted Gap</th><th>Last Release</th></tr></thead><tbody>{environment_summary_rows}</tbody></table>
      <p class="subtle">Context index: anti-incumbent pressure {float(environment_context.anti_incumbent_pressure):+.3f}, approval gap {float(environment_context.approval_gap):+.2f}, direction gap {float(environment_context.direction_gap):+.2f}, matchup gap {float(environment_context.matchup_gap):+.2f}</p>
    </section>
    <section class="panel">
      <h2>Market Check</h2>
      <p class="subtle">External market data remains a weak check only and does not drive the core forecast.</p>
      <table><thead><tr><th>Candidate</th><th>Forecast</th><th>Market</th><th>Edge</th></tr></thead><tbody>{market_rows}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    (SHOWCASE_DIR / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
