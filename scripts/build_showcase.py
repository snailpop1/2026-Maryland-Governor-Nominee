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
    leader = scores.iloc[0]

    ranking_rows = "\n".join(
        f"<tr><td>{int(row.rank)}</td><td>{escape(row.candidate)}</td><td>{pct(row.forecast_probability)}</td><td>{row.model_score:.1f}</td></tr>"
        for row in scores.itertuples(index=False)
    )
    wager_rows = "\n".join(
        f"<tr><td>{escape(row.candidate)}</td><td>{escape(row.value_flag)}</td><td>{escape(str(row.block_reasons))}</td></tr>"
        for row in wager.itertuples(index=False)
    )
    scenario_rows = "\n".join(
        f"<tr><td>{escape(row.scenario)}</td><td>{escape(row.candidate)}</td><td>{pct(row.scenario_probability)}</td></tr>"
        for row in scenarios.sort_values(["scenario", "scenario_probability"], ascending=[True, False]).groupby("scenario").head(3).itertuples(index=False)
    )
    quality_rows = "\n".join(
        f"<tr><td>{escape(row.dataset_name)}</td><td>{int(row.row_count)}</td><td>{escape(str(row.stale))}</td><td>{int(row.missing_source_count)}</td></tr>"
        for row in quality.itertuples(index=False)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Maryland GOP Governor Nominee Forecast</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ background: #16213a; color: white; padding: 28px 36px; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 48px; }}
    .headline {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .panel {{ background: white; border: 1px solid #d9dee8; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
    .metric {{ font-size: 30px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e5e8ef; padding: 10px 8px; vertical-align: top; }}
    th {{ color: #465366; font-size: 12px; text-transform: uppercase; }}
    .status {{ display: inline-block; padding: 5px 9px; border-radius: 6px; background: #fff1c2; color: #5a4100; font-weight: 700; }}
    @media (max-width: 760px) {{ .headline {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Maryland GOP Governor Nominee Forecast</h1>
    <p>As-of {escape(str(leader.last_updated))}. Safety-gated election forecast package.</p>
  </header>
  <main>
    <section class="headline">
      <div class="panel"><div>Leader</div><div class="metric">{escape(leader.candidate)}</div></div>
      <div class="panel"><div>Fair Probability</div><div class="metric">{pct(float(leader.forecast_probability))}</div></div>
      <div class="panel"><div>Official Bet Status</div><div class="metric"><span class="status">{escape(str(release.release_status))} / no bet</span></div></div>
    </section>
    <section class="panel">
      <h2>Candidate Ranking</h2>
      <table><thead><tr><th>Rank</th><th>Candidate</th><th>Probability</th><th>Score</th></tr></thead><tbody>{ranking_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Scenario Leaders</h2>
      <table><thead><tr><th>Scenario</th><th>Candidate</th><th>Probability</th></tr></thead><tbody>{scenario_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Wager Gate</h2>
      <table><thead><tr><th>Candidate</th><th>Decision</th><th>Block Reasons</th></tr></thead><tbody>{wager_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Data Quality</h2>
      <table><thead><tr><th>Dataset</th><th>Rows</th><th>Stale</th><th>Missing Sources</th></tr></thead><tbody>{quality_rows}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    (SHOWCASE_DIR / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
