# Maryland Republican Governor Nominee Forecast

Reproducible forecast project for the June 23, 2026 Maryland Republican gubernatorial primary.

This project follows the same audit-oriented structure as the Texas GOP Senate predictor: source-backed raw CSVs, a deterministic Python pipeline, generated processed outputs, a model card, an audit report, a source manifest, a static showcase dashboard, and a safety-gated betting layer.

This is election research, not financial advice. The official wager layer defaults to `no_bet` unless data freshness, source quality, liquidity, settlement terms, and scenario durability all pass.

## Current Headline

Run the pipeline for the current generated forecast:

```bash
python3 scripts/build_dataset.py --as-of 2026-05-20
python3 scripts/build_showcase.py
python3 -m pytest
```

Key files:

- `data/processed/model_output.json`
- `data/processed/candidate_forecast.csv`
- `data/processed/wager_value_table.csv`
- `data/processed/audit_report.md`
- `data/processed/model_card.md`
- `notebooks/md_gop_governor_nominee_forecast.ipynb`
- `showcase/index.html`

## Model Summary

The core model uses official ballot status, the 2022 GOP governor primary, county-level historical GOP primary geography, current finance reports, debate participation, prior nomination history, campaign infrastructure, and ballot-administration context.

Prediction markets are excluded from the core model and used only as an external price check.

## Known Limits

- No credible public candidate-level poll was found as of May 20, 2026.
- Finance reporting is incomplete for several minor active candidates.
- The available market snapshot is stale and includes non-active names.
- The forecast is therefore withheld for official betting purposes.
