# Maryland Republican Governor Nominee Forecast

Reproducible forecast project for the June 23, 2026 Maryland Republican gubernatorial primary.

This project follows an audit-oriented structure: source-backed raw CSVs, a deterministic Python pipeline, generated processed outputs, a model card, an audit report, a source manifest, a static showcase dashboard, and a conservative betting gate.

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
- `data/processed/market_comparison.csv`
- `data/processed/audit_report.md`
- `data/processed/model_card.md`
- `notebooks/md_gop_governor_nominee_forecast.ipynb`
- `showcase/index.html`

## Model Summary

The forecast is now a hybrid probabilistic model built from:

- official ballot status and filing age
- 2022 GOP governor primary statewide and county history
- time-decayed campaign finance signals
- polling, including separate candidate-poll and Maryland environment-poll layers
- debate, endorsement, and campaign event momentum
- campaign infrastructure proxies like website and social presence
- county-level geography fit
- external market prices as a weak comparison signal only
- Monte Carlo uncertainty intervals and scenario simulations

Prediction markets are excluded from the core model and used only as an external price check.

## Known Limits

- No credible public candidate-level poll was found as of May 20, 2026.
- Maryland environment polling is included from Gonzales Research and UMBC Institute of Politics coverage.
- Finance reporting is incomplete for several minor active candidates.
- The available market snapshot is stale and includes non-active names.
- The forecast remains withheld for official betting purposes.
