# Model Card

## Intended Use

This package estimates the June 23, 2026 Maryland Republican gubernatorial primary as a probabilistic research forecast. It is not a guarantee or financial advice.

## Headline

- As of: `2026-05-20`
- Top candidate: `Dan Cox`
- Top candidate fair probability: `33.8%`
- Release status: `withheld`
- Betting eligible: `False`

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
- Reliability score: `26/100`.
