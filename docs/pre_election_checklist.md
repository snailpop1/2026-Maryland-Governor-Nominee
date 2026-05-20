# Pre-Election Refresh Checklist

1. Refresh the SBE candidate list and confirm active ballot status.
2. Refresh MDCRIS finance reports after each filing deadline and update `data/raw/finance.csv`.
3. Search for new credible candidate polls; if none exist, append a no-poll row to `source_checks.csv`.
4. Refresh prediction-market prices directly before any price comparison.
5. Add new debate, endorsement, withdrawal, Trump/GOP leader, and ballot-administration events.
6. Run `python3 scripts/build_dataset.py --as-of 2026-05-20`.
7. Run `python3 scripts/build_showcase.py`.
8. Run `python3 -m pytest`.
9. Review `data/processed/audit_report.md`, `model_card.md`, `release_status.csv`, and `wager_value_table.csv` before using any forecast output.
