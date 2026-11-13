# 3–5 minute walkthrough script (record your screen while running `python scripts/demo.py`, then `python -m app.evaluate ...`)

**0:00 – Problem + principle (20s).** "Messy retail CSVs in, trustworthy chat out. The model plans; my code validates, executes and verifies."
**0:20 – Cleaning run (50s).** Section 1 of the demo: row counts `3003→3000`, *removed 3, unexplained 0*, idempotent=True. Open `evaluation_output/g1/report.json` → show `cleaning_plan` step with id/reason/risk/source.
**1:10 – One ambiguous decision (30s).** Section 2: customers `signup_date` all ambiguous → HIGH issue, `applied_with_assumption`, not silently guessed. Mention conflicting-duplicate-IDs are flagged, not resolved.
**1:40 – Retail feature (40s).** Section 3: reorder-priority score; show `factors` summing to the score and `missing_evidence`. Mention SKU/category normalisation and fan-out-checked joins.
**2:20 – Multi-turn chat with evidence (60s).** Section 4: revenue by category → "only stores in the West region" → "among those, highest return rate". Point at `filters` carried by code, `operations`, `rows_considered 1037 of 3000`. Then the clarify ("Which store is best?") and the refusal.
**3:20 – Failure / recovery (45s).** Section 5: provider down → 3 attempts → `rule_fallback`, same correct number. Then say: malformed/malicious model plans are rejected by the validator and retried with feedback; hallucinated numbers in narration are discarded.
**4:05 – Reviewer path (40s).** `pytest -q` (37 pass), `python -m app.evaluate ...` → `summary: passed=15 failed=0`, run twice → identical output. Close with the boundary table from the README.
