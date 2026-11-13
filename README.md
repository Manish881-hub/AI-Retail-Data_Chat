# AI Retail Data & Conversational Intelligence Workbench
**Track: AI Engineering** — tool/agent orchestration, validated structured outputs, context management, retries/fallbacks, prompt-injection defences, and an eval harness.

> Principle from the brief: *the model may reason and plan; the application must validate, execute and verify.*

## Quick start (clean clone, no paid service needed)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # LLM_PROVIDER=none works out of the box (deterministic rule planner)
python scripts/generate_sample_data.py    # messy sample CSVs -> data/sample (already committed)
pytest -q                                 # 37 tests

# 1) Batch evaluator (same pipeline the API uses)
python -m app.evaluate --input evaluation/sample_cases.json --output results.json --artifacts-dir ./evaluation_output
#    inspect: results.json, evaluation_output/g1/cleaned/*.csv, evaluation_output/g1/report.json

# 2) API
uvicorn app.api:app --reload              # docs at /docs ; every call needs header  X-User-Id: <you>
curl -H "X-User-Id: me" -F files=@data/sample/orders.csv -F files=@data/sample/products.csv \
     -F files=@data/sample/stores.csv http://localhost:8000/runs
curl -H "X-User-Id: me" -H "Content-Type: application/json" -d '{"question":"revenue by category"}' \
     http://localhost:8000/runs/<run_id>/chat

# 3) Scripted demo used for the walkthrough
python scripts/demo.py
```
Optional LLM: set `LLM_PROVIDER=openai|anthropic` (+ `LLM_MODEL`, `LLM_API_KEY`, and `LLM_BASE_URL` for any OpenAI-compatible endpoint) in `.env`.

## Architecture
```
 CSV upload ──► validate (type/size/name) ──► run record (SQLite) + original artifact
                    │ idempotent: hash(files)+user => same run
                    ▼
  profile_before ► detect issues ► CLEANING PLAN (id, reason, fields, risk, source, status) ► apply ► validate ► profile_after
                    (code)                                                    └─ only AI step: category suggestions for rows code could not resolve
                    ▼
        cleaned artifact (CSV + dtype schema) + report.json
                    ▼
  question ► guard ► PLANNER ► Plan(JSON) ► VALIDATOR (allow-list) ► EXECUTOR (pandas) ► result + evidence ► answer ► persisted turn
              │       LLM (retries w/ error feedback) → rule fallback       │ fan-out-checked joins, limits, timeout
              └ refuse: injection, secrets, code exec, cross-user           └ numbers come ONLY from here
```
Modules: `cleaning*.py` (pipeline), `retail.py` (category normalisation + product score), `planner.py` (rule + LLM planners, deterministic follow-up merge), `executor.py` (validator + executor), `chat.py` (orchestration, evidence, narration check), `evaluate.py` (batch entry), `api.py`, `store.py` (SQLite + files), `guard.py`.

### AI-vs-code boundary
| In code (deterministic, tested) | AI (optional, always validated) |
|---|---|
| counts, joins, filters, aggregates, schema checks, duplicate detection, date/number parsing, scoring arithmetic, validation | translating a question into a **structured plan**; optional phrasing of an explanation; suggesting categories for rows code left "unresolved" |
| follow-up context (`merge_prior`): filters accumulate in code | — |
| every number shown to the user | narration is **rejected** if it contains a number the computation did not produce (`numbers_grounded`) |

With `LLM_PROVIDER=none` everything works via the rule planner; the LLM only adds language coverage.

### Cleaning contract
Every step has a stable id (`orders.S06`), reason, affected fields, risk, source (`code`/`ai`) and final status. Row removal only happens for *exact* duplicates and is counted; unparseable cells become null **and are counted** (rows are never dropped for parse failures). Validation reports `unexplained_row_delta` (must be 0), schema additions/drops, unresolved steps, and an **idempotence check** (re-cleaning the artifact changes 0 cells).
Ambiguous decisions are surfaced, never hidden: all-ambiguous dates (`04/05/2024`) use `DATE_DAYFIRST` and are marked `applied_with_assumption`/risk high; a column mixing both conventions leaves the ambiguous values null rather than guessing; conflicting duplicate IDs are flagged, not auto-resolved; fuzzy/unresolved categories keep `category_norm_confidence`.

### Retail-domain capabilities (≥2 required; all four present)
1. **Product/category normalisation** — alias table + frequency canonical + fuzzy match; SKU canonical format; confidence kept per row.
2. **Cross-dataset relationships** — orders→products/stores/customers/inventory on explicit keys; **many-to-one uniqueness verified before every join** (`unsafe join` rejected), row-count unchanged for left joins, unmatched counts reported.
3. **Explainable priority score** — `score = Σ wᵢ·cᵢ / Σ wᵢ ×100` over urgency (50: days of cover), velocity (20), margin (20), returns (10). Missing signals are *dropped and weights renormalised*, and named in `missing_evidence`; factor points sum to the score.
4. **Sales/inventory analytics** — revenue, AOV (revenue / distinct orders), return rate, units, orders, stock, by category/region/store/brand/segment/month/year.

## Security & privacy decisions
* Filenames, CSV cells, model output and plans are untrusted. Upload: `.csv` only, streamed with a size cap, name re-derived (`safe_name`), content sniffed, per-run directories, owner-scoped lookups (other users get 404).
* Plans must pass an allow-list: known datasets/fields, fixed ops/aggs, `limit ≤ 100`, ≤3 joins, bounded `in` lists, identifier-safe aliases, PII columns (email/phone/name/…) not queryable. No model-generated Python/SQL is ever executed.
* Prompt injection: instruction-like cells are detected in cleaning (`possible_prompt_injection`), dropped from every prompt (`sanitize_value`), and the model only ever sees schema + ≤30 sanitised distinct values of low-cardinality non-PII columns, wrapped as data. Even if it were fooled, the validator bounds what a plan can do.
* **What leaves the system (when an LLM is enabled):** the question, column names/types, up to 30 distinct categorical values per low-cardinality column, up to 3 prior (question, plan) pairs, and — for narration — ≤10 result rows with strings sanitised. No raw rows, no PII columns, no secrets. Secrets live in env vars only (`.env.example`).
* Query budget: `QUERY_TIMEOUT_S` wall-clock, row cap `MAX_ROWS`, result preview ≤20 rows.

## Reliability
LLM errors/timeouts/invalid JSON → up to `LLM_MAX_RETRIES` retries with the validation error fed back and exponential backoff → **rule-planner fallback** (`trace.planner = "rule_fallback"`, `degraded = true`). One dataset failing never fails a run (`partial`); one case failing never aborts the batch evaluator. Evaluator output is deterministic (verified by a test).
Challenge cases covered: ambiguous → clarification; unanswerable → states what is missing; prompt injection in a cell; malicious/expensive plan; LLM outage / invalid JSON; raw-vs-clean (`version` is explicit in every plan and in the evidence, with data hashes).

## Trade-offs
* **Rule planner as the always-on fallback** gives determinism and zero cost but covers fewer phrasings than an LLM; it relies on conventional retail column names (headers are snake-cased; unknown schemas still work through the LLM path).
* **Pandas + CSV/SQLite** instead of Postgres/Parquet: trivially reproducible from a clean clone; not built for multi-GB data.
* **In-process background job** (FastAPI `BackgroundTasks`) instead of a queue: simple and idempotent by content hash; a worker (Celery/RQ) is the upgrade path.
* **Stand-in auth** (`X-User-Id` + optional bearer): isolation logic is real, identity is not.
* Product score weights are explicit and documented, not learned — chosen for auditability over accuracy.

## Known limitations
* Pandas queries cannot be hard-killed on timeout; the caller gets a structured error and the worker thread finishes in the background.
* Number parsing assumes `1,234.56` style (not European `1.234,56`).
* Column-role detection (revenue = qty × price, etc.) is name-based; a dataset with unusual names needs the LLM planner or a mapping.
* `MAX_ROWS` guards memory, but each dataset is loaded fully in memory (LRU of 24 loaded frames).
* Category alias table is small (apparel-oriented); fuzzy matching is flagged as uncertain rather than trusted.
* No UI (track focus is AI engineering; OpenAPI docs at `/docs`).

## Sample data
`data/sample` = `products(sku, product_name, category, brand, price, cost)`, `orders(order_id, customer_id, sku, store_id, order_date, qty, unit_price, returned)`, `stores(store_id, store_name, region)`, `customers(customer_id, segment, signup_date, email)`, `inventory(sku, store_id, stock)` — deliberately messy (case/space/SKU variants, `$1,299.00`, `N/A`, mixed date formats, exact duplicates, a prompt-injection product name, ambiguous signup dates). `evaluation/edge/` adds header-only, latin-1, ragged-row, duplicate-key cases.
