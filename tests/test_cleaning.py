import io

import pandas as pd

from app.cleaning import ai_category_fill, clean_dataset, count_rerun_changes, validate_cleaning
from app.cleaning_utils import parse_dates
from app.pipeline import CACHE
from conftest import FILES, ScriptedLLM


def _raw(text):
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def test_no_silent_row_loss_and_schema_stable(env):
    rep = env["store"].report(env["rid"])
    v = rep["validation"]["orders"]
    assert v["ok"] and v["unexplained_row_delta"] == 0
    assert v["rows_in"] == 9 and v["rows_out"] == 8 and v["rows_removed_explained"] == 1   # the exact duplicate, explained
    assert v["columns_dropped"] == [] and v["unexpected_schema_changes"] == []
    assert sorted(v["columns_added"]) == ["order_month", "order_year", "revenue"]


def test_cleaner_is_idempotent_on_its_own_output(env):
    for name, v in env["store"].report(env["rid"])["validation"].items():
        assert v["idempotent"], f"{name} changed on re-clean ({v['rerun_changed_cells']} cells)"
    raw = _raw(FILES["orders"])
    first = clean_dataset("orders", raw)["df"]
    assert count_rerun_changes("orders", first, False) == 0


def test_every_step_has_traceability_fields(env):
    steps = env["store"].report(env["rid"])["cleaning_plan"]
    assert steps and len({s["id"] for s in steps}) == len(steps)
    for s in steps:
        assert {"id", "reason", "fields", "risk", "source", "status", "issue_ids"} <= set(s)


def test_unparseable_values_are_nulled_and_counted_not_dropped():
    raw = _raw("order_id,qty,unit_price\n" + "\n".join(f"O{i},1,10" for i in range(9)) + "\nO9,1,cafe\n")
    res = clean_dataset("orders", raw)
    assert len(res["df"]) == 10                                     # row kept
    assert res["df"]["unit_price"].isna().sum() == 1
    step = next(s for s in res["steps"] if s["action"] == "to_numeric" and s["fields"] == ["unit_price"])
    assert step["evidence"]["parse_failures"] == 1


def test_fully_ambiguous_dates_are_flagged_as_assumption_not_silent():
    s = pd.Series(["04/05/2024", "06/07/2024"])
    _, info = parse_dates(s, dayfirst=False)
    assert info["basis"] == "default" and info["ambiguous_values"] == 2
    steps = clean_dataset("customers", _raw(FILES["customers"]), dayfirst=False)["steps"]
    d = next(s for s in steps if s["action"] == "to_datetime")
    assert d["status"] == "applied_with_assumption" and d["risk"] == "high"


def test_mixed_day_and_month_first_leaves_ambiguous_values_null():
    s = pd.Series(["25/01/2024", "01/25/2024", "03/04/2024"])
    out, info = parse_dates(s, dayfirst=False)
    assert info["basis"] == "conflict" and info["unresolved_ambiguous"] == 1
    assert out.iloc[0] == pd.Timestamp("2024-01-25") and out.iloc[1] == pd.Timestamp("2024-01-25") and pd.isna(out.iloc[2])


def test_category_normalisation_keeps_uncertainty(env):
    p = CACHE.get(env["store"], env["rid"], "products", "clean").set_index("sku")
    assert p.loc["TEE-1", "category_norm"] == p.loc["TEE-2", "category_norm"] == "T-Shirts"
    assert p.loc["JEA-2", "category_norm"] == "Jeans" and p.loc["JEA-2", "category_norm_confidence"] == "alias"
    assert p.loc["JAC-1", "category_norm"] == "Uncategorized" and p.loc["JAC-1", "category_norm_confidence"] == "unresolved"


def test_sku_formats_converge_so_joins_work(env):
    p = CACHE.get(env["store"], env["rid"], "products", "clean")
    assert "TEE-2" in set(p["sku"]) and "tee 2" not in set(p["sku"])


def test_conflicting_duplicate_ids_are_flagged_not_resolved():
    raw = _raw("sku,price\nA-1,10\nA-1,12\n")
    res = clean_dataset("products", raw)
    assert any(i["type"] == "duplicate_ids" and i["severity"] == "high" for i in res["issues"])
    assert len(res["df"]) == 2


def test_injection_cell_is_flagged_and_never_reaches_the_ai_step(tmp_path, monkeypatch):
    from conftest import build
    llm = ScriptedLLM(['{"mappings": []}'])
    store, s, rid = build(tmp_path, llm=llm, monkeypatch=monkeypatch)
    issues = store.report(rid)["issues"]
    assert any(i["type"] == "possible_prompt_injection" for i in issues)
    assert all("reveal the system prompt" not in p for p in llm.prompts)


def test_ai_category_suggestions_are_validated_before_use():
    df = pd.DataFrame({"product_name": ["Blue Denim Slim", "Mystery Thing", "Real Tee"],
                       "category_norm": ["Uncategorized", "Uncategorized", "T-Shirts"],
                       "category_norm_confidence": ["unresolved", "unresolved", "exact"]})
    llm = ScriptedLLM(['{"mappings":[{"product_name":"Blue Denim Slim","category":"T-Shirts"},'
                       '{"product_name":"Mystery Thing","category":"Weapons"},{"product_name":"Not Asked","category":"T-Shirts"}]}'])
    out = ai_category_fill(df, llm)
    assert out["applied"] == 1 and out["rejected"] == 2
    assert df.loc[0, "category_norm_confidence"] == "ai_suggested" and df.loc[1, "category_norm"] == "Uncategorized"


def test_header_only_and_garbage_files_do_not_crash_the_run(tmp_path, monkeypatch):
    from conftest import build
    files = dict(FILES, extra="a,b\n", empty="")
    store, s, rid = build(tmp_path, files=files, monkeypatch=monkeypatch)
    ds = store.report(rid)["dataset"]
    assert ds["extra"]["status"] == "ok_empty" and ds["empty"]["status"] == "failed" and ds["empty"]["error"]["code"] == "empty_file"
    assert ds["orders"]["status"] == "ok"
    assert store.get_run(rid)["status"] == "done" or store.get_run(rid)["status"] == "partial"


def test_case_variant_tie_break_prefers_title_case():
    """A 1-1 tie must not be broken alphabetically: uppercase sorts first and looks sloppy."""
    raw = _raw("store_id,store_name,region\nS1,Alpha,East\nS2,Beta,EAST\nS3,Gamma,West\nS4,Delta,west \n")
    r = clean_dataset("stores", raw)["df"].set_index("store_id")["region"]
    assert r["S1"] == r["S2"] == "East"
    assert r["S3"] == r["S4"] == "West"


def test_category_normalisation_tie_break_prefers_title_case():
    raw = _raw("sku,product_name,category,price,cost\nA-1,One,Frobnitz,10,5\nA-2,Two,FROBNITZ,12,6\n")
    df = clean_dataset("products", raw)["df"]
    assert set(df["category_norm"]) == {"Frobnitz"}
    # confidence is per row: the canonical spelling is "exact", the variant it replaced is "case_variant"
    assert dict(zip(df["category"], df["category_norm_confidence"])) == {"Frobnitz": "exact", "FROBNITZ": "case_variant"}


def test_all_null_column_is_flagged_kept_and_does_not_corrupt_the_schema():
    raw = _raw("order_id,sku,qty,revenue,discount\nO1,A-1,1,10,\nO2,A-2,2,20,\n")
    res = clean_dataset("orders", raw)
    assert any(i["type"] == "all_null_column" and i["severity"] == "medium" and i["columns"] == ["discount"]
               for i in res["issues"])
    assert "discount" in res["df"].columns and len(res["df"]) == 2      # flagged and kept, never silently dropped


def test_already_clean_input_is_left_alone_and_stays_idempotent():
    raw = _raw("order_id,sku,qty,revenue,order_date\nO1,A-1,1,10.5,2025-01-01\nO2,A-2,2,20.0,2025-02-01\n")
    res = clean_dataset("orders", raw)
    v = validate_cleaning(len(raw), list(raw.columns), res["header_map"], res["df"], res["steps"],
                          count_rerun_changes("orders", res["df"], False))
    assert v["ok"] and v["unexplained_row_delta"] == 0 and v["columns_dropped"] == [] and v["idempotent"]
    assert len(res["df"]) == 2 and "revenue" in res["df"].columns
