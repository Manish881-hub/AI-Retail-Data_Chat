"""Plan validation (allow-list) and execution. This is the trust boundary: nothing the model says is executed
unless it passes `validate_plan`, and every number in an answer comes from `execute`."""
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

import numpy as np
import pandas as pd

from . import retail
from .config import Settings
from .errors import ExecutionError, PlanRejected
from .guard import is_pii_column
from .models import Plan

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")


def schema_of(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        s = df[c]
        out[c] = ("num" if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                  else "dt" if pd.api.types.is_datetime64_any_dtype(s) else "str")
    return out


def _joined_schema(plan: Plan, schemas: dict) -> dict:
    cols = dict(schemas[plan.dataset])
    for j in plan.joins:
        for c, k in schemas[j.dataset].items():
            cols.setdefault(c, k)
    return cols


def validate_plan(plan: Plan, schemas: dict, datasets: dict, s: Settings) -> None:
    """schemas: {dataset: {col: kind}} for the plan's version. datasets: {name: DataFrame} for join-uniqueness checks."""
    errs = []
    if plan.intent in ("clarify", "unanswerable"):
        if not plan.message:
            errs.append(f"{plan.intent} requires a message")
        if errs:
            raise PlanRejected(errs)
        return
    if plan.dataset not in schemas:
        raise PlanRejected([f"unknown dataset {plan.dataset!r}; available: {sorted(schemas)}"])
    if plan.intent == "retail_score":
        for f in plan.filters:
            if f.field not in ("region", "category_norm", "brand") or f.op != "eq" or isinstance(f.value, list):
                errs.append(f"retail_score supports only eq filters on region/category_norm/brand, got {f.field} {f.op}")
        if plan.limit is not None and not (1 <= plan.limit <= s.max_limit):
            errs.append(f"limit must be 1..{s.max_limit}")
        if errs:
            raise PlanRejected(errs)
        return
    if plan.limit is not None and not (1 <= plan.limit <= s.max_limit):
        errs.append(f"limit must be 1..{s.max_limit}")
    if len(plan.joins) > s.max_joins:
        errs.append(f"too many joins (max {s.max_joins})")
    if len(plan.filters) > s.max_filters:
        errs.append(f"too many filters (max {s.max_filters})")
    if len(plan.group_by) > s.max_group_by:
        errs.append(f"too many group_by fields (max {s.max_group_by})")
    if len(plan.metrics) > s.max_metrics:
        errs.append(f"too many metrics (max {s.max_metrics})")
    seen = {plan.dataset}
    for j in plan.joins:
        if j.dataset not in schemas or j.dataset in seen:
            errs.append(f"join dataset {j.dataset!r} unknown or repeated")
            continue
        seen.add(j.dataset)
        left_cols = _joined_schema(Plan(intent=plan.intent, dataset=plan.dataset, joins=[x for x in plan.joins if x is not j and x.dataset in seen - {j.dataset}]), schemas)
        if len(j.left_on) != len(j.right_on) or not j.left_on or len(j.left_on) > 3:
            errs.append("join keys must be 1-3 matching pairs")
            continue
        for a in j.left_on:
            if a not in left_cols:
                errs.append(f"join left key {a!r} not found")
        for b in j.right_on:
            if b not in schemas[j.dataset]:
                errs.append(f"join right key {b!r} not found in {j.dataset}")
        if not errs:
            right = datasets[j.dataset]
            nn = right.dropna(subset=j.right_on)
            if nn.duplicated(subset=j.right_on).any():
                errs.append(f"unsafe join: {j.dataset}.{'+'.join(j.right_on)} is not unique (would fan out and inflate numbers)")
    cols = _joined_schema(plan, schemas)

    def need(field, kinds=None, allow_star=False, pii_ok=False):
        if allow_star and field == "*":
            return
        if field not in cols:
            errs.append(f"unknown field {field!r}")
        elif kinds and cols[field] not in kinds:
            errs.append(f"field {field!r} is {cols[field]}, expected {'/'.join(kinds)} (use the cleaned version for typed fields)")
        elif not pii_ok and is_pii_column(field):
            errs.append(f"field {field!r} holds personal data and cannot be queried")

    for f in plan.filters:
        need(f.field)
        if f.op in ("gt", "gte", "lt", "lte") and f.field in cols and cols[f.field] == "str":
            errs.append(f"operator {f.op} not valid for text field {f.field!r}")
        vals = f.value if isinstance(f.value, list) else [f.value]
        if f.op == "in" and (not isinstance(f.value, list) or len(vals) > s.max_in_values):
            errs.append(f"'in' needs a list of at most {s.max_in_values} values")
        if f.op != "in" and isinstance(f.value, list):
            errs.append(f"operator {f.op} takes a single value")
        if any(isinstance(v, str) and len(v) > 100 for v in vals):
            errs.append("filter value too long")
        if f.op == "contains" and f.field in cols and cols[f.field] != "str":
            errs.append("contains is only valid for text fields")
    for g in plan.group_by:
        need(g)
    names = []
    for m in plan.metrics:
        if not _IDENT.match(m.as_):
            errs.append(f"invalid metric alias {m.as_!r}")
        names.append(m.as_)
        if m.agg in ("sum", "mean", "median"):
            need(m.field, ("num",))
        elif m.agg in ("min", "max"):
            need(m.field, ("num", "dt"))
        elif m.agg == "aov":
            need(m.field, ("num",))
            if "order_id" not in cols:
                errs.append("aov needs an order_id column")
        else:
            need(m.field, allow_star=(m.agg == "count"))
    if len(set(names)) != len(names):
        errs.append("duplicate metric aliases")
    for f in plan.select:
        need(f)
    for srt in plan.sort:
        if srt.field not in names and srt.field not in plan.group_by and srt.field not in plan.select:
            errs.append(f"sort field {srt.field!r} must be a metric alias, group_by or selected field")
    if plan.intent == "aggregate" and not plan.metrics:
        errs.append("aggregate needs at least one metric")
    if plan.intent == "list" and not plan.select:
        errs.append("list needs select fields (max 10)")
    if len(plan.select) > 10:
        errs.append("select at most 10 fields")
    if plan.intent == "describe" and not plan.select:
        errs.append("describe needs numeric select fields")
    if plan.intent == "describe":
        for f in plan.select:
            if f in cols and cols[f] != "num":
                errs.append(f"describe needs numeric fields; {f!r} is {cols[f]}")
    if plan.intent == "distinct" and len(plan.select) != 1:
        errs.append("distinct needs exactly one select field")
    if errs:
        raise PlanRejected(errs)


def _py(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _frame_to_rows(df: pd.DataFrame, n: int):
    return [{c: _py(v) for c, v in r.items()} for r in df.head(n).to_dict("records")]


def _apply_filter(df: pd.DataFrame, f) -> tuple[pd.DataFrame, str]:
    s = df[f.field]
    v = f.value
    if pd.api.types.is_numeric_dtype(s):
        conv = (lambda x: float(x)) if not isinstance(v, list) else None
        try:
            vv = [float(x) for x in v] if isinstance(v, list) else float(v)
        except (TypeError, ValueError):
            raise ExecutionError("bad_filter_value", f"{f.field}: expected a number, got {v!r}")
        left = s.astype("float64")
    elif pd.api.types.is_datetime64_any_dtype(s):
        try:
            vv = [pd.Timestamp(x) for x in v] if isinstance(v, list) else pd.Timestamp(v)
        except (TypeError, ValueError):
            raise ExecutionError("bad_filter_value", f"{f.field}: expected a date, got {v!r}")
        left = s
    else:
        left = s.astype(object).where(s.notna(), None).map(lambda x: str(x).strip().casefold() if x is not None else None)
        vv = [str(x).strip().casefold() for x in v] if isinstance(v, list) else str(v).strip().casefold()
    op = f.op
    if op == "eq":
        m = left == vv
    elif op == "ne":
        m = (left != vv) & left.notna()
    elif op == "gt":
        m = left > vv
    elif op == "gte":
        m = left >= vv
    elif op == "lt":
        m = left < vv
    elif op == "lte":
        m = left <= vv
    elif op == "in":
        m = left.isin(vv)
    else:  # contains
        m = left.astype(str).str.contains(re.escape(str(vv)), regex=True, na=False)
    return df[m.fillna(False).astype(bool)], f"filter {f.field} {op} {v!r}"


def _do_execute(plan: Plan, datasets: dict, meta: dict, s: Settings) -> dict:
    ops, used = [], set()
    base = datasets[plan.dataset]
    df = base
    rows_total = len(base)
    join_ev = []
    for j in plan.joins:
        right = datasets[j.dataset]
        lk, rk = list(j.left_on), list(j.right_on)
        l, r = df.copy(), right.copy()
        for a, b in zip(lk, rk):  # compare keys as text so int-vs-str ids still match, deterministically
            l[a] = l[a].astype("string").str.strip()
            r[b] = r[b].astype("string").str.strip()
        r = r.rename(columns=dict(zip(rk, lk)))
        keep = lk + [c for c in r.columns if c not in l.columns]
        before = len(l)
        merged = l.merge(r[keep], how=j.how, on=lk, validate="many_to_one", indicator=True)
        unmatched = int((merged["_merge"] == "left_only").sum()) if j.how == "left" else before - len(merged)
        df = merged.drop(columns="_merge")
        if j.how == "left" and len(df) != before:
            raise ExecutionError("join_fanout", f"join with {j.dataset} changed row count {before}->{len(df)}")
        dropped = sorted(c for c in right.columns if c in l.columns and c not in rk)
        join_ev.append({"dataset": j.dataset, "on": dict(zip(lk, rk)), "how": j.how, "left_rows": before, "right_rows": len(right),
                        "rows_after": len(df), "unmatched_left_rows": unmatched, "fanout_check": "many_to_one verified",
                        "right_columns_ignored_due_to_name_clash": dropped})
        ops.append(f"join {j.dataset} on {'+'.join(lk)} ({j.how}, many-to-one verified, {unmatched} unmatched)")
        used.update(lk)
    for f in plan.filters:
        df, o = _apply_filter(df, f)
        ops.append(o)
        used.add(f.field)
    rows_considered = len(df)

    if plan.intent == "aggregate":
        agg_cols = {}
        for m in plan.metrics:
            used.add(m.field)
        if plan.group_by:
            used.update(plan.group_by)
            g = df.groupby(plan.group_by, dropna=True, sort=True)
            res = pd.DataFrame(index=g.size().index)
            for m in plan.metrics:
                res[m.as_] = _agg(g, df, m, plan.group_by)
            res = res.reset_index()
            ops.append("groupby " + ", ".join(plan.group_by))
        else:
            res = pd.DataFrame([{m.as_: _agg(None, df, m, []) for m in plan.metrics}])
        ops += [f"{m.agg}({m.field}) as {m.as_}" for m in plan.metrics]
    elif plan.intent == "list":
        used.update(plan.select)
        res = df[plan.select].copy()
        ops.append("select " + ", ".join(plan.select))
    elif plan.intent == "describe":
        used.update(plan.select)
        d = df[plan.select].astype("float64").describe().T.reset_index().rename(columns={"index": "field", "50%": "median"})
        res = d[["field", "count", "mean", "std", "min", "25%", "median", "75%", "max"]]
        ops.append("describe " + ", ".join(plan.select))
    elif plan.intent == "distinct":
        used.update(plan.select)
        c = plan.select[0]
        vc = df[c].value_counts(dropna=True)
        res = pd.DataFrame({c: vc.index, "rows": vc.values})
        ops.append(f"distinct {c}")
    else:
        raise ExecutionError("unsupported_intent", plan.intent)

    sort = plan.sort or ([] if plan.intent != "distinct" else [])
    if sort:
        by = [x.field for x in sort]
        tie = [c for c in plan.group_by if c not in by]
        res = res.sort_values(by + tie, ascending=[x.dir == "asc" for x in sort] + [True] * len(tie), kind="mergesort", na_position="last")
        ops.append("sort " + ", ".join(f"{x.field} {x.dir}" for x in sort))
    limit = plan.limit
    if limit:
        res = res.head(limit)
        ops.append(f"limit {limit}")
    n = len(res)
    ev = {**meta, "columns_used": sorted(used), "filters": [f.model_dump() for f in plan.filters], "joins": join_ev, "operations": ops,
          "rows_total": rows_total, "rows_considered": rows_considered, "result_row_count": n,
          "truncated": n > s.preview_rows, "result_preview_rows": min(n, s.preview_rows)}
    return {"columns": [str(c) for c in res.columns], "rows": _frame_to_rows(res, s.preview_rows), "evidence": ev}


def _agg(g, df, m, group_by):
    src = g if g is not None else df
    f = m.field
    if m.agg == "aov":
        if g is None:
            n = df["order_id"].nunique()
            return float(df[f].sum() / n) if n else np.nan
        rev, n = g[f].sum(), g["order_id"].nunique()
        return rev / n.replace(0, np.nan)
    if m.agg == "count":
        if f == "*":
            return g.size() if g is not None else len(df)
        return src[f].count()
    return getattr(src[f], m.agg)()


def execute(plan: Plan, datasets: dict, meta: dict, s: Settings) -> dict:
    """Run with a wall-clock budget. Pandas cannot be interrupted; on timeout the caller gets a structured error
    and the worker thread finishes in the background (documented limitation)."""
    if plan.intent == "retail_score":
        return _score(plan, datasets, meta, s)
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_do_execute, plan, datasets, meta, s)
    try:
        return fut.result(timeout=s.query_timeout_s)
    except FutTimeout:
        raise ExecutionError("timeout", f"query exceeded {s.query_timeout_s}s budget")
    except ExecutionError:
        raise
    except Exception as e:
        raise ExecutionError("execution_failed", f"{type(e).__name__}: {e}")
    finally:
        ex.shutdown(wait=False)


def _score(plan: Plan, datasets: dict, meta: dict, s: Settings) -> dict:
    roles = meta["roles"]
    orders, products = datasets[roles["orders"]], datasets[roles["products"]]
    inv = datasets.get(roles.get("inventory")) if roles.get("inventory") else None
    stores = datasets.get(roles.get("stores")) if roles.get("stores") else None
    f = {x.field: x.value for x in plan.filters if x.op == "eq"}
    out = retail.product_scores(orders, products, inv, stores, f)
    n = plan.limit or 10
    rows = out["rows"][:n]
    flat = [{"sku": r["sku"], "product": r["product_name"], "category": r["category"], "priority_score": r["score"],
             "stockout_risk": r["stockout_risk"], **{f"signal_{k}": v for k, v in r["signals"].items()}} for r in rows]
    ev = {**{k: v for k, v in meta.items() if k != "roles"}, "operations": [
        "score = renormalised weighted sum of: " + ", ".join(f"{k} ({v:g})" for k, v in out["weights"].items()),
        f"sales window = {out['window_days']} days", f"filters {f}" if f else "no filters", f"top {n} of {len(out['rows'])} scored SKUs"],
          "columns_used": ["sku", "qty", "order_date", "returned", "stock", "price", "cost"], "filters": [x.model_dump() for x in plan.filters],
          "joins": [], "rows_total": len(orders), "rows_considered": len(orders), "result_row_count": len(rows), "truncated": False,
          "score_factors": [{"sku": r["sku"], "factors": r["factors"], "missing_evidence": r["missing_evidence"]} for r in rows],
          "weights": out["weights"]}
    return {"columns": list(flat[0].keys()) if flat else [], "rows": flat[: s.preview_rows], "evidence": ev}
