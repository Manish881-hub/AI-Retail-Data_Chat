"""Retail-domain logic that must live in code, not in a model:
category normalisation (with uncertainty preserved) and an auditable product-priority score."""
import difflib
import re

import numpy as np
import pandas as pd

from .cleaning_utils import canonical_spelling

ALIASES = {
    "T-Shirts": {"tshirt", "tshirts", "tee", "tees", "teeshirt", "teeshirts"},
    "Jeans": {"jean", "jeans", "denim", "denims"},
    "Jackets": {"jacket", "jackets", "outerwear", "coat", "coats"},
    "Footwear": {"footwear", "shoe", "shoes", "sneaker", "sneakers"},
    "Accessories": {"accessory", "accessories"},
}
_ALIAS_KEY = {k: label for label, ks in ALIASES.items() for k in ks}
UNRESOLVED_KEYS = {"misc", "miscellaneous", "other", "others", "unknown", "uncategorized", "general", "na", "none"}
UNCATEGORIZED = "Uncategorized"


def _key(v: str) -> str:
    return re.sub(r"[^a-z0-9]", "", v.lower())


def normalize_categories(series: pd.Series, existing_norm: pd.Series | None = None):
    """-> DataFrame[category_norm, category_norm_confidence].
    confidence: exact | case_variant | alias | fuzzy | unresolved | missing.
    'fuzzy' and 'unresolved' are the uncertain ones; callers surface them instead of hiding them."""
    s = series.astype("object")
    keys = s.map(lambda v: _key(str(v)) if isinstance(v, str) and v.strip() else "")
    # canonical spelling per key = most frequent original spelling (ties: alphabetical)
    spell = {}
    for k, grp in s[keys != ""].groupby(keys[keys != ""]):
        vc = grp.str.strip().str.replace(r"\s+", " ", regex=True).value_counts()
        spell[k] = canonical_spelling(vc)
    counts = keys[keys != ""].value_counts().to_dict()
    labels, conf = {}, {}
    fuzzy_pool = [k for k in spell if k not in _ALIAS_KEY and k not in UNRESOLVED_KEYS]
    for k in spell:
        if k in UNRESOLVED_KEYS:
            labels[k], conf[k] = UNCATEGORIZED, "unresolved"
        elif k in _ALIAS_KEY:
            labels[k] = _ALIAS_KEY[k]
            conf[k] = "alias"
        else:
            labels[k], conf[k] = spell[k], "exact"
    if len(fuzzy_pool) <= 500:  # O(n^2) guard
        by_freq = sorted(fuzzy_pool, key=lambda k: (-counts[k], k))
        canon = []
        for k in by_freq:
            m = difflib.get_close_matches(k, canon, n=1, cutoff=0.85)
            if m:
                labels[k], conf[k] = labels[m[0]], "fuzzy"
            else:
                canon.append(k)
    # alias targets: align a spelling that *is* the alias label to "exact"
    for k in spell:
        if conf[k] == "alias" and _key(labels[k]) == k:
            conf[k] = "exact"
    norm, confs = [], []
    raw_str = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    for v, k in zip(raw_str, keys):
        if k == "":
            norm.append(None)
            confs.append("missing")
        else:
            norm.append(labels[k])
            c = conf[k]
            if c == "exact" and v != spell[k]:
                c = "case_variant"
            confs.append(c)
    out = pd.DataFrame({"category_norm": norm, "category_norm_confidence": confs}, index=series.index)
    if existing_norm is not None:  # idempotence: never overwrite a value a previous run/AI step set
        keep = existing_norm.notna() & (existing_norm.astype(str) != UNCATEGORIZED)
        out.loc[keep, "category_norm"] = existing_norm[keep]
    return out


# ---------------------------------------------------------------- product priority score
WEIGHTS = {"urgency": 50.0, "velocity": 20.0, "margin": 20.0, "returns": 10.0}


def product_scores(orders: pd.DataFrame, products: pd.DataFrame, inventory: pd.DataFrame | None,
                   stores: pd.DataFrame | None = None, filters: dict | None = None) -> dict:
    """Reorder-priority score per SKU in [0,100], from explicit signals only:
      urgency  (50) = 1 - days_of_cover/30 clipped to [0,1]   (needs sales velocity + stock)
      velocity (20) = percentile rank of units/day among SKUs
      margin   (20) = (price-cost)/price clipped to [0,1]      (needs price + cost)
      returns  (10) = 1 - return_rate
    A signal that cannot be computed is *dropped and the remaining weights renormalised*;
    the dropped signal is listed in `missing_evidence` for that SKU. Nothing is imputed."""
    filters = filters or {}
    o = orders.copy()
    if stores is not None and "region" in filters and {"store_id", "region"} <= set(stores.columns):
        ids = stores.loc[stores["region"].astype(str).str.casefold() == str(filters["region"]).casefold(), "store_id"]
        o = o[o["store_id"].isin(ids)]
        if inventory is not None:
            inventory = inventory[inventory["store_id"].isin(ids)]
    p = products.copy()
    for col in ("category_norm", "brand"):
        if col in filters and col in p.columns:
            p = p[p[col].astype(str).str.casefold() == str(filters[col]).casefold()]
    o = o[o["sku"].isin(p["sku"])]

    qty_ok = "qty" in o.columns
    days = None
    if "order_date" in o.columns and o["order_date"].notna().any():
        d = o["order_date"].dropna()
        days = max((d.max() - d.min()).days + 1, 1)
    g = o.groupby("sku")
    units = g["qty"].sum() if qty_ok else pd.Series(dtype=float)
    lines = g.size()
    ret = g["returned"].mean() if "returned" in o.columns else pd.Series(dtype=float)
    stock = (inventory.groupby("sku")["stock"].sum().clip(lower=0)
             if inventory is not None and "stock" in inventory.columns else pd.Series(dtype=float))

    rows = []
    vel_series = (units / days) if (days and qty_ok) else pd.Series(dtype=float)
    vel_rank = vel_series.rank(pct=True) if len(vel_series) else vel_series
    for _, r in p.drop_duplicates("sku").iterrows():
        sku = r["sku"]
        comp, missing, facts = {}, [], {}
        v = vel_series.get(sku, np.nan) if len(vel_series) else np.nan
        if sku in lines.index and pd.notna(v):
            facts["units_per_day"] = round(float(v), 4)
            comp["velocity"] = float(vel_rank.get(sku, 0.0))
        else:
            missing.append("velocity (no sales or no usable dates/qty)")
        st = stock.get(sku, np.nan) if len(stock) else np.nan
        if pd.notna(st) and pd.notna(v):
            facts["stock"] = float(st)
            if v > 0:
                cover = st / v
                facts["days_of_cover"] = round(float(cover), 1)
                comp["urgency"] = float(np.clip(1 - cover / 30.0, 0, 1))
            else:
                comp["urgency"] = 0.0
        else:
            missing.append("urgency (needs stock and sales velocity)")
        rr = ret.get(sku, np.nan) if len(ret) else np.nan
        if pd.notna(rr):
            facts["return_rate"] = round(float(rr), 4)
            comp["returns"] = float(1 - rr)
        else:
            missing.append("returns")
        price, cost = pd.to_numeric(r.get("price"), errors="coerce"), pd.to_numeric(r.get("cost"), errors="coerce")
        if pd.notna(price) and pd.notna(cost) and price > 0:
            facts["margin_pct"] = round(float((price - cost) / price), 4)
            comp["margin"] = float(np.clip((price - cost) / price, 0, 1))
        else:
            missing.append("margin (needs price and cost)")
        if not comp:
            continue
        wsum = sum(WEIGHTS[k] for k in comp)
        score = sum(WEIGHTS[k] * comp[k] for k in comp) / wsum * 100
        rows.append({"sku": sku, "product_name": r.get("product_name"), "category": r.get("category_norm", r.get("category")),
                     "score": round(float(score), 2), "stockout_risk": bool(facts.get("stock", 1) == 0 and facts.get("units_per_day", 0) > 0),
                     "factors": {k: round(100 * WEIGHTS[k] * comp[k] / wsum, 2) for k in comp},
                     "signals": facts, "missing_evidence": missing})
    rows.sort(key=lambda x: (-x["score"], str(x["sku"])))
    return {"weights": WEIGHTS, "window_days": days, "rows": rows}
