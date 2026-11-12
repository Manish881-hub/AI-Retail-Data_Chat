"""Dataset profiling: schema, counts, nulls, duplicates, cardinality, distributions."""
import numpy as np
import pandas as pd

from .cleaning_utils import NULL_TOKENS, parse_numeric


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def profile_df(df: pd.DataFrame) -> dict:
    cols = {}
    for c in df.columns:
        s = df[c]
        txt_null = s.astype(str).str.strip().str.lower().isin(NULL_TOKENS) if s.dtype == object or str(s.dtype).startswith("str") else s.isna()
        nulls = int((txt_null | s.isna()).sum())
        nn = s[~(txt_null | s.isna())]
        info = {"dtype": str(s.dtype), "nulls": nulls,
                "null_pct": round(100 * nulls / max(len(s), 1), 2), "distinct": int(nn.nunique())}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if len(nn):
                x = nn.astype(float)
                info.update(min=_py(x.min()), max=_py(x.max()), mean=_py(x.mean()), std=_py(x.std()),
                            p50=_py(x.median()))
        elif pd.api.types.is_datetime64_any_dtype(s):
            if len(nn):
                info.update(min=_py(nn.min()), max=_py(nn.max()))
        else:
            if len(nn):
                parsed = parse_numeric(nn)
                info["numeric_parse_rate"] = round(float(parsed.notna().mean()), 3)
                info["top_values"] = {str(k): int(v) for k, v in nn.astype(str).value_counts().head(5).items()}
        cols[str(c)] = info
    return {"rows": int(len(df)), "columns": int(len(df.columns)),
            "duplicate_rows": int(df.duplicated().sum()) if len(df) else 0,
            "schema": {str(c): str(df[c].dtype) for c in df.columns}, "column_profiles": cols}
