"""Pure, deterministic parsers shared by profiling and cleaning (counts, parsing, dates)."""
import re
import warnings

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a", "<na>", "nat"}
BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}
BOOL_NAMES = {"returned", "is_returned", "return", "refunded", "active", "is_active"}
_ID_RE = re.compile(r"(^|_)(id|sku|code|zip|pin|phone)$")
_NUM_STRIP = re.compile(r"[\s,$€£₹]")


def is_id_like(col: str) -> bool:
    return bool(_ID_RE.search(col))


def snake(name, i: int) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or f"col_{i}"


def null_mask(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    return s.isna() | t.isin(NULL_TOKENS)


def parse_numeric(s: pd.Series) -> pd.Series:
    """Parse '$1,299.00', ' 12 ', '(5)' -> floats; anything else -> NaN (caller counts failures)."""
    t = s.astype(str).str.strip()
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    t = t.str.replace(_NUM_STRIP, "", regex=True)
    return pd.to_numeric(t, errors="coerce").astype("float64")


def to_int_if_integral(x: pd.Series) -> pd.Series:
    nn = x.dropna()
    if len(nn) and (nn == np.floor(nn)).all() and nn.abs().max() < 2**53:
        return x.astype("Int64")
    return x


def parse_dates(s: pd.Series, dayfirst: bool):
    """Returns (datetime series, info). Never guesses silently:
    - ISO dates parse exactly.
    - 'a/b/yyyy': a>12 => day-first, b>12 => month-first; per-value evidence wins.
    - all-ambiguous columns use the configured default and are reported as an assumption.
    - columns with BOTH conventions: ambiguous values become NaT (unresolved) instead of a guess."""
    t = s.astype(str).str.strip()
    nn = ~null_mask(s)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso = nn & t.str.match(r"^\d{4}-\d{2}-\d{2}([ T].*)?$")
    if iso.any():
        out[iso] = pd.to_datetime(t[iso], errors="coerce", format="ISO8601")
    ext = t.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$").astype("float64")
    a, b, y = ext[0], ext[1], ext[2]
    slash = nn & ~iso & a.notna()
    dmy = int((slash & (a > 12) & (b <= 12)).sum())
    mdy = int((slash & (b > 12) & (a <= 12)).sum())
    amb = slash & (a <= 12) & (b <= 12)
    if dmy and not mdy:
        default, basis = True, "evidence"
    elif mdy and not dmy:
        default, basis = False, "evidence"
    elif dmy and mdy:
        default, basis = None, "conflict"
    else:
        default, basis = dayfirst, "default"
    rowdf = pd.Series(np.nan, index=s.index)
    rowdf[slash & (a > 12)] = 1.0
    rowdf[slash & (b > 12)] = 0.0
    if default is not None:
        rowdf[amb] = 1.0 if default else 0.0
    valid = slash & rowdf.notna()
    if valid.any():
        d_first = rowdf == 1.0
        day = a.where(d_first, b)
        month = b.where(d_first, a)
        out[valid] = pd.to_datetime(pd.DataFrame({"year": y[valid], "month": month[valid], "day": day[valid]}),
                                    errors="coerce")
    other = nn & ~iso & ~slash
    if other.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out[other] = pd.to_datetime(t[other], errors="coerce", format="mixed", dayfirst=dayfirst)
            except (ValueError, TypeError):
                pass
    failures = int((nn & out.isna()).sum())
    info = {"iso_values": int(iso.sum()), "slash_values": int(slash.sum()), "dayfirst_evidence": dmy,
            "monthfirst_evidence": mdy, "ambiguous_values": int(amb.sum()), "basis": basis,
            "assumed_dayfirst": default, "unresolved_ambiguous": int(amb.sum()) if default is None else 0,
            "parse_failures": failures}
    return out, info


def serialize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Stable text form for CSV: ISO dates, nullable ints."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            nn = out[c].dropna()
            fmt = "%Y-%m-%d" if (nn == nn.dt.normalize()).all() else "%Y-%m-%d %H:%M:%S"
            out[c] = out[c].dt.strftime(fmt)
    return out


def kind_of(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "int"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "str"


def read_typed_csv(path, schema: dict) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    for c, k in schema.items():
        if c not in df.columns:
            continue
        if k == "int":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif k == "float":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        elif k == "datetime":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df
