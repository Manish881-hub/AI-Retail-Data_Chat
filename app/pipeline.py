"""Ingest -> profile -> clean -> validate -> persist. Used by BOTH the API and the batch evaluator."""
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .cleaning import clean_dataset, count_rerun_changes, validate_cleaning
from .cleaning_utils import kind_of, read_typed_csv, serialize_df
from .config import Settings
from .errors import DataError
from .profiling import profile_df
from .store import Store


def sha12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def safe_name(stem: str) -> str:
    n = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower()
    return n[:40] or "dataset"


def read_csv_safe(path: Path, s: Settings):
    """Untrusted CSV in. Returns (all-text DataFrame, meta). Raises DataError with a stable code."""
    data = path.read_bytes()
    if not data.strip():
        raise DataError("empty_file", "file is empty")
    if b"\x00" in data[:4096]:
        raise DataError("not_csv", "binary content detected")
    enc = None
    for e in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            data.decode(e)
            enc = e
            break
        except UnicodeDecodeError:
            continue
    meta = {"encoding": enc, "size_bytes": len(data), "bad_lines_skipped": 0}
    kw = dict(dtype=str, keep_default_na=False, encoding=enc)
    try:
        df = pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        raise DataError("empty_file", "no columns found")
    except pd.errors.ParserError:
        bad = []
        df = pd.read_csv(path, engine="python", on_bad_lines=lambda r: bad.append(r) or None, **kw)
        meta["bad_lines_skipped"] = len(bad)
    if len(df) > s.max_rows:
        raise DataError("too_many_rows", f"{len(df)} rows exceeds MAX_ROWS={s.max_rows}")
    meta["rows_read"] = int(len(df))
    return df, meta


def process_run(store: Store, run_id: str, s: Settings, llm=None) -> dict:
    store.set_status(run_id, "running")
    rd = store.run_dir(run_id)
    report = {"run_id": run_id, "dataset": {}, "profile_before": {}, "issues": [], "cleaning_plan": [],
              "profile_after": {}, "validation": {}, "chat_evaluation": [], "artifacts": {}}
    ok = failed = 0
    for f in sorted((rd / "original").glob("*.csv")):
        name = f.stem
        info = {"status": "failed", "original_artifact": str(f.relative_to(rd)), "data_version_raw": sha12(f.read_bytes())}
        report["dataset"][name] = info
        try:
            raw, meta = read_csv_safe(f, s)
            info.update(meta)
            report["profile_before"][name] = profile_df(raw)
            if len(raw) == 0:
                report["issues"].append({"id": f"{name}.empty_dataset._", "dataset": name, "type": "empty_dataset", "columns": list(raw.columns),
                                         "severity": "high", "count": 0, "detail": "header-only file: no data rows", "examples": [], "step_id": None})
                info.update(status="ok_empty", rows=0, columns=len(raw.columns))
                report["profile_after"][name] = report["profile_before"][name]
                report["validation"][name] = {"ok": True, "rows_in": 0, "rows_out": 0, "note": "header-only; nothing to clean"}
                ok += 1
                continue
            res = clean_dataset(name, raw, s.date_dayfirst, llm)
            if meta["bad_lines_skipped"]:
                res["issues"].append({"id": f"{name}.malformed_rows._", "dataset": name, "type": "malformed_rows", "columns": [],
                                      "severity": "high", "count": meta["bad_lines_skipped"], "examples": [], "step_id": None,
                                      "detail": "rows with the wrong number of fields were skipped by the parser and are NOT in the cleaned artifact"})
            clean = res["df"]
            rerun = count_rerun_changes(name, clean, s.date_dayfirst)
            val = validate_cleaning(len(raw), list(raw.columns), res["header_map"], clean, res["steps"], rerun)
            out = rd / "cleaned" / f"{name}.csv"
            serialize_df(clean).to_csv(out, index=False)
            schema = {c: kind_of(clean[c]) for c in clean.columns}
            (rd / "cleaned" / f"{name}.schema.json").write_text(json.dumps(schema))
            info.update(status="ok" if val["ok"] else "ok_with_issues", rows=int(len(clean)), columns=int(len(clean.columns)),
                        cleaned_artifact=str(out.relative_to(rd)), data_version_clean=sha12(out.read_bytes()))
            report["issues"] += res["issues"]
            report["cleaning_plan"] += res["steps"]
            report["profile_after"][name] = profile_df(clean)
            report["validation"][name] = val
            report["artifacts"][name] = {"original": info["original_artifact"], "cleaned": info["cleaned_artifact"]}
            ok += 1
        except DataError as e:
            info["error"] = e.to_dict()
            failed += 1
        except Exception as e:  # never let one dataset kill the run
            info["error"] = {"code": "internal_error", "message": f"{type(e).__name__}: {e}"}
            failed += 1
    (rd / "report.json").write_text(json.dumps(report, default=str, indent=1))
    status = "done" if failed == 0 and ok else ("partial" if ok else "failed")
    store.set_status(run_id, status, None if status != "failed" else "no dataset could be processed")
    return report


class DatasetCache:
    """Small LRU of loaded datasets, keyed by artifact version, so memory stays bounded."""

    def __init__(self, max_items: int = 24):
        self.max, self._d = max_items, OrderedDict()

    def get(self, store: Store, run_id: str, name: str, version: str) -> pd.DataFrame:
        rd = store.run_dir(run_id)
        if version == "clean":
            p = rd / "cleaned" / f"{name}.csv"
            key = (run_id, name, version, p.stat().st_mtime_ns)
        else:
            p = rd / "original" / f"{name}.csv"
            key = (run_id, name, version, p.stat().st_mtime_ns)
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        if version == "clean":
            df = read_typed_csv(p, json.loads((rd / "cleaned" / f"{name}.schema.json").read_text()))
        else:
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
        self._d[key] = df
        while len(self._d) > self.max:
            self._d.popitem(last=False)
        return df


CACHE = DatasetCache()


def dataset_names(store: Store, run_id: str) -> list[str]:
    rep = store.report(run_id) or {}
    return [n for n, i in rep.get("dataset", {}).items() if i.get("status") in ("ok", "ok_with_issues")]
