"""Batch evaluator:  python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output

Calls the SAME pipeline (process_run) and SAME chat service as the API. One failing case never aborts the batch.

Input:  {"datasets": {"orders": "orders.csv", ...}  |  "datasets_dir": "dir",
         "cases": [{"id": "c1", "question": "...", "expect": {...}} ,
                   {"id": "c2", "conversation": [{"question": "...", "expect": {...}}, ...]}]}
        A case may carry its own "datasets" to evaluate against a different dataset collection.
Expect keys (all optional): status, intent, value (+tolerance), first_group, row_count, contains, not_contains, filters [[field,value]], planner."""
import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

from .chat import ChatService
from .config import get_settings
from .llm import build_llm
from .pipeline import process_run, safe_name
from .store import Store


def _resolve(spec: dict, base: Path) -> dict:
    ds = spec.get("datasets")
    if ds is None and spec.get("datasets_dir"):
        ds = {p.stem: str(p) for p in sorted((base / spec["datasets_dir"]).glob("*.csv"))}
    return {safe_name(k): (base / v if not Path(v).is_absolute() else Path(v)) for k, v in (ds or {}).items()}


def check_expect(turn: dict, expect: dict) -> list[str]:
    f, ans, plan = [], turn["answer"], turn["plan"] or {}
    if "status" in expect and turn["status"] != expect["status"]:
        f.append(f"status {turn['status']!r} != {expect['status']!r}")
    if "intent" in expect and plan.get("intent") != expect["intent"]:
        f.append(f"intent {plan.get('intent')!r} != {expect['intent']!r}")
    if "planner" in expect and turn["trace"].get("planner") != expect["planner"]:
        f.append(f"planner {turn['trace'].get('planner')!r} != {expect['planner']!r}")
    rows = ans.get("rows") or []
    if "value" in expect:
        names = [m["as"] for m in plan.get("metrics", [])] or [c for c in (ans.get("columns") or [])]
        got = next((rows[0][n] for n in names if rows and isinstance(rows[0].get(n), (int, float))), None)
        if got is None or not math.isclose(got, expect["value"], rel_tol=1e-4, abs_tol=expect.get("tolerance", 0.01)):
            f.append(f"value {got!r} != {expect['value']!r}")
    if "first_group" in expect:
        gb = plan.get("group_by") or []
        got = rows[0].get(gb[0]) if rows and gb else None
        if str(got) != str(expect["first_group"]):
            f.append(f"first group {got!r} != {expect['first_group']!r}")
    if "row_count" in expect and (ans.get("evidence") or {}).get("result_row_count") != expect["row_count"]:
        f.append(f"row_count {(ans.get('evidence') or {}).get('result_row_count')!r} != {expect['row_count']!r}")
    text = (ans.get("text") or "").lower()
    for s in expect.get("contains", []):
        if s.lower() not in text:
            f.append(f"answer missing {s!r}")
    for s in expect.get("not_contains", []):
        if s.lower() in text:
            f.append(f"answer must not contain {s!r}")
    for fld, val in expect.get("filters", []):
        if not any(x["field"] == fld and str(x["value"]).lower() == str(val).lower() for x in plan.get("filters", [])):
            f.append(f"plan missing filter {fld}={val}")
    return f


def run_evaluation(spec: dict, base: Path, artifacts_dir: Path) -> dict:
    s = get_settings()
    llm = build_llm(s)
    shared = _resolve(spec, base)
    groups: dict[str, dict] = {}
    for case in spec.get("cases", []):
        ds = _resolve(case, base) if case.get("datasets") or case.get("datasets_dir") else shared
        key = json.dumps({k: str(v) for k, v in sorted(ds.items())})
        groups.setdefault(key, {"datasets": ds, "cases": []})["cases"].append(case)
    runs, chat_eval, summary = [], [], {"cases": 0, "turns": 0, "passed": 0, "failed": 0, "errors": 0}
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp))
        svc = ChatService(store, s, llm)
        for gi, (key, g) in enumerate(groups.items(), 1):
            run_id, gid = f"eval-g{gi}", f"g{gi}"
            store.create_run(run_id, "evaluator", gid)
            load_errors = {}
            for name, path in g["datasets"].items():
                try:
                    shutil.copy(path, store.run_dir(run_id) / "original" / f"{name}.csv")
                except OSError as e:
                    load_errors[name] = {"code": "file_unreadable", "message": str(e)}
            try:
                report = process_run(store, run_id, s, llm)
            except Exception as e:  # structured error, keep going
                report = {"dataset": {}, "profile_before": {}, "issues": [], "cleaning_plan": [], "profile_after": {}, "validation": {},
                          "artifacts": {}, "run_error": {"code": "pipeline_failed", "message": f"{type(e).__name__}: {e}"}}
            for n, e in load_errors.items():
                report["dataset"][n] = {"status": "failed", "error": e}
            out_dir = artifacts_dir / gid
            (out_dir / "cleaned").mkdir(parents=True, exist_ok=True)
            for p in (store.run_dir(run_id) / "cleaned").glob("*"):
                shutil.copy(p, out_dir / "cleaned" / p.name)
            report["artifacts"] = {n: {**a, "original": a["original"], "cleaned": f"{gid}/{a['cleaned']}"} for n, a in report.get("artifacts", {}).items()}
            for case in g["cases"]:
                turns = case.get("conversation") or [{"question": case.get("question", ""), "expect": case.get("expect", {})}]
                summary["cases"] += 1
                sid = f"{gid}-{case['id']}"
                for ti, t in enumerate(turns, 1):
                    summary["turns"] += 1
                    try:
                        turn = svc.ask(run_id, "evaluator", sid, t["question"])
                    except Exception as e:
                        turn = {"status": "error", "plan": {}, "trace": {}, "answer": {"text": "", "error": {"code": "chat_failed", "message": f"{type(e).__name__}: {e}"}}}
                    turn["trace"].pop("timing_ms", None)
                    failures = check_expect(turn, t.get("expect", {})) if t.get("expect") else []
                    if turn["status"] == "error":
                        summary["errors"] += 1
                    summary["passed" if not failures else "failed"] += 1
                    chat_eval.append({"case_id": case["id"], "turn": ti, "run": gid, "question": t["question"], "plan": turn["plan"],
                                      "answer": turn["answer"], "trace": turn["trace"], "status": turn["status"],
                                      "checks": {"passed": not failures, "failures": failures}})
            runs.append({"run": gid, "datasets": {k: str(v) for k, v in g["datasets"].items()}, **{k: report.get(k) for k in
                         ("dataset", "profile_before", "issues", "cleaning_plan", "profile_after", "validation", "artifacts")}})
            (out_dir / "report.json").write_text(json.dumps(runs[-1], indent=1, default=str))
    first = runs[0] if runs else {k: {} for k in ("dataset", "profile_before", "profile_after", "validation", "artifacts")} | {"issues": [], "cleaning_plan": []}
    return {"dataset": first["dataset"], "profile_before": first["profile_before"], "issues": first["issues"],
            "cleaning_plan": first["cleaning_plan"], "profile_after": first["profile_after"], "validation": first["validation"],
            "chat_evaluation": chat_eval, "artifacts": first["artifacts"], "summary": summary, "runs": runs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.evaluate")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--artifacts-dir", default="./evaluation_output")
    a = ap.parse_args(argv)
    try:
        inp = Path(a.input)
        spec = json.loads(inp.read_text())
        if isinstance(spec, list):
            spec = {"cases": spec}
    except (OSError, ValueError) as e:
        out = {"error": {"code": "bad_input", "message": str(e)}, "chat_evaluation": [], "summary": {}}
        Path(a.output).write_text(json.dumps(out, indent=1))
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    result = run_evaluation(spec, inp.resolve().parent, Path(a.artifacts_dir))
    Path(a.output).write_text(json.dumps(result, indent=1, default=str, sort_keys=False))
    sm = result["summary"]
    print(f"cases={sm['cases']} turns={sm['turns']} passed={sm['passed']} failed={sm['failed']} errors={sm['errors']} -> {a.output}")
    return 0 if sm["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
