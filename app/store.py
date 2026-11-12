"""Persistence: SQLite for run state + chat turns, filesystem for artifacts. One directory per run."""
import json
import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "app.db"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content_hash TEXT NOT NULL,
              status TEXT NOT NULL, error TEXT, created_at REAL, updated_at REAL);
            CREATE INDEX IF NOT EXISTS ix_runs_user_hash ON runs(user_id, content_hash);
            CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, user_id TEXT NOT NULL,
              session_id TEXT NOT NULL, idx INTEGER NOT NULL, question TEXT, status TEXT, plan_json TEXT, answer_json TEXT,
              trace_json TEXT, created_at REAL);
            CREATE INDEX IF NOT EXISTS ix_turns ON turns(run_id, session_id, idx);
            """)

    def _conn(self):
        c = sqlite3.connect(self.db, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ---- runs
    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def create_run(self, run_id, user_id, content_hash):
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?)", (run_id, user_id, content_hash, "queued", None, now, now))
        self.run_dir(run_id).joinpath("original").mkdir(parents=True, exist_ok=True)
        self.run_dir(run_id).joinpath("cleaned").mkdir(parents=True, exist_ok=True)

    def find_run(self, user_id, content_hash):
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE user_id=? AND content_hash=? AND status IN ('queued','running','done','partial') "
                          "ORDER BY created_at DESC LIMIT 1", (user_id, content_hash)).fetchone()
        return dict(r) if r else None

    def set_status(self, run_id, status, error=None):
        with self._lock, self._conn() as c:
            c.execute("UPDATE runs SET status=?, error=?, updated_at=? WHERE run_id=?", (status, error, time.time(), run_id))

    def get_run(self, run_id, user_id=None):
        """Owner-scoped: a run of another user is indistinguishable from a missing one."""
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not r or (user_id is not None and r["user_id"] != user_id):
            return None
        return dict(r)

    def report(self, run_id):
        p = self.run_dir(run_id) / "report.json"
        return json.loads(p.read_text()) if p.exists() else None

    # ---- chat turns
    def add_turn(self, run_id, user_id, session_id, question, status, plan, answer, trace):
        with self._lock, self._conn() as c:
            idx = c.execute("SELECT COALESCE(MAX(idx),0)+1 FROM turns WHERE run_id=? AND session_id=?", (run_id, session_id)).fetchone()[0]
            c.execute("INSERT INTO turns(run_id,user_id,session_id,idx,question,status,plan_json,answer_json,trace_json,created_at)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (run_id, user_id, session_id, idx, question, status, json.dumps(plan), json.dumps(answer, default=str),
                       json.dumps(trace, default=str), time.time()))
        return idx

    def turns(self, run_id, session_id, user_id):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM turns WHERE run_id=? AND session_id=? AND user_id=? ORDER BY idx",
                             (run_id, session_id, user_id)).fetchall()
        return [{"index": r["idx"], "question": r["question"], "status": r["status"], "plan": json.loads(r["plan_json"]),
                 "answer": json.loads(r["answer_json"]), "trace": json.loads(r["trace_json"])} for r in rows]
