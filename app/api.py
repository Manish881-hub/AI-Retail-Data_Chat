"""HTTP API.  Auth here is a deliberate stand-in: `X-User-Id` identifies the tenant (plus an optional shared bearer token).
Every run/session lookup is owner-scoped, so another user's run is indistinguishable from a missing one (404)."""
import hashlib
import re
import uuid
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .chat import ChatService
from .config import Settings, get_settings
from .llm import build_llm
from .pipeline import dataset_names, process_run, safe_name
from .store import Store

_USER = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{1,64}$")


def create_app(settings: Optional[Settings] = None, llm=None) -> FastAPI:
    s = settings or get_settings()
    store = Store(s.data_dir)
    model = llm if llm is not None else build_llm(s)
    chat = ChatService(store, s, model)
    app = FastAPI(title="Retail Data & Conversational Intelligence Workbench", version="1.0")

    def user(x_user_id: str = Header(...), authorization: Optional[str] = Header(default=None)) -> str:
        if s.api_token and authorization != f"Bearer {s.api_token}":
            raise HTTPException(401, "invalid or missing bearer token")
        if not _USER.match(x_user_id):
            raise HTTPException(400, "X-User-Id must match [A-Za-z0-9_-]{1,64}")
        return x_user_id

    def own_run(run_id: str, uid: str) -> dict:
        r = store.get_run(run_id, uid) if re.match(r"^[0-9a-f]{12}$", run_id) else None
        if not r:
            raise HTTPException(404, "run not found")
        return r

    @app.get("/health")
    def health():
        return {"ok": True, "llm": "configured" if model else "none (rule planner)"}

    @app.post("/runs", status_code=202)
    async def create_run(background: BackgroundTasks, files: list[UploadFile] = File(...), uid: str = Depends(user)):
        if not files or len(files) > s.max_files:
            raise HTTPException(400, f"upload 1..{s.max_files} CSV files")
        cap = s.max_upload_mb * 1024 * 1024
        blobs: dict[str, bytes] = {}
        for f in files:
            name = safe_name((f.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0])
            if not (f.filename or "").lower().endswith(".csv"):
                raise HTTPException(400, f"{f.filename!r}: only .csv files are accepted")
            if name in blobs:
                raise HTTPException(400, f"duplicate dataset name {name!r}")
            buf, size = bytearray(), 0
            while chunk := await f.read(1024 * 1024):
                size += len(chunk)
                if size > cap:
                    raise HTTPException(413, f"{f.filename!r} exceeds {s.max_upload_mb} MB")
                buf += chunk
            blobs[name] = bytes(buf)
        digest = hashlib.sha256(b"".join(n.encode() + hashlib.sha256(b).digest() for n, b in sorted(blobs.items()))).hexdigest()
        existing = store.find_run(uid, digest)  # idempotent upload: same files from same user => same run
        if existing:
            return {"run_id": existing["run_id"], "status": existing["status"], "reused": True}
        run_id = uuid.uuid4().hex[:12]
        store.create_run(run_id, uid, digest)
        for n, b in blobs.items():
            (store.run_dir(run_id) / "original" / f"{n}.csv").write_bytes(b)
        background.add_task(_process, run_id)
        return {"run_id": run_id, "status": "queued", "reused": False}

    def _process(run_id: str):
        try:
            process_run(store, run_id, s, model)
        except Exception as e:  # last-resort guard; run is marked failed, API stays up
            store.set_status(run_id, "failed", f"{type(e).__name__}: {e}")

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, uid: str = Depends(user)):
        r = own_run(run_id, uid)
        rep = store.report(run_id) or {}
        return {"run_id": run_id, "status": r["status"], "error": r["error"],
                "datasets": {n: {k: v for k, v in i.items() if k in ("status", "rows", "columns", "error", "data_version_clean")}
                             for n, i in rep.get("dataset", {}).items()}}

    @app.get("/runs/{run_id}/report")
    def report(run_id: str, uid: str = Depends(user)):
        own_run(run_id, uid)
        rep = store.report(run_id)
        if rep is None:
            raise HTTPException(409, "run not finished")
        return rep

    @app.get("/runs/{run_id}/datasets/{name}/{which}")
    def download(run_id: str, name: str, which: str, uid: str = Depends(user)):
        own_run(run_id, uid)
        if which not in ("original", "cleaned") or name not in (store.report(run_id) or {}).get("dataset", {}):
            raise HTTPException(404, "artifact not found")
        p = store.run_dir(run_id) / which / f"{name}.csv"
        if not p.exists():
            raise HTTPException(404, "artifact not found")
        return FileResponse(p, media_type="text/csv", filename=f"{name}_{which}.csv")

    @app.post("/runs/{run_id}/chat")
    def ask(run_id: str, body: ChatRequest, uid: str = Depends(user)):
        own_run(run_id, uid)
        return chat.ask(run_id, uid, body.session_id, body.question)

    @app.get("/runs/{run_id}/sessions/{session_id}")
    def session(run_id: str, session_id: str, uid: str = Depends(user)):
        own_run(run_id, uid)
        if not re.match(r"^[A-Za-z0-9_\-]{1,64}$", session_id):
            raise HTTPException(400, "bad session id")
        return {"session_id": session_id, "turns": store.turns(run_id, session_id, uid)}

    return app


app = create_app()
