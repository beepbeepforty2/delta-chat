"""FastAPI surface over the delta pipeline.

    uv run uvicorn src.web.app:app          (or: make web)

Every route here is plumbing. Nothing in this module decides what a delta
is, where it sits, or how severe it is -- that all comes from `jobs.py`
calling the same `compute_deltas` the CLI and the eval scorecard call, and
from `payload.build_payload`, which also produces the JSON embedded in the
downloadable report.html. If the browser and `make run` ever disagree about
a change, the bug is upstream of here.

Binds to localhost with no authentication, by deliberate choice (see the
plan): this is a tool an engineer runs on their own machine against their
own drawings, not a service. The upload guards below exist to catch honest
mistakes -- a .dwg dropped in the wrong slot, a 2 GB scan -- not attackers.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chat.chat import answer as chat_answer
from src.markup.payload import KIND_LABELS, kind_color_map
from src.web.citations import CitationResolver
from src.web.jobs import (
    REPO_ROOT,
    SAMPLE_PAIRS,
    Job,
    JobStore,
    reset_for_force,
    run_analysis,
)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(os.environ.get("WEB_MAX_UPLOAD_MB", "50")) * 1024 * 1024
# The pipeline is CPU-bound (align's Hungarian assignment dominates), so
# more workers than this just thrashes a laptop. Two lets a second tab start
# while the first is running, which is the only concurrency a local tool
# realistically sees.
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("WEB_WORKERS", "2")),
                                thread_name_prefix="delta-job")

# Whitelist, not a path join: the key IS the whole permitted vocabulary, so
# a traversal attempt resolves to nothing rather than to a file.
ARTIFACTS: dict[str, tuple[str, str]] = {
    "report.json": ("reports/delta_report.json", "application/json"),
    "report.md": ("reports/delta_report.md", "text/markdown"),
    "report.html": ("reports/report.html", "text/html"),
    "markup_a.pdf": ("reports/markup_a.pdf", "application/pdf"),
    "markup_b.pdf": ("reports/markup_b.pdf", "application/pdf"),
}

app = FastAPI(title="delta-chat", docs_url="/api/docs", redoc_url=None)
app.state.store = JobStore()
# Tests set this to a fake so the suite never makes a network call. None
# means `answer()` uses its own default client -- deliberately not
# reimplemented here, because a local copy would have to duplicate the
# token/cost accounting in chat.py and would drift from it.
app.state.call_llm = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _store() -> JobStore:
    return app.state.store


def _job_or_404(job_id: str) -> Job:
    job = _store().get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return job


def _ready_or_409(job_id: str) -> Job:
    """Guards every route that needs results. A 409 with the current status
    lets the client re-poll instead of showing a hard error for a job that
    is merely still running."""
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}, not done")
    return job


def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Streams to disk with a size ceiling. Reading UploadFile whole would
    put an arbitrarily large body in memory before we could reject it."""
    total = 0
    head = b""
    with open(dest, "wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            if not head:
                head = chunk[:5]
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"{upload.filename} exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit")
            f.write(chunk)
    if not head.startswith(b"%PDF"):
        dest.unlink(missing_ok=True)
        raise HTTPException(
            415, f"{upload.filename} is not a PDF. DWG is not supported yet -- "
                 "export to PDF, or see src/ingest/dwg.py.")


def _start(job: Job) -> None:
    _EXECUTOR.submit(run_analysis, job)


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    """`llm_configured` drives whether the Ask tab offers a question box or
    explains how to enable it -- better than letting the first question fail
    with a RuntimeError from deep in llm.py."""
    from src.chat.llm import get_model

    return {
        "ok": True,
        "llm_configured": bool(os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("LLM_API_KEY")),
        "model": get_model(),
        "samples": [{"key": k, "label": v["label"]} for k, v in SAMPLE_PAIRS.items()
                     if (REPO_ROOT / v["a"]).exists() and (REPO_ROOT / v["b"]).exists()],
        "kind_color": kind_color_map(),
        "kind_label": KIND_LABELS,
    }


@app.post("/api/jobs", status_code=202)
def create_job(a: UploadFile, b: UploadFile) -> dict:
    job = _store().create(label_a=a.filename or "revision A",
                           label_b=b.filename or "revision B")
    try:
        _save_upload(a, job.path_a)
        _save_upload(b, job.path_b)
    except HTTPException:
        _store().delete(job.job_id)
        raise
    _start(job)
    return job.status_dict()


@app.post("/api/jobs/sample/{name}", status_code=202)
def create_sample_job(name: str) -> dict:
    """Analyse a bundled vendor pair. The app can then demonstrate itself
    with no input, which matters for a first-time user who has not yet got
    two revisions of the same drawing to hand."""
    spec = SAMPLE_PAIRS.get(name)
    if spec is None:
        raise HTTPException(404, f"no such sample: {name}")
    src_a, src_b = REPO_ROOT / spec["a"], REPO_ROOT / spec["b"]
    if not src_a.exists() or not src_b.exists():
        raise HTTPException(404, f"sample {name} is not present in this checkout")

    job = _store().create(label_a=f"{name} / A", label_b=f"{name} / B")
    job.path_a.write_bytes(src_a.read_bytes())
    job.path_b.write_bytes(src_b.read_bytes())
    _start(job)
    return job.status_dict()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job_or_404(job_id).status_dict()


@app.post("/api/jobs/{job_id}/force", status_code=202)
def force_job(job_id: str) -> dict:
    """Re-run a refused pair with the precheck gate bypassed. The heuristic
    is good but not omniscient -- a drawing with no title block, or renamed
    between revisions, is a real case an engineer can recognise and the
    tool cannot. The PrecheckResult is kept, so the warning stays visible
    for the whole session rather than the override erasing the doubt."""
    job = _job_or_404(job_id)
    if job.status not in ("refused", "error"):
        raise HTTPException(409, f"job is {job.status}; force only applies to a refused job")
    reset_for_force(job)
    _start(job)
    return job.status_dict()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not _store().delete(job_id):
        raise HTTPException(404, f"no such job: {job_id}")
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/payload")
def job_payload(job_id: str) -> dict:
    job = _ready_or_409(job_id)
    return {
        **job.payload,
        "job_id": job.job_id,
        "precheck": job.precheck_dict(),
        "kind_color": kind_color_map(),
        "kind_label": KIND_LABELS,
        "correlation_id": job.correlation_id,
        "elapsed_ms": job.elapsed_ms,
    }


@app.get("/api/jobs/{job_id}/pdf/{side}")
def job_pdf(job_id: str, side: str) -> FileResponse:
    """The original uploaded PDF, for pdf.js to render client-side. Serving
    the source rather than a server-rendered raster is what makes zoom
    vector-crisp -- and delta boxes still overlay as plain percentages,
    because bboxes are normalized to the page (see payload.py)."""
    job = _job_or_404(job_id)
    if side not in ("a", "b"):
        raise HTTPException(404, "side must be 'a' or 'b'")
    path = job.path_a if side == "a" else job.path_b
    if not path.exists():
        raise HTTPException(404, "pdf not found")
    return FileResponse(path, media_type="application/pdf",
                         headers={"Content-Disposition": "inline"})


@app.get("/api/jobs/{job_id}/download/{artifact}")
def job_download(job_id: str, artifact: str) -> FileResponse:
    job = _ready_or_409(job_id)
    entry = ARTIFACTS.get(artifact)
    if entry is None:
        raise HTTPException(404, f"unknown artifact: {artifact}")
    rel, media_type = entry
    path = job.dir / rel
    if not path.exists():
        raise HTTPException(404, f"{artifact} was not produced for this job")
    return FileResponse(path, media_type=media_type, filename=artifact)


class ChatRequest(BaseModel):
    question: str


@app.post("/api/jobs/{job_id}/chat")
def job_chat(job_id: str, req: ChatRequest) -> dict:
    job = _ready_or_409(job_id)
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "question is empty")

    try:
        result = chat_answer(question, job.index, call_llm=app.state.call_llm)
    except RuntimeError as e:
        # llm.py raises this when no credential is configured. A 503 with the
        # real message is more use than a 500 stack trace.
        raise HTTPException(503, str(e)) from e

    resolver = CitationResolver(job.doc_a, job.doc_b, job.payload["deltas"])
    return {
        "question": result.question,
        # On every refusal path chat.py still returns the raw model output in
        # `text` (chat.py:145-156). The client renders `reason` when refused;
        # `text` is passed through only so the trace is inspectable.
        "text": result.text,
        "refused": result.refused,
        "reason": result.reason,
        "citations": resolver.resolve_all(result.citations),
        "retrieved_ids": result.retrieved_ids,
    }


# --------------------------------------------------------------------------
# static frontend -- mounted last so /api/* always wins
# --------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
