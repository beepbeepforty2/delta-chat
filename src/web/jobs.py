"""One comparison = one Job: uploaded bytes in, artifacts + a live chat
index out.

This deliberately does NOT reuse `cli._run_pipeline`. That function is the
right shape but the wrong interface -- it prints refusals to stderr and
returns the int `1`, which is correct for a shell and useless for a browser
that needs to render *why* a pair was rejected and offer an override. What
it reuses instead is everything below the presentation line:
`_resolve_with_pid`, `check_same_document` and `compute_deltas`, unchanged,
so a delta shown in the browser is the same object the CLI would print and
the eval scorecard would score. There is no second engine here.

Three things the CLI never had to worry about, handled here:

1. **Raster filename collisions.** Both ingest adapters write page images to
   `raster_cache/{pid}_sheet{n}.png`, and the CLI hardcodes pid to "A"/"B"
   (src/cli.py:72). Two browser tabs analysing different pairs would
   overwrite each other's pages -- silently, producing a report whose
   drawings belong to somebody else's job. Fixed by scoping the pid to the
   job id. That is safe rather than a hack: `doc.pid` is never read anywhere
   in the codebase, and the "A"/"B" labels that appear in chat citations
   come from `build_chunks`, which hardcodes them independently
   (src/chat/retrieval.py:94). The alternative -- threading a cache_dir
   parameter through FormatAdapter and all three adapters -- changes a
   public interface to fix a filename.

2. **Relative default paths.** `reports/`, `traces/` and `config/domain.yaml`
   all default relative to the process CWD, which for a server is wherever
   it happened to be launched. Everything here is absolute.

3. **Tracer is documented as not thread-safe** (tracer.py:68). One per job,
   never shared, written into the job's own directory.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.chat.retrieval import BM25Index, build_chunks
from src.cli import _resolve_with_pid, compute_deltas
from src.delta.model import Delta
from src.delta.precheck import PrecheckResult, check_same_document
from src.delta.report import write_report
from src.markup.html_report import render_html_report
from src.markup.payload import build_payload
from src.markup.pdf_annotate import render_pdf_markup
from src.observability.tracer import Tracer

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_ROOT = Path(os.environ.get("WEB_JOB_ROOT", REPO_ROOT / ".web_jobs")).resolve()
# Passed explicitly to BM25Index rather than relying on retrieval.py's
# relative default, which a server launched from another directory would
# silently miss -- a missing aliases file is not an error there, it just
# turns off query expansion, so the failure would show up as subtly worse
# chat answers rather than as a crash.
DOMAIN_ALIASES = REPO_ROOT / "config" / "domain.yaml"

# Bundled pairs so the app can demonstrate itself with no input at all --
# the single most useful thing for someone opening it for the first time.
SAMPLE_PAIRS: dict[str, dict[str, str]] = {
    "real_pair": {
        "label": "Vendor P&ID — text and tag revisions",
        "a": "data/samples/real_pair/a/L0.pdf",
        "b": "data/samples/real_pair/b/L0.pdf",
    },
    "real_pair_valves": {
        "label": "Vendor P&ID — valve and geometry edits",
        "a": "data/samples/real_pair_valves/a/L0.pdf",
        "b": "data/samples/real_pair_valves/b/L0.pdf",
    },
}

# Ordered, human-readable. The UI shows these verbatim, so they are phrased
# for a P&ID engineer rather than describing the module that runs.
STAGES = [
    ("ingest", "Reading both drawings"),
    ("precheck", "Checking they are the same drawing"),
    ("diff", "Comparing revisions"),
    ("report", "Rendering the report"),
    ("index", "Preparing chat"),
]
STAGE_LABELS = dict(STAGES)


@dataclass
class Job:
    job_id: str
    label_a: str
    label_b: str
    dir: Path
    path_a: Path
    path_b: Path
    forced: bool = False
    status: str = "queued"          # queued | running | done | refused | error
    stage: str | None = None
    error: str | None = None
    created_ts: float = field(default_factory=time.time)
    finished_ts: float | None = None
    correlation_id: str | None = None

    # Live objects, kept in memory for the lifetime of the job. doc_a/doc_b
    # are retained not for the report (already written to disk) but because
    # resolving a chat citation back to a bbox needs the element index --
    # see src/web/citations.py.
    doc_a: CanonicalDocument | None = None
    doc_b: CanonicalDocument | None = None
    deltas: list[Delta] | None = None
    index: BM25Index | None = None
    precheck: PrecheckResult | None = None
    payload: dict | None = None

    @property
    def elapsed_ms(self) -> int:
        end = self.finished_ts if self.finished_ts is not None else time.time()
        return int((end - self.created_ts) * 1000)

    def precheck_dict(self) -> dict | None:
        """Structured, so the client can branch on `identity_tier` rather
        than on `reason`. cli.py:89 documents a real bug caused by
        substring-matching that human-readable field."""
        p = self.precheck
        if p is None:
            return None
        return {
            "is_pair": p.is_pair,
            "reason": p.reason,
            "identity_tier": p.identity_tier,
            "drawing_no_a": p.drawing_no_a,
            "drawing_no_b": p.drawing_no_b,
            "equipment_a": p.equipment_a,
            "equipment_b": p.equipment_b,
            # Accepted, but on the weakest evidence available. cmd_run prints
            # this to stderr; a GUI that dropped it would be quietly claiming
            # more confidence than the pipeline has.
            "weak_identity": p.identity_tier in ("tag_overlap", "none"),
        }

    def status_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage or ""),
            "stages": [{"key": k, "label": v} for k, v in STAGES],
            "label_a": self.label_a,
            "label_b": self.label_b,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "forced": self.forced,
            "precheck": self.precheck_dict(),
            "correlation_id": self.correlation_id,
        }


class JobStore:
    """In-process, single-user by design (see the plan: this binds to
    localhost and has no auth). Jobs die with the process; the lock exists
    because uvicorn runs the CPU-bound pipeline in a threadpool, so two
    requests genuinely do touch this dict concurrently."""

    def __init__(self, root: Path = JOB_ROOT):
        self.root = Path(root)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, label_a: str, label_b: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root / job_id
        (job_dir / "reports").mkdir(parents=True, exist_ok=True)
        job = Job(
            job_id=job_id,
            label_a=label_a,
            label_b=label_b,
            dir=job_dir,
            path_a=job_dir / "a.pdf",
            path_b=job_dir / "b.pdf",
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.dir, ignore_errors=True)
        _purge_rasters(job.job_id)
        return True


def _purge_rasters(job_id: str) -> None:
    """The page images live in the shared raster_cache/ (set by an
    import-time constant in both adapters, so it cannot be redirected per
    job) and are namespaced by the job-scoped pid instead. Deleting a job
    has to clean them up by that prefix or they accumulate forever."""
    from src.ingest.pdf_native import RASTER_CACHE_DIR

    cache = Path(RASTER_CACHE_DIR)
    if not cache.is_dir():
        return
    for png in cache.glob(f"{job_id}_*.png"):
        png.unlink(missing_ok=True)


def run_analysis(job: Job) -> None:
    """Runs the full pipeline for one job, recording progress and outcome on
    the Job itself. Never raises: a failure becomes status="error" with the
    message attached, because the caller is a background task whose
    exception nobody would see."""
    tracer = Tracer(trace_dir=str(job.dir / "traces"))
    job.correlation_id = tracer.correlation_id
    job.status = "running"
    try:
        with tracer.span("request", pid_a=str(job.path_a), pid_b=str(job.path_b), mode="web"):
            job.stage = "ingest"
            with tracer.span("ingest"):
                with tracer.span("ingest_a", path=str(job.path_a)) as s:
                    # Job-scoped pid: namespaces this job's page rasters. See
                    # the module docstring for why this is safe.
                    job.doc_a = _resolve_with_pid(f"{job.job_id}_A", str(job.path_a))
                    s.set("n_elements", sum(len(sh.elements) for sh in job.doc_a.sheets))
                with tracer.span("ingest_b", path=str(job.path_b)) as s:
                    job.doc_b = _resolve_with_pid(f"{job.job_id}_B", str(job.path_b))
                    s.set("n_elements", sum(len(sh.elements) for sh in job.doc_b.sheets))

            job.stage = "precheck"
            with tracer.span("precheck") as s:
                job.precheck = check_same_document(job.doc_a, job.doc_b)
                s.set("is_pair", job.precheck.is_pair)
                s.set("reason", job.precheck.reason)
                s.set("forced", job.forced)

            if not job.precheck.is_pair and not job.forced:
                # Not an error -- a deliberate, explained refusal. The client
                # renders the structured reason and may re-run with force.
                job.status = "refused"
                job.stage = None
                return

            job.stage = "diff"
            job.deltas = compute_deltas(job.doc_a, job.doc_b, tracer)

            job.stage = "report"
            with tracer.span("report") as s:
                out_dir = str(job.dir / "reports")
                json_path, md_path = write_report(job.deltas, job.label_a, job.label_b, out_dir)
                s.set("json_path", json_path)
                s.set("md_path", md_path)
            with tracer.span("html_report"):
                render_html_report(job.doc_a, job.doc_b, job.deltas, job.label_a, job.label_b,
                                    str(job.dir / "reports" / "report.html"))
            with tracer.span("markup"):
                render_pdf_markup(job.doc_a, job.doc_b, job.deltas,
                                   str(job.path_a), str(job.path_b), str(job.dir / "reports"))

            job.payload = build_payload(job.doc_a, job.doc_b, job.deltas,
                                         job.label_a, job.label_b, inline_images=False)

            job.stage = "index"
            with tracer.span("build_index") as s:
                chunks = build_chunks(job.doc_a, job.doc_b, job.deltas)
                job.index = BM25Index(chunks, aliases_path=str(DOMAIN_ALIASES))
                s.set("n_chunks", len(chunks))

            job.status = "done"
            job.stage = None
    except Exception as e:  # noqa: BLE001 -- background task, nothing above catches
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.stage = None
    finally:
        job.finished_ts = time.time()
        tracer.finish()


def reset_for_force(job: Job) -> None:
    """Prepare a refused job for a re-run that skips the precheck gate. The
    PrecheckResult is deliberately kept -- the UI must go on showing the
    warning after the override, not pretend the pair was clean."""
    job.forced = True
    job.status = "queued"
    job.stage = None
    job.error = None
    job.finished_ts = None
    job.created_ts = time.time()
