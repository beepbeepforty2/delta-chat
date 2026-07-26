"""End-to-end tests for the web API, against the real pipeline on real
sample PDFs. Nothing here mocks compute_deltas -- the whole point of the
web layer is that it shows what the CLI shows, and a test that stubbed the
engine could not catch it drifting.

The only injected fake is the LLM (`app.state.call_llm`), because chat is
the one path that would otherwise make a network call.
"""
import json
import pathlib
import re
import time

import pytest
from fastapi.testclient import TestClient

from src.web import app as app_module
from src.web.jobs import JobStore

REPO_ROOT = pathlib.Path(__file__).parent.parent
SAMPLE_A = REPO_ROOT / "data" / "samples" / "real_pair" / "a" / "L0.pdf"
SAMPLE_B = REPO_ROOT / "data" / "samples" / "real_pair" / "b" / "L0.pdf"
VALVES_A = REPO_ROOT / "data" / "samples" / "real_pair_valves" / "a" / "L0.pdf"
VALVES_B = REPO_ROOT / "data" / "samples" / "real_pair_valves" / "b" / "L0.pdf"
# Two genuinely different drawings (different drawing numbers), so the
# precheck refusal path is exercised for real rather than conditionally
# skipped -- it is one of the most user-visible behaviours in the UI.
NOT_A_PAIR = REPO_ROOT / "eval" / "datasets" / "v0" / "pairs" / "not_a_pair_903"
NOT_A_PAIR_A = NOT_A_PAIR / "a" / "L0.pdf"
NOT_A_PAIR_B = NOT_A_PAIR / "b" / "L0.pdf"

pytestmark = pytest.mark.skipif(
    not SAMPLE_A.exists(), reason="data/samples/real_pair is not present in this checkout")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A fresh JobStore rooted in tmp_path, so a test run never writes into
    the developer's real .web_jobs/ and jobs cannot leak between tests."""
    app_module.app.state.store = JobStore(root=tmp_path / "jobs")
    app_module.app.state.call_llm = None
    with TestClient(app_module.app) as c:
        yield c


def _wait(client, job_id, want=("done", "refused", "error"), timeout=120.0):
    """Poll exactly the way the browser does. Jobs run on a real background
    executor, not inline, so this is also what proves the polling contract
    works at all."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in want:
            return body
        time.sleep(0.05)
    pytest.fail(f"job {job_id} still {body['status']} after {timeout}s")


def _upload(client, path_a, path_b):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        return client.post("/api/jobs", files={"a": ("a.pdf", fa, "application/pdf"),
                                                "b": ("b.pdf", fb, "application/pdf")})


@pytest.fixture
def done_job(client):
    r = _upload(client, SAMPLE_A, SAMPLE_B)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "done", body
    return job_id


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------

def test_health_reports_llm_state_and_the_shared_kind_vocabulary(client):
    body = client.get("/api/health").json()

    assert body["ok"] is True
    assert isinstance(body["llm_configured"], bool)
    # The UI must not hardcode colours; they come from overlay.COLORS.
    assert body["kind_color"]["add"] == "#228b22"
    assert body["kind_label"]["unclassified_visual_change"] == "Unclassified visual change"


def test_health_lists_only_samples_actually_present(client):
    keys = {s["key"] for s in client.get("/api/health").json()["samples"]}
    assert "real_pair" in keys


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------

def test_upload_then_poll_then_payload(client, done_job):
    payload = client.get(f"/api/jobs/{done_job}/payload").json()

    assert payload["deltas"], "the sample pair has known changes"
    assert payload["summary"]["n_primary"] > 0
    assert payload["sheets"][0]["has_a"] and payload["sheets"][0]["has_b"]
    # inline_images=False: no base64 on the wire.
    assert "img_a" not in payload["sheets"][0]


def test_web_payload_agrees_with_the_cli_pipeline(client, done_job, tmp_path):
    """The load-bearing test. If the browser and `make run` ever disagree
    about how many changes there are, this fails."""
    from src.cli import _resolve_with_pid, compute_deltas
    from src.observability.tracer import Tracer

    payload = client.get(f"/api/jobs/{done_job}/payload").json()

    doc_a = _resolve_with_pid("cli_A", str(SAMPLE_A))
    doc_b = _resolve_with_pid("cli_B", str(SAMPLE_B))
    cli_deltas = compute_deltas(doc_a, doc_b, Tracer(trace_dir=str(tmp_path / "traces")))

    assert len(payload["deltas"]) == len(cli_deltas)
    assert [d["did"] for d in payload["deltas"]] == [d.did for d in cli_deltas]
    assert [d["severity"] for d in payload["deltas"]] == [d.severity for d in cli_deltas]


def test_every_box_is_normalized_or_explicitly_absent(client, done_job):
    """The frontend lays boxes out as CSS percentages with no clamping, so
    an out-of-range coordinate would silently draw off-canvas."""
    for rec in client.get(f"/api/jobs/{done_job}/payload").json()["deltas"]:
        for key in ("box_a", "box_b"):
            box = rec[key]
            if box is None:
                continue
            assert len(box) == 4
            assert all(0.0 <= v <= 1.0 for v in box), f"{rec['did']} {key}={box}"


def test_status_exposes_stage_progress_while_running(client):
    r = _upload(client, SAMPLE_A, SAMPLE_B)
    body = r.json()

    assert body["status"] in ("queued", "running")
    # The UI renders these labels verbatim, so they must be present up front
    # rather than appearing only once a stage is reached.
    assert [s["key"] for s in body["stages"]] == ["ingest", "precheck", "diff", "report", "index"]
    assert body["stages"][0]["label"] == "Reading both drawings"
    _wait(client, body["job_id"])


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("artifact,media_type", [
    ("report.json", "application/json"),
    ("report.md", "text/markdown"),
    ("report.html", "text/html"),
    ("markup_a.pdf", "application/pdf"),
    ("markup_b.pdf", "application/pdf"),
])
def test_every_artifact_downloads(client, done_job, artifact, media_type):
    r = client.get(f"/api/jobs/{done_job}/download/{artifact}")

    assert r.status_code == 200
    assert r.headers["content-type"].startswith(media_type)
    assert len(r.content) > 0


def test_downloaded_json_matches_the_served_payload(client, done_job):
    downloaded = json.loads(client.get(f"/api/jobs/{done_job}/download/report.json").content)
    payload = client.get(f"/api/jobs/{done_job}/payload").json()

    assert len(downloaded["deltas"]) == len(payload["deltas"])
    assert [d["did"] for d in downloaded["deltas"]] == [d["did"] for d in payload["deltas"]]


def test_downloaded_markup_pdfs_are_real_pdfs(client, done_job):
    r = client.get(f"/api/jobs/{done_job}/download/markup_a.pdf")
    assert r.content.startswith(b"%PDF")


def test_pdf_endpoint_serves_the_original_for_pdfjs(client, done_job):
    r = client.get(f"/api/jobs/{done_job}/pdf/a")

    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content == SAMPLE_A.read_bytes()


def test_unknown_artifact_is_404_not_a_path_traversal(client, done_job):
    assert client.get(f"/api/jobs/{done_job}/download/report.json/../../a.pdf").status_code == 404
    assert client.get(f"/api/jobs/{done_job}/download/a.pdf").status_code == 404


def test_unknown_pdf_side_is_404(client, done_job):
    assert client.get(f"/api/jobs/{done_job}/pdf/c").status_code == 404


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------

def test_chat_returns_resolved_citations(client, done_job):
    payload = client.get(f"/api/jobs/{done_job}/payload").json()
    did = payload["deltas"][0]["did"]
    sheet = payload["deltas"][0]["sheet"]

    app_module.app.state.call_llm = lambda system, user: (
        f"That change is recorded as [delta:{sheet}:A-1:{did}].")

    r = client.post(f"/api/jobs/{done_job}/chat", json={"question": "what changed?"})
    body = r.json()

    assert r.status_code == 200
    assert body["refused"] is False, body
    (cit,) = body["citations"]
    assert cit["id"] == did
    assert cit["raw"] == f"[delta:{sheet}:A-1:{did}]"
    # Resolved to somewhere on the drawing -- this is what a chip clicks to.
    assert cit["resolved"] is not None
    assert cit["resolved"]["did"] == did


def test_chat_marks_an_uncited_answer_as_refused(client, done_job):
    """chat.py deterministically refuses an answer with no citations. The
    API must pass that through rather than presenting ungrounded prose as
    an answer."""
    app_module.app.state.call_llm = lambda system, user: "Lots of things changed, trust me."

    body = client.post(f"/api/jobs/{done_job}/chat", json={"question": "what changed?"}).json()

    assert body["refused"] is True
    assert body["reason"]


def test_an_element_citation_resolves_to_one_pane_and_carries_no_delta(client, done_job):
    """A citation to a raw element ([A:1:F-7:el_…]) is navigable but is not a
    finding: it resolves to a box on ONE side and has no `did`. The frontend
    keys clickability on `resolved`, not on `did` -- keying it on `did` made
    every element citation render as a dead chip despite having a location."""
    seen = {}

    def echo_first_a_citation(system, user):
        # The prompt embeds each retrieved chunk behind its citation tag, so
        # this picks a genuinely retrievable id rather than inventing one.
        m = re.search(r"\[A:[^\]]+\]", user)
        assert m, "no revision-A chunk was retrieved for this question"
        seen["raw"] = m.group(0)
        return f"Revision A records this as {m.group(0)}."

    app_module.app.state.call_llm = echo_first_a_citation
    body = client.post(f"/api/jobs/{done_job}/chat",
                        json={"question": "what does the drawing say?"}).json()

    assert body["refused"] is False, body
    (cit,) = body["citations"]
    assert cit["raw"] == seen["raw"]
    assert cit["source"] == "A"
    r = cit["resolved"]
    assert r is not None
    assert "did" not in r                       # not a delta, so not selectable
    assert r["box_a"] is not None               # ...but still locatable
    assert r["box_b"] is None                   # only on the side it came from
    assert isinstance(r["sheet"], int)


def test_a_hallucinated_citation_never_reaches_the_client_as_a_chip(client, done_job):
    """Defence in depth against a chip that jumps to the wrong valve.
    chat.py:161 validates cited ids against what was actually retrieved and
    returns only `result.valid`, so an invented id is dropped here -- and
    src/web/citations.py independently resolves an unknown id to None if one
    ever did get through (test_web_citations.py covers that leg)."""
    app_module.app.state.call_llm = lambda system, user: "See [delta:1:A-1:delta9999]."

    body = client.post(f"/api/jobs/{done_job}/chat", json={"question": "what changed?"}).json()

    assert body["refused"] is True
    assert "delta9999" in body["reason"]
    assert body["citations"] == []


def test_empty_question_is_rejected(client, done_job):
    assert client.post(f"/api/jobs/{done_job}/chat", json={"question": "   "}).status_code == 422


def test_chat_before_the_job_is_done_is_409(client):
    job_id = _upload(client, SAMPLE_A, SAMPLE_B).json()["job_id"]
    # Racy by nature; only assert when we genuinely caught it mid-flight.
    r = client.post(f"/api/jobs/{job_id}/chat", json={"question": "what changed?"})
    assert r.status_code in (200, 409)
    _wait(client, job_id)


# --------------------------------------------------------------------------
# refusals and overrides
# --------------------------------------------------------------------------

@pytest.mark.skipif(not NOT_A_PAIR_A.exists(), reason="run `make dataset` for the seeded pairs")
def test_mismatched_drawings_are_refused_with_a_structured_reason(client):
    """A dead end with no explanation is the worst possible outcome for a
    non-developer, so the refusal has to carry the evidence the UI needs to
    say *which* two drawings these are."""
    job_id = _upload(client, NOT_A_PAIR_A, NOT_A_PAIR_B).json()["job_id"]

    body = _wait(client, job_id)

    assert body["status"] == "refused"
    pre = body["precheck"]
    assert pre["is_pair"] is False
    assert pre["identity_tier"] == "drawno"
    assert pre["drawing_no_a"] and pre["drawing_no_b"]
    assert pre["drawing_no_a"] != pre["drawing_no_b"]
    assert pre["reason"]
    # Refused is not an error: no exception happened, so no error message.
    assert body["error"] is None


@pytest.mark.skipif(not NOT_A_PAIR_A.exists(), reason="run `make dataset` for the seeded pairs")
def test_a_refused_job_serves_no_results(client):
    job_id = _upload(client, NOT_A_PAIR_A, NOT_A_PAIR_B).json()["job_id"]
    _wait(client, job_id)

    assert client.get(f"/api/jobs/{job_id}/payload").status_code == 409
    assert client.get(f"/api/jobs/{job_id}/download/report.json").status_code == 409
    assert client.post(f"/api/jobs/{job_id}/chat", json={"question": "x"}).status_code == 409


@pytest.mark.skipif(not NOT_A_PAIR_A.exists(), reason="run `make dataset` for the seeded pairs")
def test_force_overrides_a_refusal_but_keeps_the_warning(client):
    """The precheck heuristic is good, not omniscient -- a drawing renamed
    between revisions is a real case an engineer can recognise and the tool
    cannot. The override must not erase the doubt."""
    job_id = _upload(client, NOT_A_PAIR_A, NOT_A_PAIR_B).json()["job_id"]
    assert _wait(client, job_id)["status"] == "refused"

    assert client.post(f"/api/jobs/{job_id}/force").status_code == 202
    body = _wait(client, job_id)

    assert body["status"] == "done"
    assert body["forced"] is True
    assert body["precheck"]["is_pair"] is False    # still says so, forever
    assert client.get(f"/api/jobs/{job_id}/payload").status_code == 200


def test_force_on_a_healthy_job_is_rejected(client, done_job):
    assert client.post(f"/api/jobs/{done_job}/force").status_code == 409


def test_weak_identity_is_surfaced_as_a_flag(client, done_job):
    pre = client.get(f"/api/jobs/{done_job}").json()["precheck"]
    assert pre["weak_identity"] == (pre["identity_tier"] in ("tag_overlap", "none"))


# --------------------------------------------------------------------------
# upload guards
# --------------------------------------------------------------------------

def test_non_pdf_upload_is_rejected_by_magic_bytes(client, tmp_path):
    fake = tmp_path / "drawing.pdf"          # right extension, wrong content
    fake.write_bytes(b"AC1027 this is really a DWG")

    r = _upload(client, fake, SAMPLE_B)

    assert r.status_code == 415
    assert "not a PDF" in r.json()["detail"]


def test_oversized_upload_is_rejected(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1024)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.7" + b"\0" * 5000)

    assert _upload(client, big, SAMPLE_B).status_code == 413


def test_a_rejected_upload_leaves_no_job_behind(client, tmp_path):
    fake = tmp_path / "drawing.pdf"
    fake.write_bytes(b"not a pdf at all")
    store = app_module.app.state.store

    _upload(client, fake, SAMPLE_B)

    assert store._jobs == {}


# --------------------------------------------------------------------------
# lifecycle and isolation
# --------------------------------------------------------------------------

def test_unknown_job_is_404_everywhere(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/payload").status_code == 404
    assert client.get("/api/jobs/deadbeef/pdf/a").status_code == 404
    assert client.post("/api/jobs/deadbeef/chat", json={"question": "x"}).status_code == 404


def test_deleting_a_job_removes_its_artifacts_and_page_rasters(client, done_job):
    from src.ingest.pdf_native import RASTER_CACHE_DIR

    job = app_module.app.state.store.get(done_job)
    job_dir = job.dir
    rasters = list(pathlib.Path(RASTER_CACHE_DIR).glob(f"{done_job}_*.png"))
    assert rasters, "ingest should have written job-scoped page rasters"

    assert client.delete(f"/api/jobs/{done_job}").status_code == 200

    assert not job_dir.exists()
    assert not list(pathlib.Path(RASTER_CACHE_DIR).glob(f"{done_job}_*.png"))


@pytest.mark.skipif(not VALVES_A.exists(), reason="real_pair_valves not present")
def test_concurrent_jobs_do_not_share_page_rasters(client):
    """The test that justifies the job-scoped pid. Both ingest adapters
    write raster_cache/{pid}_sheet{n}.png, and the CLI hardcodes pid to
    "A"/"B" -- so before the fix, two jobs in flight would overwrite each
    other's page images and each report would show the other's drawing.
    Nothing else in the suite would have caught that."""
    id1 = _upload(client, SAMPLE_A, SAMPLE_B).json()["job_id"]
    id2 = _upload(client, VALVES_A, VALVES_B).json()["job_id"]

    b1, b2 = _wait(client, id1), _wait(client, id2)
    assert b1["status"] == "done" and b2["status"] == "done", (b1, b2)

    store = app_module.app.state.store
    j1, j2 = store.get(id1), store.get(id2)
    paths = []
    for job in (j1, j2):
        paths += list(job.doc_a.raster_paths.values()) + list(job.doc_b.raster_paths.values())

    assert len(paths) == len(set(paths)), f"page rasters collided across jobs: {paths}"
    for job, job_id in ((j1, id1), (j2, id2)):
        for p in list(job.doc_a.raster_paths.values()) + list(job.doc_b.raster_paths.values()):
            assert job_id in pathlib.Path(p).name


def test_two_jobs_keep_independent_payloads(client):
    id1 = _upload(client, SAMPLE_A, SAMPLE_B).json()["job_id"]
    id2 = _upload(client, SAMPLE_B, SAMPLE_A).json()["job_id"]   # reversed
    _wait(client, id1)
    _wait(client, id2)

    p1 = client.get(f"/api/jobs/{id1}/payload").json()
    p2 = client.get(f"/api/jobs/{id2}/payload").json()

    assert p1["job_id"] != p2["job_id"]
    # Reversing the pair turns adds into removes, so these must not be equal.
    kinds1 = sorted(d["kind"] for d in p1["deltas"])
    kinds2 = sorted(d["kind"] for d in p2["deltas"])
    assert kinds1 != kinds2 or all(k in ("modify", "move") for k in kinds1)


# --------------------------------------------------------------------------
# the frontend is actually served
# --------------------------------------------------------------------------

def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>" in r.text


def test_static_assets_are_served(client):
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    # app.js imports these two as ES modules; a 404 on either leaves the page
    # blank with only a console error to show for it.
    assert client.get("/static/md.js").status_code == 200
    assert client.get("/static/vendor/pdfjs/pdf.min.mjs").status_code == 200
