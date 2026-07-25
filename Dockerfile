# delta-chat: document delta computation and grounded chat over document
# revisions. See README.md for the full project write-up.
#
# The build bakes in the seeded eval dataset (`make dataset`) and runs the
# full test suite (`make test`) as a build step -- a failed test fails the
# build, so a successfully built image is itself evidence the containerized
# environment is correct, not just the host dev environment it was built on.
# This is fully hermetic: no LLM credentials are needed at build time (every
# chat-related test injects a fake call_llm, never a live API call).
FROM python:3.11-slim

# tesseract-ocr is a system binary pytesseract wraps (not pip-installable),
# needed by src/ingest/pdf_scanned.py for the scanned-PDF adapter. Nothing
# else here needs a system package -- PyMuPDF/numpy/scipy all ship
# self-contained manylinux wheels for this base image's platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

RUN make dataset && make test

# No reason to run a CLI tool as root.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Zero-config default: the deterministic-only scorecard (delta P/R/F1,
# calibration, semantic-null detection, null-pair/not-a-pair checks) --
# works immediately with no credential and no volume mount. The chat/
# llm_direct sections need a live LLM credential (src/chat/llm.py raises
# without one), so they're off by default here rather than crashing the
# container on first run; see README's Docker section for how to enable
# them via --env-file.
CMD ["python", "-m", "eval.run_eval", "--dataset", "eval/datasets/v0", "--skip-chat", "--skip-baseline"]
