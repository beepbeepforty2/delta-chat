.PHONY: install dataset run html-report web chat eval eval-holdout markup visual-diff trace test uicheck
install:
	uv sync --extra dev
web:                ## browser UI: pick two PDFs, see what changed, ask about it
	uv run --extra web uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --reload
dataset:            ## generate labeled eval pairs (seeded, reproducible)
	cd eval/datasets && uv run python -m generator.generate --out v0 --n 6 --seed 42
run:                ## ingest a pair and produce a delta report
	uv run python -m src.cli run --a $(A) --b $(B) --out reports/
html-report:        ## same as run, plus an interactive reports/report.html
	uv run python -m src.cli run --a $(A) --b $(B) --out reports/ --html
chat:
	uv run python -m src.cli chat --a $(A) --b $(B)
eval:               ## scorecard on the seeded set -- NEVER pooled with holdout
	uv run python -m eval.run_eval --dataset eval/datasets/v0
eval-holdout:       ## held-out real-P&ID scorecard; report separately from `eval`
	DELTA_RASTER_DIFF=1 uv run python -m eval.run_eval \
	  --dataset eval/datasets/holdout --levels L0 --skip-chat --skip-baseline
markup:
	uv run python -m src.cli markup --a $(A) --b $(B) --out reports/
visual-diff:        ## human-in-the-loop debug viewer (tools/visual_diff.py)
	uv run python -m tools.visual_diff --a $(A) --b $(B) --out $(or $(OUT),visual_diff.html)
trace:              ## pretty-print a trace: make trace ID=<correlation_id>
	uv run python -m src.observability.print_trace $(ID)
test:
	uv run pytest tests/ -q
uicheck:            ## optional browser checks of the web UI -- needs `make web` running
	@node -e "require.resolve('playwright-core')" 2>/dev/null \
	  || { echo "run: npm install playwright-core   (a driver, not a browser -- it uses your installed Chrome)"; exit 1; }
	node tools/uicheck.mjs
