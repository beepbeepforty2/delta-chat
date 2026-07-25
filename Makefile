.PHONY: install dataset run chat eval markup visual-diff trace test
install:
	pip install -e ".[dev]"
dataset:            ## generate labeled eval pairs (seeded, reproducible)
	cd eval/datasets && python -m generator.generate --out v0 --n 6 --seed 42
run:                ## ingest a pair and produce a delta report
	python -m src.cli run --a $(A) --b $(B) --out reports/
chat:
	python -m src.cli chat --a $(A) --b $(B)
eval:
	python -m eval.run_eval --dataset eval/datasets/v0
markup:
	python -m src.cli markup --a $(A) --b $(B) --out reports/
visual-diff:        ## human-in-the-loop debug viewer (tools/visual_diff.py)
	python -m tools.visual_diff --a $(A) --b $(B) --out $(or $(OUT),visual_diff.html)
trace:              ## pretty-print a trace: make trace ID=<correlation_id>
	python -m src.observability.print_trace $(ID)
test:
	pytest tests/ -q
