.PHONY: bench

PYTHON ?= python

bench:
	PYTHONPATH=src $(PYTHON) bench/run_benchmarks.py --bars 120000 --output-dir bench/results
