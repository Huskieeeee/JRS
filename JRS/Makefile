PYTHON ?= $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
.PHONY: setup check test doctor smoke prepare-smoke gpu-check
setup:
	PYTHON=$(PYTHON) bash scripts/dev/setup.sh
check:
	PYTHON=$(PYTHON) bash scripts/dev/check.sh
test:
	$(PYTHON) -m unittest discover -s tests -v
doctor:
	$(PYTHON) -m floor_tase doctor
smoke:
	$(PYTHON) -m floor_tase smoke --out results/dev_smoke --workers 1
prepare-smoke:
	$(PYTHON) -m floor_tase prepare smoke --device cpu
gpu-check:
	$(PYTHON) -m floor_tase gpu --require-cuda
