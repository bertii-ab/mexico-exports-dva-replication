.PHONY: all help install check-env stage1 stage2 stage3 run figures clean

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Replication Commands:"
	@echo "  make install     Install required Python packages"
	@echo "  make run         Run full replication pipeline (Stages 2 & 3)"
	@echo "  make stage1      Acquire data from APIs (OECD, BIS)"
	@echo "  make stage2      Run ANN-MIDAS temporal disaggregation"
	@echo "  make stage3      Estimate NARDL models and causality tests"
	@echo "  make clean       Remove temporary Python cache files"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) run_pipeline.py

stage1:
	$(PYTHON) scripts/stage1_data_acquisition/data_extraction_mt.py

stage2:
	$(PYTHON) scripts/stage2_ann_midas/ann_midas.py

stage3:
	$(PYTHON) scripts/stage3_econometrics/nardl.py
	$(PYTHON) scripts/stage3_econometrics/toda_yamamoto.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
