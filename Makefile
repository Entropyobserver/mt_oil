.PHONY: help setup-env setup-dvc install test clean

help:
	@echo "Available commands:"
	@echo "  make setup-env       - Create conda environment"
	@echo "  make install         - Install dependencies"
	@echo "  make setup-dvc       - Initialize DVC"
	@echo "  make test            - Run all tests"
	@echo "  make clean           - Clean temporary files"
	@echo "  make exp-scaling     - Run data scaling experiment"
	@echo "  make exp-gridsearch  - Run grid search experiment"
	@echo "  make exp-optuna      - Run Optuna optimization"
	@echo "  make exp-final       - Run final evaluation"
	@echo "  make exp-bidirectional - Compare translation directions"
	@echo "  make exp-model-comparison - Compare NLLB vs M2M-100"
	@echo "  make eval-comprehensive - Comprehensive evaluation with all metrics"
	@echo "  make analyze-all     - Analyze all experiment results"

setup-env:
	conda env create -f environment.yml
	conda activate nob-eng-mt

install:
	pip install -e .
	pre-commit install

setup-dvc:
	dvc init
	dvc remote add -d storage s3://your-bucket/dvc-storage

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/

exp-scaling:
	python 20_experiments/21_baseline/01_data_scaling.py

exp-gridsearch:
	python 20_experiments/21_baseline/02_gridsearch.py

exp-optuna:
	python 20_experiments/21_baseline/03_optuna_stage1.py
	python 20_experiments/21_baseline/04_optuna_stage2.py

exp-final:
	python 20_experiments/21_baseline/05_final_eval.py

exp-bidirectional:
	python experiments/bidirectional/compare_directions.py

exp-model-comparison:
	python experiments/model_comparison/nllb_vs_m2m.py

eval-comprehensive:
	@echo "Usage: make eval-comprehensive MODEL_DIR=path/to/model/output"
	python experiments/evaluation/comprehensive_evaluation.py \
		--model-dir $(MODEL_DIR) \
		--use-terminology \
		--dataset npd

analyze-all:
	python 30_analysis/01_analyze_data_scaling.py
	python 30_analysis/02_analyze_gridsearch.py
	python 30_analysis/03_analyze_optuna.py
	python 30_analysis/04_generate_paper_plots.py

mlflow-ui:
	mlflow ui --backend-store-uri file:///$(PWD)/50_mlruns

format:
	black src/ 20_experiments/ 30_analysis/ 70_tests/
	isort src/ 20_experiments/ 30_analysis/ 70_tests/

lint:
	ruff check src/ 20_experiments/ 30_analysis/ 70_tests/
	mypy src/