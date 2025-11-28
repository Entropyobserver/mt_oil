.PHONY: help install test train-scaling train-param eval analyze mlflow clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run tests"
	@echo "  make train-scaling  - Run data scaling experiment"
	@echo "  make train-param    - Run parameter search"
	@echo "  make eval           - Run final evaluation"
	@echo "  make analyze        - Run all analyses"
	@echo "  make mlflow         - Start MLflow UI"
	@echo "  make clean          - Clean outputs"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

train-scaling:
	python experiments/exp01_data_scaling.py

train-param:
	python experiments/exp02_param_search.py

eval:
	python experiments/exp03_final_eval.py

analyze:
	python experiments/analysis01_scaling.py
	python experiments/analysis02_param.py

mlflow:
	mlflow ui --backend-store-uri mlruns --port 5000

clean:
	rm -rf outputs/*/checkpoints
	rm -rf __pycache__ */__pycache__
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete