\# Norwegian-English Translation with LoRA



Production-ready machine translation system for Norwegian Bokmål to English using LoRA fine-tuning on NLLB-200.



\## Quick Start



\### Installation



```bash

pip install -r requirements.txt

```



\### Run Experiments



```bash

make train-scaling

make train-param

make eval

make analyze

```



\### View Results



```bash

make mlflow

```



Navigate to http://localhost:5000



\## Project Structure



```

nob-eng-translation/

├── configs/                  # Configuration files

│   ├── config.yaml          # Global config

│   ├── exp01\_scaling.yaml   # Data scaling config

│   ├── exp02\_param\_search.yaml

│   └── exp03\_final\_eval.yaml

├── src/                      # Source code

│   ├── \_\_init\_\_.py

│   ├── data\_loader.py       # Data loading

│   ├── lora\_trainer.py      # Model training

│   ├── metrics.py           # Evaluation metrics

│   ├── visualizer.py        # Plotting

│   └── mlflow\_tracker.py    # MLflow wrapper

├── experiments/              # Experiment scripts

│   ├── exp01\_data\_scaling.py

│   ├── exp02\_param\_search.py

│   ├── exp03\_final\_eval.py

│   ├── analysis01\_scaling.py

│   └── analysis02\_param.py

├── jobs/                     # SLURM job files

│   ├── scaling.sbatch

│   └── param\_search.sbatch

├── tests/                    # Unit tests

│   └── test\_all.py

├── data/                     # Data directory

├── outputs/                  # Experiment outputs

├── mlruns/                   # MLflow tracking

├── Makefile

├── requirements.txt

└── README.md

```



\## Configuration



Edit `configs/config.yaml` for global settings:



```yaml

model:

&nbsp; name: facebook/nllb-200-distilled-600M



lora:

&nbsp; r: 16

&nbsp; alpha: 32

&nbsp; dropout: 0.1



training:

&nbsp; epochs: 3

&nbsp; batch\_size: 4

&nbsp; learning\_rate: 5e-4

```



\## Experiments



\### 1. Data Scaling



```bash

python experiments/exp01\_data\_scaling.py

python experiments/analysis01\_scaling.py

```



\### 2. Hyperparameter Search



```bash

python experiments/exp02\_param\_search.py

python experiments/analysis02\_param.py

```



\### 3. Final Evaluation



```bash

python experiments/exp03\_final\_eval.py

```



\## HPC Usage



```bash

sbatch jobs/scaling.sbatch

sbatch jobs/param\_search.sbatch

```



\## Testing



```bash

make test

```



\## Results



Results are saved to:

\- `outputs/`: Experiment outputs

\- `mlruns/`: MLflow tracking data



View in MLflow UI:

```bash

make mlflow

```



\## License



MIT License

