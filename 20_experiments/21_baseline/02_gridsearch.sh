#!/bin/bash
#SBATCH -A uppmax2025-3-5
#SBATCH -M snowy
#SBATCH -p node
#SBATCH --time=72:00:00
#SBATCH -J mt_data_scaling
#SBATCH --gres=gpu:1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2025-3-5/private/yaxj1/conda_envs/mt
pip install --upgrade accelerate>=0.26.0
cd /proj/uppmax2025-3-5/private/yaxj1/mt_oli_full/
python 20_experiments/21_baseline/01_data_scaling.py 