import sys
from pathlib import Path
import json
import random

import pandas as pd
import hydra
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader
from src.models.lora_trainer import LoRATrainer
from src.utils.seed_manager import SeedManager
from src.utils.logger import ExperimentLogger
from src.utils.visualization import Visualizer


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    
    output_dir = Path(cfg.paths.output_dir) / "data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ExperimentLogger("data_scaling", output_dir)
    logger.info("Starting data scaling experiment")
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    data_loader = DataLoader(random_seed=cfg.project.random_seed)
    train_data, val_data, _ = data_loader.load_splits(cfg.data.paths.splits)
    
    logger.info(f"Loaded {len(train_data)} training samples")
    logger.info(f"Loaded {len(val_data)} validation samples")
    
    sample_sizes = [100, 500, 1000, 2000, 4000, 6000, 8000, len(train_data)]
    seeds = [42, 123, 456]
    
    results = []
    
    for seed in seeds:
        SeedManager.set_seed(seed)
        
        for size in sample_sizes:
            logger.info(f"Training with seed={seed}, size={size}")
            
            train_subset = data_loader.sample_data(train_data, size)
            
            trainer = LoRATrainer(
                model_name=cfg.model.pretrained_name,
                src_lang=cfg.model.tokenizer.src_lang,
                tgt_lang=cfg.model.tokenizer.tgt_lang
            )
            
            config = {
                "output_dir": str(output_dir / f"seed_{seed}_size_{size}"),
                "r": cfg.adapter.r,
                "alpha": cfg.adapter.lora_alpha,
                "dropout": cfg.adapter.lora_dropout,
                "epochs": cfg.training.num_train_epochs,
                "batch_size": cfg.training.batch_size.train,
                "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
                "learning_rate": cfg.training.optimizer.learning_rate,
                "eval_steps": cfg.training.evaluation.eval_steps,
                "save_steps": cfg.training.save.save_steps,
                "early_stopping_patience": cfg.training.early_stopping.patience
            }
            
            try:
                result = trainer.train(train_subset, val_data, config)
                
                results.append({
                    "seed": seed,
                    "data_size": size,
                    "bleu": result["bleu"],
                    "chrf": result["chrf"],
                    "loss": result["loss"]
                })
                
                logger.log_metrics(result, prefix=f"Size {size}: ")
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                results.append({
                    "seed": seed,
                    "data_size": size,
                    "bleu": 0.0,
                    "chrf": 0.0,
                    "loss": 999.0,
                    "failed": True
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    visualizer = Visualizer()
    
    summary = results_df.groupby("data_size").agg({
        "bleu": ["mean", "std"],
        "chrf": ["mean", "std"]
    }).reset_index()
    
    visualizer.plot_learning_curve(
        summary,
        output_dir / "learning_curve.png",
        x_col="data_size",
        y_col=("bleu", "mean")
    )
    
    logger.info("Data scaling experiment completed")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()