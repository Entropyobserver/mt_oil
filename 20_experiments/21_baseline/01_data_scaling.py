import sys
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from evaluation.base_evaluator import BaseEvaluator
from utils.seed_manager import SeedManager
from utils.logger import Logger
from utils.visualization import Visualizer
from utils.mlflow_wrapper import MLflowWrapper


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    logger = Logger.get_logger(__name__, cfg)
    logger.info("Starting data scaling experiment")
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=False)
    visualizer = Visualizer()
    mlflow_wrapper = MLflowWrapper(cfg)
    
    data_sizes = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    results = []
    
    for size in data_sizes:
        logger.info(f"Training with {size} samples")
        
        train_ds, val_ds, test_ds = data_manager.load_splits()
        train_ds = train_ds.subset(size, seed=cfg.project.random_seed)
        
        trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
        
        train_config = {
            'output_dir': str(Path(cfg.paths.output_dir) / f"size_{size}"),
            'r': cfg.adapter.r,
            'alpha': cfg.adapter.lora_alpha,
            'dropout': cfg.adapter.lora_dropout,
            'epochs': cfg.training.num_train_epochs,
            'batch_size': cfg.training.batch_size.train,
            'gradient_accumulation_steps': cfg.training.gradient_accumulation_steps,
            'learning_rate': cfg.training.optimizer.learning_rate,
            'eval_steps': cfg.training.evaluation.eval_steps,
            'save_steps': cfg.training.save.save_steps,
            'early_stopping_patience': cfg.training.early_stopping.patience
        }
        
        train_data = [s.to_dict() for s in train_ds.samples]
        val_data = [s.to_dict() for s in val_ds.samples]
        
        result = trainer.train(train_data, val_data, train_config)
        predictions = trainer.generate_predictions(result['model'], test_ds)
        
        sources = [s.source for s in test_ds.samples]
        references = [s.target for s in test_ds.samples]
        
        metrics = evaluator.evaluate_all(sources, predictions, references)
        
        results.append({
            'data_size': size,
            'bleu': metrics['bleu'],
            'chrf': metrics['chrf']
        })
        
        logger.info(f"Size {size}: BLEU={metrics['bleu']:.4f}, chrF={metrics['chrf']:.2f}")
    
    results_df = pd.DataFrame(results)
    results_path = Path(cfg.paths.output_dir) / "data_scaling_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    visualizer.plot_learning_curve(
        results_df,
        x_col='data_size',
        y_cols=['bleu', 'chrf'],
        save_path=Path(cfg.paths.output_dir) / "learning_curve.png",
        title="Data Scaling Analysis"
    )
    
    logger.info("Data scaling experiment completed")


if __name__ == "__main__":
    main()