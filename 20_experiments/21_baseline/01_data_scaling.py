import sys
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

from src.data.data_loader import DataManager
from src.models.model_factory import ModelFactory
from src.evaluation.base_evaluator import BaseEvaluator
from src.utils.seed_manager import SeedManager
from src.utils.logger import Logger
from src.utils.visualization import Visualizer
from src.utils.mlflow_wrapper import MLflowWrapper


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    logger = Logger.get_logger(__name__, cfg)
    logger.info("Starting data scaling experiment")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
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
        
        train_result = trainer.train(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            output_dir=Path(cfg.paths.output_dir) / f"size_{size}"
        )
        
        predictions = trainer.generate_predictions(
            train_result['model'],
            test_ds
        )
        
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
    results_df.to_csv(results_path, index=False)
    
    visualizer.plot_learning_curve(
        results_df,
        x_col='data_size',
        y_cols=['bleu', 'chrf'],
        save_path=Path(cfg.paths.output_dir) / "learning_curve.png",
        title="Data Scaling Analysis"
    )
    
    mlflow_wrapper.log_experiment_results(
        metrics={'final_bleu': results_df['bleu'].iloc[-1]},
        model=None,
        config=cfg,
        artifacts={'results': results_path}
    )
    
    logger.info("Data scaling experiment completed")


if __name__ == "__main__":
    main()