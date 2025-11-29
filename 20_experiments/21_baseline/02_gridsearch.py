import sys
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from evaluation.base_evaluator import BaseEvaluator
from utils.seed_manager import SeedManager
from utils.logger import Logger


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    logger = Logger.get_logger(__name__, cfg)
    logger.info("Starting grid search experiment")
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=False)
    
    train_ds, val_ds, test_ds = data_manager.load_splits()
    
    param_grid = cfg.experiment.parameter_grid
    r_values = param_grid.r
    alpha_values = param_grid.alpha
    dropout_values = param_grid.dropout
    
    results = []
    total_configs = len(r_values) * len(alpha_values) * len(dropout_values)
    config_num = 0
    
    for r in r_values:
        for alpha in alpha_values:
            for dropout in dropout_values:
                config_num += 1
                logger.info(f"Config {config_num}/{total_configs}: r={r}, alpha={alpha}, dropout={dropout}")
                
                for run in range(cfg.experiment.num_runs):
                    logger.info(f"  Run {run + 1}/{cfg.experiment.num_runs}")
                    
                    trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
                    
                    train_config = {
                        'output_dir': str(Path(cfg.paths.output_dir) / f"gs_r{r}_a{alpha}_d{dropout}_run{run}"),
                        'r': r,
                        'alpha': alpha,
                        'dropout': dropout,
                        'epochs': cfg.training.num_train_epochs,
                        'batch_size': cfg.training.batch_size.train,
                        'gradient_accumulation_steps': cfg.training.gradient_accumulation_steps,
                        'learning_rate': cfg.training.optimizer.learning_rate,
                        'eval_steps': cfg.training.evaluation.eval_steps,
                        'save_steps': cfg.training.save.save_steps,
                        'early_stopping_patience': cfg.training.early_stopping.patience
                    }
                    
                    try:
                        train_data = [s.to_dict() for s in train_ds.samples]
                        val_data = [s.to_dict() for s in val_ds.samples]
                        
                        result = trainer.train(train_data, val_data, train_config)
                        predictions = trainer.generate_predictions(result['model'], test_ds)
                        
                        sources = [s.source for s in test_ds.samples]
                        references = [s.target for s in test_ds.samples]
                        
                        metrics = evaluator.evaluate_all(sources, predictions, references)
                        
                        results.append({
                            'r': r,
                            'alpha': alpha,
                            'dropout': dropout,
                            'run': run,
                            'val_bleu': result['bleu'],
                            'val_chrf': result['chrf'],
                            'test_bleu': metrics['bleu'],
                            'test_chrf': metrics['chrf']
                        })
                        
                        logger.info(f"    BLEU: {metrics['bleu']:.4f}, chrF: {metrics['chrf']:.2f}")
                        
                    except Exception as e:
                        logger.error(f"Config failed: {e}")
    
    results_df = pd.DataFrame(results)
    results_path = Path(cfg.paths.output_dir) / "gridsearch_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    if not results_df.empty:
        best_config = results_df.loc[results_df['test_bleu'].idxmax()]
        logger.info(f"\nBest Configuration:")
        logger.info(f"  r={best_config['r']}, alpha={best_config['alpha']}, dropout={best_config['dropout']}")
        logger.info(f"  Test BLEU: {best_config['test_bleu']:.4f}")
    
    logger.info("Grid search completed")


if __name__ == "__main__":
    main()