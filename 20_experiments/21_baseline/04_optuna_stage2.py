import sys
from pathlib import Path
import json
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

from src.data.data_loader import DataManager
from src.models.model_factory import ModelFactory
from src.evaluation.base_evaluator import BaseEvaluator
from src.utils.seed_manager import SeedManager
from src.utils.logger import Logger


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    logger = Logger.get_logger(__name__, cfg)
    logger.info("Stage 2: Fine validation on full dataset")

    SeedManager.set_seed(cfg.project.random_seed)

    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=False)

    train_ds, val_ds, test_ds = data_manager.load_splits()

    results_path = Path(cfg.paths.output_dir) / "optuna_stage1_results.csv"
    if not results_path.exists():
        logger.error("Stage 1 results not found. Run stage 1 first.")
        return

    import pandas as pd
    trials_df = pd.read_csv(results_path)

    best_trials = trials_df.nlargest(3, 'values_0')

    final_results = []

    for idx, trial in best_trials.iterrows():
        params = {
            'r': int(trial['params_r']),
            'alpha': int(trial['params_alpha']),
            'dropout': float(trial['params_dropout'])
        }

        logger.info(f"Validating config: {params}")

        for run in range(3):
            trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')

            train_config = {
                'output_dir': Path(cfg.paths.output_dir) / f"stage2_config_{idx}_run_{run}",
                'r': params['r'],
                'alpha': params['alpha'],
                'dropout': params['dropout'],
                'epochs': cfg.training.num_train_epochs,
                'batch_size': cfg.training.batch_size.train,
                'gradient_accumulation_steps': cfg.training.gradient_accumulation_steps,
                'learning_rate': cfg.training.optimizer.learning_rate,
                'eval_steps': cfg.training.evaluation.eval_steps,
                'save_steps': cfg.training.save.save_steps,
                'early_stopping_patience': cfg.training.early_stopping.patience
            }

            try:
                result = trainer.train(
                    [s.to_dict() for s in train_ds.samples],
                    [s.to_dict() for s in val_ds.samples],
                    train_config
                )

                predictions = trainer.generate_predictions(result['model'], test_ds)
                sources = [s.source for s in test_ds.samples]
                references = [s.target for s in test_ds.samples]

                metrics = evaluator.evaluate_all(sources, predictions, references)

                final_results.append({
                    'config_id': idx,
                    'run': run,
                    'r': params['r'],
                    'alpha': params['alpha'],
                    'dropout': params['dropout'],
                    'val_bleu': result['bleu'],
                    'test_bleu': metrics['bleu'],
                    'test_chrf': metrics['chrf']
                })

                logger.info(f"  Run {run}: BLEU={metrics['bleu']:.4f}, chrF={metrics['chrf']:.2f}")

            except Exception as e:
                logger.error(f"Validation failed: {e}")

    results_df = pd.DataFrame(final_results)
    results_path = Path(cfg.paths.output_dir) / "optuna_stage2_results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.loc[results_df['test_bleu'].idxmax()]

    best_config = {
        'r': int(best_row['r']),
        'alpha': int(best_row['alpha']),
        'dropout': float(best_row['dropout']),
        'test_bleu': float(best_row['test_bleu']),
        'test_chrf': float(best_row['test_chrf'])
    }

    config_path = Path(cfg.paths.output_dir) / "best_config.json"
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    logger.info(f"\nBest Configuration Saved:")
    logger.info(f"  r={best_config['r']}, alpha={best_config['alpha']}, dropout={best_config['dropout']}")
    logger.info(f"  Test BLEU: {best_config['test_bleu']:.4f}")

    logger.info("Stage 2 completed")


if __name__ == "__main__":
    main()