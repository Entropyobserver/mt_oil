import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import optuna
from optuna.pruners import SuccessiveHalvingPruner

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from utils.seed_manager import SeedManager
from utils.logger import Logger


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    logger = Logger.get_logger(__name__, cfg)
    logger.info("Stage 1: Coarse hyperparameter search with Optuna")

    SeedManager.set_seed(cfg.project.random_seed)

    data_manager = DataManager(cfg)
    train_ds, val_ds, _ = data_manager.load_splits(use_subset=2000)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    def objective(trial):
        r = trial.suggest_categorical('r', [8, 16, 32])
        alpha = trial.suggest_categorical('alpha', [16, 32, 64])
        dropout = trial.suggest_float('dropout', 0.0, 0.2, step=0.05)

        logger.info(f"Trial {trial.number}: r={r}, alpha={alpha}, dropout={dropout}")

        trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')

        train_config = {
            'output_dir': str(Path(cfg.paths.output_dir) / f"optuna_trial_{trial.number}"),
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
            return result['bleu']

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(
        direction='maximize',
        pruner=SuccessiveHalvingPruner(),
        study_name='nllb_optuna_stage1'
    )

    study.optimize(objective, n_trials=cfg.experiment.num_trials)

    trials_df = study.trials_dataframe()
    results_path = Path(cfg.paths.output_dir) / "optuna_stage1_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(results_path, index=False)

    best_trial = study.best_trial
    logger.info(f"\nBest Trial: {best_trial.number}")
    logger.info(f"  Parameters: {best_trial.params}")
    logger.info(f"  Value: {best_trial.value:.4f}")

    logger.info("Stage 1 completed")


if __name__ == "__main__":
    main()