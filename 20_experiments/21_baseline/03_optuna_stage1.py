import sys
from pathlib import Path
import json
import hydra
from omegaconf import DictConfig
import optuna
from optuna.pruners import SuccessiveHalvingPruner

project_root = Path(__file__).parent.parent.parent
src_path = project_root / "10_src"
sys.path.insert(0, str(src_path))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from utils.seed_manager import SeedManager
from utils.logger import Logger


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    
    output_dir = Path(cfg.paths.output_dir) / "02_optuna_stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger.get_logger("optuna_stage1", cfg, output_dir / "experiment.log")
    logger.info("="*80)
    logger.info("EXPERIMENT 2b: OPTUNA STAGE 1 - COARSE SEARCH")
    logger.info("="*80)
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    data_manager = DataManager(cfg)
    
    logger.info("Loading data splits with subset for efficient search")
    train_ds, val_ds, _ = data_manager.load_splits(use_subset=2000)
    
    logger.info(f"Train: {len(train_ds)} samples (subset for efficiency)")
    logger.info(f"Val: {len(val_ds)} samples")
    
    num_trials = 50
    if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'num_trials'):
        num_trials = cfg.experiment.num_trials
    
    logger.info(f"\nOptuna Configuration:")
    logger.info(f"  Number of trials: {num_trials}")
    logger.info(f"  Sampler: TPE")
    logger.info(f"  Pruner: Successive Halving")
    logger.info(f"  Search space:")
    logger.info(f"    r: [8, 16, 32]")
    logger.info(f"    alpha: [16, 32, 64]")
    logger.info(f"    dropout: [0.0, 0.2]")
    
    train_data = [s.to_dict() for s in train_ds.samples]
    val_data = [s.to_dict() for s in val_ds.samples]
    
    def objective(trial):
        
        r = trial.suggest_categorical('r', [8, 16, 32])
        alpha = trial.suggest_categorical('alpha', [16, 32, 64])
        dropout = trial.suggest_float('dropout', 0.0, 0.2, step=0.05)
        
        logger.info(f"\nTrial {trial.number}:")
        logger.info(f"  r={r}, alpha={alpha}, dropout={dropout}")
        
        trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
        
        trial_dir = output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        train_config = {
            'output_dir': str(trial_dir / "training"),
            'r': r,
            'alpha': alpha,
            'dropout': dropout,
            'epochs': cfg.training.num_train_epochs,
            'batch_size': cfg.training.batch_size.train,
            'gradient_accumulation_steps': cfg.training.gradient_accumulation_steps,
            'learning_rate': cfg.training.optimizer.learning_rate,
            'eval_steps': cfg.training.evaluation.eval_steps,
            'save_steps': cfg.training.save.save_steps,
            'early_stopping_patience': 2,
            'save_total_limit': 1,
            'save_final_model': False
        }
        
        try:
            result = trainer.train(train_data, val_data, train_config)
            bleu_score = result['bleu']
            
            logger.info(f"  Result: BLEU={bleu_score:.4f}")
            
            with open(trial_dir / "result.json", 'w') as f:
                json.dump({
                    'trial': trial.number,
                    'params': {'r': r, 'alpha': alpha, 'dropout': dropout},
                    'bleu': bleu_score,
                    'chrf': result['chrf'],
                    'loss': result['loss']
                }, f, indent=2)
            
            return bleu_score
            
        except Exception as e:
            logger.error(f"  Trial {trial.number} failed: {e}")
            return 0.0
    
    logger.info("\nStarting Optuna optimization...")
    
    study = optuna.create_study(
        direction='maximize',
        pruner=SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0
        ),
        study_name='nllb_lora_stage1'
    )
    
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("OPTUNA STAGE 1 COMPLETE")
    logger.info(f"{'='*80}")
    
    trials_df = study.trials_dataframe()
    results_csv = output_dir / "results.csv"
    trials_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    logger.info(f"\nSummary:")
    logger.info(f"  Completed trials: {len(completed_trials)}")
    logger.info(f"  Pruned trials: {len(pruned_trials)}")
    logger.info(f"  Total trials: {len(study.trials)}")
    if len(study.trials) > 0:
        logger.info(f"  Pruning efficiency: {len(pruned_trials)/len(study.trials)*100:.1f}%")
    
    if completed_trials:
        best_trial = study.best_trial
        logger.info(f"\nBest Trial:")
        logger.info(f"  Number: {best_trial.number}")
        logger.info(f"  Parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
        logger.info(f"  Value (BLEU): {best_trial.value:.4f}")
        
        top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
        logger.info(f"\nTop 5 Trials for Stage 2 Validation:")
        
        top_configs = []
        for i, trial in enumerate(top_trials):
            logger.info(f"\n  {i+1}. Trial {trial.number}:")
            logger.info(f"     r={trial.params['r']}, alpha={trial.params['alpha']}, dropout={trial.params['dropout']}")
            logger.info(f"     BLEU: {trial.value:.4f}")
            
            top_configs.append({
                'trial_number': trial.number,
                'r': trial.params['r'],
                'alpha': trial.params['alpha'],
                'dropout': trial.params['dropout'],
                'stage1_bleu': trial.value
            })
        
        with open(output_dir / "top_configs.json", 'w') as f:
            json.dump(top_configs, f, indent=2)
        
        logger.info(f"\nTop configurations saved to {output_dir / 'top_configs.json'}")
        logger.info("Run Stage 2 to validate these configs on full dataset:")
        logger.info("  python 04_optuna_stage2.py")
    else:
        logger.warning("\nNo completed trials. Please check your setup.")
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("Experiment 2b Stage 1 completed successfully!")


if __name__ == "__main__":
    main()