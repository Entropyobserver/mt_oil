import sys
from pathlib import Path
import json
import pandas as pd
import hydra
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
src_path = project_root / "10_src"
sys.path.insert(0, str(src_path))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from evaluation.base_evaluator import BaseEvaluator
from utils.seed_manager import SeedManager
from utils.logger import Logger


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    
    output_dir = Path(cfg.paths.output_dir) / "02_optuna_stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger.get_logger("optuna_stage2", cfg, output_dir / "experiment.log")
    logger.info("="*80)
    logger.info("EXPERIMENT 2c: OPTUNA STAGE 2 - VALIDATION ON FULL DATASET")
    logger.info("="*80)
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    stage1_dir = Path(cfg.paths.output_dir) / "02_optuna_stage1"
    top_configs_file = stage1_dir / "top_configs.json"
    
    if not top_configs_file.exists():
        logger.error(f"Top configs file not found: {top_configs_file}")
        logger.error("Please run Stage 1 first: python 03_optuna_stage1.py")
        return
    
    with open(top_configs_file) as f:
        top_configs = json.load(f)
    
    logger.info(f"Loaded {len(top_configs)} configurations from Stage 1")
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=False)
    
    logger.info("Loading full dataset")
    train_ds, val_ds, test_ds = data_manager.load_splits()
    
    recommended_size_file = Path(cfg.paths.output_dir) / "01_data_scaling" / "best_model.json"
    if recommended_size_file.exists():
        with open(recommended_size_file) as f:
            best_info = json.load(f)
            recommended_size = best_info['data_size']
        logger.info(f"Using recommended size from Experiment 1: {recommended_size}")
        train_ds = train_ds.subset(recommended_size, seed=cfg.project.random_seed)
    
    logger.info(f"Train: {len(train_ds)} samples")
    logger.info(f"Val: {len(val_ds)} samples")
    logger.info(f"Test: {len(test_ds)} samples")
    
    num_runs = 3
    logger.info(f"\nValidation setup:")
    logger.info(f"  Configurations to validate: {len(top_configs)}")
    logger.info(f"  Runs per configuration: {num_runs}")
    logger.info(f"  Total experiments: {len(top_configs) * num_runs}")
    
    results = []
    best_bleu = 0.0
    best_config = None
    best_model_path = None
    
    for config_idx, config in enumerate(top_configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {config_idx+1}/{len(top_configs)}")
        logger.info(f"  r={config['r']}, alpha={config['alpha']}, dropout={config['dropout']}")
        logger.info(f"  Stage 1 BLEU: {config['stage1_bleu']:.4f}")
        logger.info(f"{'='*60}")
        
        for run in range(num_runs):
            logger.info(f"\n  Run {run+1}/{num_runs}")
            
            seed = cfg.project.random_seed + run
            SeedManager.set_seed(seed)
            
            trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
            
            run_dir = output_dir / f"config_{config_idx}_run_{run}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            train_config = {
                'output_dir': str(run_dir / "training"),
                'r': config['r'],
                'alpha': config['alpha'],
                'dropout': config['dropout'],
                'epochs': cfg.training.num_train_epochs,
                'batch_size': cfg.training.batch_size.train,
                'gradient_accumulation_steps': cfg.training.gradient_accumulation_steps,
                'learning_rate': cfg.training.optimizer.learning_rate,
                'eval_steps': cfg.training.evaluation.eval_steps,
                'save_steps': cfg.training.save.save_steps,
                'early_stopping_patience': cfg.training.early_stopping.patience,
                'save_total_limit': 1,
                'save_final_model': True
            }
            
            try:
                train_data = [s.to_dict() for s in train_ds.samples]
                val_data = [s.to_dict() for s in val_ds.samples]
                
                train_result = trainer.train(train_data, val_data, train_config)
                
                test_predictions = trainer.generate_predictions(
                    train_result['model'],
                    test_ds,
                    batch_size=8
                )
                
                sources = [s.source for s in test_ds.samples]
                references = [s.target for s in test_ds.samples]
                
                test_metrics = evaluator.evaluate_all(sources, test_predictions, references)
                
                result_entry = {
                    'config_id': config_idx,
                    'run': run,
                    'seed': seed,
                    'r': config['r'],
                    'alpha': config['alpha'],
                    'dropout': config['dropout'],
                    'stage1_bleu': config['stage1_bleu'],
                    'val_bleu': train_result['bleu'],
                    'val_chrf': train_result['chrf'],
                    'test_bleu': test_metrics['bleu'],
                    'test_chrf': test_metrics['chrf'],
                    'model_path': train_result['final_model_path']
                }
                results.append(result_entry)
                
                logger.info(f"    Val  - BLEU: {train_result['bleu']:.4f}, chrF: {train_result['chrf']:.2f}")
                logger.info(f"    Test - BLEU: {test_metrics['bleu']:.4f}, chrF: {test_metrics['chrf']:.2f}")
                
                if test_metrics['bleu'] > best_bleu:
                    best_bleu = test_metrics['bleu']
                    best_config = config
                    best_model_path = train_result['final_model_path']
                    logger.info(f"    *** NEW BEST MODEL ***")
                
                with open(run_dir / "metrics.json", 'w') as f:
                    json.dump({
                        'config': config,
                        'run': run,
                        'seed': seed,
                        'val_metrics': {
                            'bleu': train_result['bleu'],
                            'chrf': train_result['chrf']
                        },
                        'test_metrics': test_metrics
                    }, f, indent=2)
                
            except Exception as e:
                logger.error(f"    Run {run+1} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                results.append({
                    'config_id': config_idx,
                    'run': run,
                    'seed': seed,
                    'r': config['r'],
                    'alpha': config['alpha'],
                    'dropout': config['dropout'],
                    'stage1_bleu': config['stage1_bleu'],
                    'val_bleu': 0.0,
                    'val_chrf': 0.0,
                    'test_bleu': 0.0,
                    'test_chrf': 0.0,
                    'model_path': None,
                    'failed': True
                })
    
    logger.info(f"\n{'='*80}")
    logger.info("OPTUNA STAGE 2 VALIDATION COMPLETE")
    logger.info(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    valid_df = results_df[~results_df.get('failed', False)]
    
    if not valid_df.empty:
        summary = valid_df.groupby('config_id').agg({
            'test_bleu': ['mean', 'std', 'min', 'max'],
            'test_chrf': ['mean', 'std']
        }).round(4)
        
        logger.info("\nValidation Summary by Configuration:")
        logger.info(f"\n{summary}")
        
        summary_file = output_dir / "summary_statistics.csv"
        summary.to_csv(summary_file)
    
    if best_config and best_model_path:
        logger.info(f"\nBest Validated Configuration:")
        logger.info(f"  r={best_config['r']}, alpha={best_config['alpha']}, dropout={best_config['dropout']}")
        logger.info(f"  Stage 1 BLEU: {best_config['stage1_bleu']:.4f}")
        logger.info(f"  Stage 2 Test BLEU: {best_bleu:.4f}")
        logger.info(f"  Model Path: {best_model_path}")
        
        final_best_config = {
            'r': best_config['r'],
            'alpha': best_config['alpha'],
            'dropout': best_config['dropout'],
            'stage1_bleu': best_config['stage1_bleu'],
            'stage2_test_bleu': best_bleu,
            'model_path': str(best_model_path),
            'recommended_for_production': True
        }
        
        with open(output_dir / "best_config.json", 'w') as f:
            json.dump(final_best_config, f, indent=2)
        
        logger.info(f"\nBest config saved to {output_dir / 'best_config.json'}")
        logger.info("Use this configuration for Experiment 3 (Final Evaluation)")
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("Experiment 2c Stage 2 completed successfully!")


if __name__ == "__main__":
    main()