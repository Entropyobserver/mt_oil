import sys
from pathlib import Path
import json
import pandas as pd
import hydra
from omegaconf import DictConfig
from itertools import product

project_root = Path(__file__).parent.parent.parent
src_path = project_root / "10_src"
sys.path.insert(0, str(src_path))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from evaluation.base_evaluator import BaseEvaluator
from utils.seed_manager import SeedManager
from utils.logger import Logger
from utils.visualization import Visualizer


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    
    output_dir = Path(cfg.paths.output_dir) / "02_gridsearch"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger.get_logger("gridsearch", cfg, output_dir / "experiment.log")
    logger.info("="*80)
    logger.info("EXPERIMENT 2a: GRID SEARCH HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=False)
    visualizer = Visualizer()
    
    logger.info("Loading data splits")
    train_ds, val_ds, test_ds = data_manager.load_splits()
    
    recommended_size_file = Path(cfg.paths.output_dir) / "01_data_scaling" / "best_model.json"
    if recommended_size_file.exists():
        with open(recommended_size_file) as f:
            best_info = json.load(f)
            recommended_size = best_info['data_size']
        logger.info(f"Using recommended size from Experiment 1: {recommended_size}")
        train_ds = train_ds.subset(recommended_size, seed=cfg.project.random_seed)
    else:
        logger.info("No recommended size found, using 8000 samples")
        train_ds = train_ds.subset(8000, seed=cfg.project.random_seed)
    
    logger.info(f"Train: {len(train_ds)} samples")
    logger.info(f"Val: {len(val_ds)} samples")
    logger.info(f"Test: {len(test_ds)} samples")
    
    if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'parameter_grid'):
        r_values = cfg.experiment.parameter_grid.r
        alpha_values = cfg.experiment.parameter_grid.alpha
        dropout_values = cfg.experiment.parameter_grid.dropout
    else:
        r_values = [8, 16, 32]
        alpha_values = [16, 32, 64]
        dropout_values = [0.0, 0.1, 0.2]
    
    logger.info(f"\nParameter Grid:")
    logger.info(f"  r: {r_values}")
    logger.info(f"  alpha: {alpha_values}")
    logger.info(f"  dropout: {dropout_values}")
    
    configs = list(product(r_values, alpha_values, dropout_values))
    total_configs = len(configs)
    logger.info(f"\nTotal configurations: {total_configs}")
    
    results = []
    best_bleu = 0.0
    best_config = None
    best_model_path = None
    
    for idx, (r, alpha, dropout) in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {idx+1}/{total_configs}: r={r}, alpha={alpha}, dropout={dropout}")
        logger.info(f"{'='*60}")
        
        config_name = f"r{r}_a{alpha}_d{dropout}"
        config_output_dir = output_dir / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
        
        train_config = {
            'output_dir': str(config_output_dir / "training"),
            'r': r,
            'alpha': alpha,
            'dropout': dropout,
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
            
            logger.info("Training model...")
            train_result = trainer.train(train_data, val_data, train_config)
            
            logger.info("Generating predictions...")
            test_predictions = trainer.generate_predictions(
                train_result['model'],
                test_ds,
                batch_size=8
            )
            
            sources = [s.source for s in test_ds.samples]
            references = [s.target for s in test_ds.samples]
            
            test_metrics = evaluator.evaluate_all(sources, test_predictions, references)
            
            result_entry = {
                'r': r,
                'alpha': alpha,
                'dropout': dropout,
                'val_bleu': train_result['bleu'],
                'val_chrf': train_result['chrf'],
                'val_loss': train_result['loss'],
                'test_bleu': test_metrics['bleu'],
                'test_chrf': test_metrics['chrf'],
                'model_path': train_result['final_model_path']
            }
            results.append(result_entry)
            
            logger.info(f"Results:")
            logger.info(f"  Val  - BLEU: {train_result['bleu']:.4f}, chrF: {train_result['chrf']:.2f}")
            logger.info(f"  Test - BLEU: {test_metrics['bleu']:.4f}, chrF: {test_metrics['chrf']:.2f}")
            
            if test_metrics['bleu'] > best_bleu:
                best_bleu = test_metrics['bleu']
                best_config = {'r': r, 'alpha': alpha, 'dropout': dropout}
                best_model_path = train_result['final_model_path']
                logger.info(f"  *** NEW BEST CONFIGURATION ***")
            
            with open(config_output_dir / "metrics.json", 'w') as f:
                json.dump({
                    'config': {'r': r, 'alpha': alpha, 'dropout': dropout},
                    'val_metrics': {
                        'bleu': train_result['bleu'],
                        'chrf': train_result['chrf'],
                        'loss': train_result['loss']
                    },
                    'test_metrics': test_metrics
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            results.append({
                'r': r,
                'alpha': alpha,
                'dropout': dropout,
                'val_bleu': 0.0,
                'val_chrf': 0.0,
                'val_loss': 999.0,
                'test_bleu': 0.0,
                'test_chrf': 0.0,
                'model_path': None,
                'failed': True
            })
    
    logger.info(f"\n{'='*80}")
    logger.info("GRID SEARCH COMPLETE")
    logger.info(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if not results_df.empty and 'failed' in results_df.columns:
        valid_df = results_df[~results_df['failed']]
    else:
        valid_df = results_df
    
    if len(valid_df) > 0:
        logger.info("\nGenerating visualizations...")
        
        if len(valid_df) > 1:
            visualizer.plot_heatmap(
                valid_df,
                index_col='alpha',
                columns_col='r',
                values_col='test_bleu',
                save_path=output_dir / "heatmap_bleu.png",
                title="BLEU Score Heatmap: Alpha vs Rank"
            )
            logger.info(f"Heatmap saved to {output_dir / 'heatmap_bleu.png'}")
        
        top_5 = valid_df.nlargest(5, 'test_bleu')
        logger.info("\nTop 5 Configurations:")
        for idx, row in top_5.iterrows():
            logger.info(f"  {idx+1}. r={row['r']}, alpha={row['alpha']}, dropout={row['dropout']}")
            logger.info(f"     Test BLEU: {row['test_bleu']:.4f}")
    
    if best_config and best_model_path:
        logger.info(f"\nBest Configuration:")
        logger.info(f"  r={best_config['r']}, alpha={best_config['alpha']}, dropout={best_config['dropout']}")
        logger.info(f"  Test BLEU: {best_bleu:.4f}")
        logger.info(f"  Model Path: {best_model_path}")
        
        best_config_info = {
            **best_config,
            'test_bleu': best_bleu,
            'model_path': str(best_model_path)
        }
        
        with open(output_dir / "best_config.json", 'w') as f:
            json.dump(best_config_info, f, indent=2)
        
        logger.info(f"\nBest config saved to {output_dir / 'best_config.json'}")
        logger.info("Use this configuration for Experiment 3 (Final Evaluation)")
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("Experiment 2a completed successfully!")


if __name__ == "__main__":
    main()