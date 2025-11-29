import sys
from pathlib import Path
import json
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

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
    
    output_dir = Path(cfg.paths.output_dir) / "01_data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger.get_logger("data_scaling", cfg, output_dir / "experiment.log")
    logger.info("="*80)
    logger.info("EXPERIMENT 1: DATA SCALING ANALYSIS")
    logger.info("="*80)
    
    SeedManager.set_seed(cfg.project.random_seed)
    logger.info(f"Random seed: {cfg.project.random_seed}")
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=False)
    visualizer = Visualizer()
    
    logger.info("Loading data splits")
    train_ds, val_ds, test_ds = data_manager.load_splits()
    logger.info(f"Train: {len(train_ds)} samples")
    logger.info(f"Val: {len(val_ds)} samples")
    logger.info(f"Test: {len(test_ds)} samples")
    
    data_sizes = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    if len(train_ds) < max(data_sizes):
        data_sizes = [s for s in data_sizes if s <= len(train_ds)]
    data_sizes.append(len(train_ds))
    data_sizes = sorted(list(set(data_sizes)))
    
    logger.info(f"Data sizes to test: {data_sizes}")
    
    results = []
    best_bleu = 0.0
    best_size = None
    best_model_path = None
    
    for idx, size in enumerate(data_sizes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {idx+1}/{len(data_sizes)}: Training with {size} samples")
        logger.info(f"{'='*60}")
        
        train_subset = train_ds.subset(size, seed=cfg.project.random_seed)
        logger.info(f"Training subset created: {len(train_subset)} samples")
        
        trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
        
        size_output_dir = output_dir / f"size_{size}"
        size_output_dir.mkdir(parents=True, exist_ok=True)
        
        train_config = {
            'output_dir': str(size_output_dir / "training"),
            'r': cfg.adapter.r,
            'alpha': cfg.adapter.lora_alpha,
            'dropout': cfg.adapter.lora_dropout,
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
            train_data = [s.to_dict() for s in train_subset.samples]
            val_data = [s.to_dict() for s in val_ds.samples]
            
            logger.info("Starting training...")
            train_result = trainer.train(train_data, val_data, train_config)
            
            logger.info("Generating predictions on test set...")
            test_predictions = trainer.generate_predictions(
                train_result['model'], 
                test_ds,
                batch_size=8
            )
            
            sources = [s.source for s in test_ds.samples]
            references = [s.target for s in test_ds.samples]
            
            logger.info("Evaluating predictions...")
            test_metrics = evaluator.evaluate_all(sources, test_predictions, references)
            
            result_entry = {
                'data_size': size,
                'val_bleu': train_result['bleu'],
                'val_chrf': train_result['chrf'],
                'val_loss': train_result['loss'],
                'test_bleu': test_metrics['bleu'],
                'test_chrf': test_metrics['chrf'],
                'model_path': train_result['final_model_path']
            }
            results.append(result_entry)
            
            logger.info(f"Results for size {size}:")
            logger.info(f"  Val  - BLEU: {train_result['bleu']:.4f}, chrF: {train_result['chrf']:.2f}")
            logger.info(f"  Test - BLEU: {test_metrics['bleu']:.4f}, chrF: {test_metrics['chrf']:.2f}")
            
            if test_metrics['bleu'] > best_bleu:
                best_bleu = test_metrics['bleu']
                best_size = size
                best_model_path = train_result['final_model_path']
                logger.info(f"  *** NEW BEST MODEL ***")
            
            result_file = size_output_dir / "metrics.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'val_metrics': {
                        'bleu': train_result['bleu'],
                        'chrf': train_result['chrf'],
                        'loss': train_result['loss']
                    },
                    'test_metrics': test_metrics
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Training failed for size {size}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            results.append({
                'data_size': size,
                'val_bleu': 0.0,
                'val_chrf': 0.0,
                'val_loss': 999.0,
                'test_bleu': 0.0,
                'test_chrf': 0.0,
                'model_path': None,
                'failed': True
            })
    
    logger.info(f"\n{'='*80}")
    logger.info("DATA SCALING ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")
    
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    summary_df = results_df.groupby('data_size').agg({
        'test_bleu': ['mean', 'std'],
        'test_chrf': ['mean', 'std']
    }).round(4)
    
    logger.info("\nSummary Statistics:")
    logger.info(f"\n{summary_df}")
    
    if len(results_df[~results_df.get('failed', False)]) > 1:
        logger.info("\nGenerating visualizations...")
        
        valid_df = results_df[~results_df.get('failed', False)]
        
        visualizer.plot_learning_curve(
            valid_df,
            x_col='data_size',
            y_cols=['test_bleu', 'test_chrf'],
            save_path=output_dir / "learning_curve.png",
            title="Data Scaling Analysis",
            xlabel="Training Data Size"
        )
        logger.info(f"Learning curve saved to {output_dir / 'learning_curve.png'}")
    
    if best_model_path:
        logger.info(f"\nBest Model:")
        logger.info(f"  Data Size: {best_size}")
        logger.info(f"  Test BLEU: {best_bleu:.4f}")
        logger.info(f"  Model Path: {best_model_path}")
        
        best_model_info = {
            'data_size': best_size,
            'test_bleu': best_bleu,
            'model_path': str(best_model_path),
            'recommended_for_next_experiments': True
        }
        
        with open(output_dir / "best_model.json", 'w') as f:
            json.dump(best_model_info, f, indent=2)
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("Experiment 1 completed successfully!")


if __name__ == "__main__":
    main()