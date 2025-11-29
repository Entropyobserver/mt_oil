import sys
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from evaluation.base_evaluator import BaseEvaluator
from evaluation.error_analyzer import ErrorAnalyzer
from utils.seed_manager import SeedManager
from utils.logger import Logger
from utils.mlflow_wrapper import MLflowWrapper


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    logger = Logger.get_logger(__name__, cfg)
    logger.info("Starting final evaluation")
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=True)
    error_analyzer = ErrorAnalyzer()
    mlflow_wrapper = MLflowWrapper(cfg)
    
    logger.info("Loading data splits")
    train_ds, val_ds, test_ds = data_manager.load_splits()
    
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
    
    train_config = {
        'output_dir': str(Path(cfg.paths.output_dir) / "final_model"),
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
    
    logger.info("Training final model")
    train_data = [s.to_dict() for s in train_ds.samples]
    val_data = [s.to_dict() for s in val_ds.samples]
    train_result = trainer.train(train_data, val_data, train_config)
    
    logger.info("Generating predictions on test set")
    predictions = trainer.generate_predictions(train_result['model'], test_ds)
    
    sources = [s.source for s in test_ds.samples]
    references = [s.target for s in test_ds.samples]
    
    logger.info("Evaluating predictions")
    metrics = evaluator.evaluate_all(sources, predictions, references)
    
    logger.info("\nFinal Test Results:")
    logger.info(f"BLEU: {metrics['bleu']:.4f}")
    logger.info(f"chrF: {metrics['chrf']:.2f}")
    if 'comet' in metrics:
        logger.info(f"COMET: {metrics['comet']:.4f}")
    
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    error_analysis = error_analyzer.analyze_samples(sources, predictions, references)
    error_path = output_dir / "error_analysis.json"
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)
    
    score_dist = error_analyzer.compute_score_distribution(predictions, references)
    dist_path = output_dir / "score_distribution.json"
    with open(dist_path, 'w') as f:
        json.dump(score_dist, f, indent=2)
    
    predictions_output = [
        {
            'source': src,
            'prediction': pred,
            'reference': ref
        }
        for src, pred, ref in zip(sources, predictions, references)
    ]
    pred_path = output_dir / "test_predictions.json"
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("Final evaluation completed")


if __name__ == "__main__":
    main()