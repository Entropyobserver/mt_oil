import sys
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "10_src"))

from src.data.data_loader import DataManager
from src.models.model_factory import ModelFactory
from src.evaluation.base_evaluator import BaseEvaluator
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.utils.seed_manager import SeedManager
from src.utils.logger import Logger
from src.utils.mlflow_wrapper import MLflowWrapper


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    logger = Logger.get_logger(__name__, cfg)
    logger.info("Starting final evaluation")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    SeedManager.set_seed(cfg.project.random_seed)
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=True)
    error_analyzer = ErrorAnalyzer()
    mlflow_wrapper = MLflowWrapper(cfg)
    
    logger.info("Loading data splits")
    train_ds, val_ds, test_ds = data_manager.load_splits()
    
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
    
    logger.info("Training final model")
    train_result = trainer.train(
        train_dataset=train_ds,
        eval_dataset=val_ds
    )
    
    logger.info("Generating predictions on test set")
    predictions = trainer.generate_predictions(
        train_result['model'],
        test_ds
    )
    
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
    
    mlflow_wrapper.log_experiment_results(
        metrics=metrics,
        model=train_result['model'],
        config=cfg,
        artifacts={
            'results': results_path,
            'error_analysis': error_path,
            'predictions': pred_path
        }
    )
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("Final evaluation completed")


if __name__ == "__main__":
    main()