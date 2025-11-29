import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).parent.parent.parent
src_path = project_root / "10_src"
sys.path.insert(0, str(src_path))

from data.data_loader import DataManager
from models.model_factory import ModelFactory
from evaluation.base_evaluator import BaseEvaluator
from evaluation.error_analyzer import ErrorAnalyzer
from utils.seed_manager import SeedManager
from utils.logger import Logger


class SimpleModelRegistry:
    
    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self):
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {"models": [], "production": None}
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_path, name, experiment, metrics, config, description=""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{name}_{timestamp}"
        
        storage_path = self.registry_dir / "experiments" / experiment / model_id
        storage_path.mkdir(parents=True, exist_ok=True)
        
        if Path(model_path).is_dir():
            shutil.copytree(model_path, storage_path / "model", dirs_exist_ok=True)
        
        metadata = {
            "id": model_id,
            "name": name,
            "experiment": experiment,
            "timestamp": timestamp,
            "metrics": metrics,
            "config": config,
            "description": description,
            "path": str(storage_path / "model")
        }
        
        with open(storage_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.registry["models"].append(metadata)
        self._save_registry()
        
        return storage_path / "model", model_id
    
    def set_production(self, model_id, version="v1.0"):
        model_metadata = None
        for model in self.registry["models"]:
            if model["id"] == model_id:
                model_metadata = model
                break
        
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        production_dir = self.registry_dir / "production"
        production_dir.mkdir(parents=True, exist_ok=True)
        
        versioned_dir = production_dir / version
        if versioned_dir.exists():
            shutil.rmtree(versioned_dir)
        
        shutil.copytree(Path(model_metadata["path"]), versioned_dir)
        
        self.registry["production"] = {
            "model_id": model_id,
            "version": version,
            "path": str(versioned_dir),
            "promoted_at": datetime.now().isoformat()
        }
        self._save_registry()
        
        return versioned_dir


@hydra.main(version_base=None, config_path="../../00_configs", config_name="00_config")
def main(cfg: DictConfig):
    
    output_dir = Path(cfg.paths.output_dir) / "03_final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger.get_logger("final_eval", cfg, output_dir / "experiment.log")
    logger.info("="*80)
    logger.info("EXPERIMENT 3: FINAL EVALUATION & MODEL REGISTRATION")
    logger.info("="*80)
    
    SeedManager.set_seed(cfg.project.random_seed)
    logger.info(f"Random seed: {cfg.project.random_seed}")
    
    data_manager = DataManager(cfg)
    evaluator = BaseEvaluator(use_comet=True)
    error_analyzer = ErrorAnalyzer()
    registry = SimpleModelRegistry(registry_dir=Path(cfg.paths.models_dir))
    
    logger.info("\nLoading data splits")
    train_ds, val_ds, test_ds = data_manager.load_splits()
    logger.info(f"Train: {len(train_ds)} samples")
    logger.info(f"Val: {len(val_ds)} samples")
    logger.info(f"Test: {len(test_ds)} samples")
    
    best_config_sources = [
        Path(cfg.paths.output_dir) / "02_optuna_stage2" / "best_config.json",
        Path(cfg.paths.output_dir) / "02_gridsearch" / "best_config.json",
    ]
    
    best_config = None
    for source in best_config_sources:
        if source.exists():
            with open(source) as f:
                best_config = json.load(f)
            logger.info(f"Loaded best config from: {source}")
            break
    
    if best_config:
        r = best_config['r']
        alpha = best_config['alpha']
        dropout = best_config['dropout']
        logger.info(f"Using best hyperparameters:")
        logger.info(f"  r={r}, alpha={alpha}, dropout={dropout}")
    else:
        r = cfg.adapter.r
        alpha = cfg.adapter.lora_alpha
        dropout = cfg.adapter.lora_dropout
        logger.info(f"Using default hyperparameters from config:")
        logger.info(f"  r={r}, alpha={alpha}, dropout={dropout}")
    
    trainer = ModelFactory.create_trainer(cfg, trainer_type='lora')
    
    train_config = {
        'output_dir': str(output_dir / "training"),
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
        'save_total_limit': 2,
        'save_final_model': True
    }
    
    logger.info("\nStarting final model training...")
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {train_config['epochs']}")
    logger.info(f"  Batch size: {train_config['batch_size']}")
    logger.info(f"  Learning rate: {train_config['learning_rate']}")
    
    train_data = [s.to_dict() for s in train_ds.samples]
    val_data = [s.to_dict() for s in val_ds.samples]
    
    train_result = trainer.train(train_data, val_data, train_config)
    
    logger.info("\nTraining completed!")
    logger.info(f"  Val BLEU: {train_result['bleu']:.4f}")
    logger.info(f"  Val chrF: {train_result['chrf']:.2f}")
    logger.info(f"  Val Loss: {train_result['loss']:.4f}")
    
    logger.info("\nGenerating predictions on test set...")
    test_predictions = trainer.generate_predictions(
        train_result['model'],
        test_ds,
        batch_size=8
    )
    
    sources = [s.source for s in test_ds.samples]
    references = [s.target for s in test_ds.samples]
    
    logger.info("Evaluating predictions...")
    test_metrics = evaluator.evaluate_all(sources, test_predictions, references)
    
    logger.info(f"\n{'='*60}")
    logger.info("FINAL TEST SET RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  BLEU:  {test_metrics['bleu']:.4f}")
    logger.info(f"  chrF:  {test_metrics['chrf']:.2f}")
    if 'comet' in test_metrics:
        logger.info(f"  COMET: {test_metrics['comet']:.4f}")
    logger.info(f"{'='*60}")
    
    logger.info("\nSaving evaluation results...")
    
    final_results = {
        'configuration': {
            'r': r,
            'alpha': alpha,
            'dropout': dropout,
            'training_samples': len(train_ds),
            'model_name': cfg.model.pretrained_name
        },
        'validation_metrics': {
            'bleu': train_result['bleu'],
            'chrf': train_result['chrf'],
            'loss': train_result['loss']
        },
        'test_metrics': test_metrics,
        'dataset_sizes': {
            'train': len(train_ds),
            'val': len(val_ds),
            'test': len(test_ds)
        }
    }
    
    results_file = output_dir / "final_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"  Results: {results_file}")
    
    logger.info("\nPerforming error analysis...")
    error_analysis = error_analyzer.analyze_samples(sources, test_predictions, references)
    error_file = output_dir / "error_analysis.json"
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"  Error analysis: {error_file}")
    
    score_dist = error_analyzer.compute_score_distribution(test_predictions, references)
    dist_file = output_dir / "score_distribution.json"
    with open(dist_file, 'w') as f:
        json.dump(score_dist, f, indent=2)
    logger.info(f"  Score distribution: {dist_file}")
    
    predictions_output = [
        {'source': src, 'prediction': pred, 'reference': ref}
        for src, pred, ref in zip(sources, test_predictions, references)
    ]
    pred_file = output_dir / "test_predictions.json"
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_output, f, ensure_ascii=False, indent=2)
    logger.info(f"  Predictions: {pred_file}")
    
    logger.info("\nRegistering model in model registry...")
    final_model_path = Path(train_result['final_model_path'])
    
    registered_path, model_id = registry.register_model(
        model_path=final_model_path,
        name="nob-eng-lora-final",
        experiment="final_evaluation",
        metrics={
            "test_bleu": test_metrics['bleu'],
            "test_chrf": test_metrics['chrf'],
            "test_comet": test_metrics.get('comet', 0.0),
            "val_bleu": train_result['bleu'],
            "val_chrf": train_result['chrf']
        },
        config=OmegaConf.to_container(cfg, resolve=True),
        description="Final production model trained on full NPD dataset"
    )
    
    logger.info(f"  Model ID: {model_id}")
    logger.info(f"  Registered at: {registered_path}")
    
    if test_metrics['bleu'] > 0.45:
        logger.info("\nPromoting model to production...")
        try:
            production_path = registry.set_production(model_id, version="v1.0")
            logger.info(f"  Production path: {production_path}")
            logger.info(f"  Model promoted to production as v1.0")
            
            model_card = f"""# Norwegian-English Translation Model

## Model Information
- **Model ID**: {model_id}
- **Version**: v1.0
- **Base Model**: {cfg.model.pretrained_name}
- **Training Method**: LoRA Fine-tuning

## Performance
- **Test BLEU**: {test_metrics['bleu']:.4f}
- **Test chrF**: {test_metrics['chrf']:.2f}
{"- **Test COMET**: " + f"{test_metrics['comet']:.4f}" if 'comet' in test_metrics else ""}

## Configuration
- **LoRA Rank (r)**: {r}
- **LoRA Alpha**: {alpha}
- **Dropout**: {dropout}

## Training Data
- **Training samples**: {len(train_ds)}
- **Validation samples**: {len(val_ds)}
- **Test samples**: {len(test_ds)}
- **Dataset**: Norwegian Petroleum Directorate (NPD)

## Usage
```python
from pathlib import Path
from models.lora_trainer import LoRATrainer

trainer = LoRATrainer()
model = trainer.load_model(Path("{production_path}"))
predictions = trainer.generate_predictions(model, test_dataset)
```

## Deployment
This model is ready for production deployment.
Location: `{production_path}`
"""
            
            card_file = production_path / "MODEL_CARD.md"
            with open(card_file, 'w') as f:
                f.write(model_card)
            logger.info(f"  Model card: {card_file}")
            
        except Exception as e:
            logger.error(f"Failed to promote to production: {e}")
    else:
        logger.warning(f"\nModel BLEU ({test_metrics['bleu']:.4f}) below threshold (0.45)")
        logger.warning("Model registered but NOT promoted to production")
    
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT 3 COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nAll outputs saved to:")
    logger.info(f"  Experiment: {output_dir}")
    logger.info(f"  Registry: {registry.registry_dir}")
    logger.info(f"\nFinal model summary:")
    logger.info(f"  Model ID: {model_id}")
    logger.info(f"  Test BLEU: {test_metrics['bleu']:.4f}")
    logger.info(f"  Registry: {registry.registry_file}")
    
    logger.info("\nExperiment 3 completed successfully!")


if __name__ == "__main__":
    main()