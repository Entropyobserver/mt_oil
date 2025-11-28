from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import mlflow


class MLflowWrapper:
    def __init__(self, config: DictConfig):
        self.config = config
        self.tracking_uri = Path(config.paths.mlruns_dir).absolute().as_uri()
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.experiment_name = config.experiment.name
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        all_tags = {}
        
        if hasattr(self.config.experiment, 'tags'):
            all_tags.update(self.config.experiment.tags)
        
        if tags:
            all_tags.update(tags)
        
        return mlflow.start_run(run_name=run_name, tags=all_tags)
    
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_params_from_config(self, config: DictConfig):
        flat_config = self._flatten_dict(OmegaConf.to_container(config, resolve=True))
        self.log_params(flat_config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None):
        mlflow.log_artifact(str(local_path), artifact_path)
    
    def log_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)
    
    def end_run(self):
        mlflow.end_run()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_experiment_results(
        self,
        metrics: Dict[str, float],
        model,
        config: DictConfig,
        artifacts: Optional[Dict[str, Path]] = None
    ):
        
        with self.start_run():
            self.log_params_from_config(config)
            self.log_metrics(metrics)
            
            if artifacts:
                for name, path in artifacts.items():
                    self.log_artifact(path, artifact_path=name)
            
            if model:
                self.log_model(model, "model")