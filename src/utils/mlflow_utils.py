import mlflow
import mlflow.pytorch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MLflowLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('mlflow', {}).get('enabled', False)
        self.active_run = None
        
        if self.enabled:
            tracking_uri = config.get('mlflow', {}).get('tracking_uri', None)
            if tracking_uri:
                # Force setting the tracking URI from config to avoid Env Var conflicts if desired
                # Or just respect it. But here we want to ensure it works.
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow Tracking URI set to: {tracking_uri}")
                
            experiment_name = config.get('mlflow', {}).get('experiment_name', 'copyair')
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        if not self.enabled:
            return None
        
        if run_name is None:
            run_name = f"{self.config['model'].get('architecture', 'unet')}_{self.config['training'].get('optimizer', {}).get('type', 'adam')}"
        
        # Start run and store it
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run

    def log_params(self, params: Dict[str, Any]):
        if not self.enabled:
            return

        flat_params = self._flatten(params)
        safe_params = {
            k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else (v if v is not None else 'null')) 
            for k, v in flat_params.items()
        }
        try:
            mlflow.log_params(safe_params)
        except Exception as e:
            logger.warning(f"No se pudieron registrar todos los parámetros en MLflow: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self.enabled:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.enabled:
            try:
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
            except Exception as e:
                # Avoid crashing if artifact logging fails (common with SQLite without artifact store)
                logger.warning(f"Failed to log artifact {local_path}: {e}")

    def log_model(self, model, artifact_path: str = 'model'):
        if self.enabled:
            try:
                mlflow.pytorch.log_model(model, artifact_path=artifact_path)
            except Exception as e:
                logger.warning(f"No se pudo loggear el modelo en MLflow: {e}")

    def end_run(self):
        if self.enabled:
            mlflow.end_run()
            self.active_run = None

    def _flatten(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, sep=sep))
            else:
                if isinstance(v, (list, tuple, dict)):
                    try:
                        items[new_key] = json.dumps(v)
                    except Exception:
                        items[new_key] = str(v)
                else:
                    items[new_key] = v
        return items


    def get_run_id(self) -> Optional[str]:
        if not self.enabled:
            return None
        
        if self.active_run:
            return self.active_run.info.run_id
            
        # Fallback to global active run
        run = mlflow.active_run()
        return run.info.run_id if run else None
