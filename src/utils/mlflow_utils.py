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
        if self.enabled:
            tracking_uri = config.get('mlflow', {}).get('tracking_uri', None)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            experiment_name = config.get('mlflow', {}).get('experiment_name', 'copyair')
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        if not self.enabled:
            return None
        
        if run_name is None:
            run_name = f"{self.config['model'].get('architecture', 'unet')}_{self.config['training'].get('optimizer', {}).get('type', 'adam')}"
        
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        if not self.enabled:
            return

        flat_params = self._flatten(params)
        # mlflow.log_params expects string/int/float values; convert complex types to json strings
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
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(self, model, artifact_path: str = 'model'):
        if self.enabled:
            try:
                mlflow.pytorch.log_model(model, artifact_path=artifact_path)
            except Exception as e:
                logger.warning(f"No se pudo loggear el modelo en MLflow: {e}")

    def end_run(self):
        if self.enabled:
            mlflow.end_run()

    def _flatten(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, sep=sep))
            else:
                # Convert non-primitive values to JSON strings so MLflow can store them
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
        run = mlflow.active_run()
        return run.info.run_id if run else None
