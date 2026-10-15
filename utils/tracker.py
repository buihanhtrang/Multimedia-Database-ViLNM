"""Experiment tracking.

`Tracker` is a thin wrapper over MLflow that becomes a no-op when `--mlflow` is
off, so the trainer never has to check.  The MLflow run id is stored in every
checkpoint; a resumed session reopens that run, so a training job split over
several sessions shows up as one continuous run in the UI.
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger("ViLNM.tracker")


def _flatten(prefix: str, d: Dict) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(key, v))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


class Tracker:
    def __init__(self, args, run_id: Optional[str] = None):
        self.enabled = bool(getattr(args, "mlflow", False))
        self.run_id = run_id
        if not self.enabled:
            return
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        # MLflow 3's usage-telemetry thread sends events to MLflow's servers and
        # draws from Python's global `random` at arbitrary moments, which breaks
        # bit-exact resume.  Off unless the user explicitly turns it back on.
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        import mlflow
        self._mlflow = mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment or args.dataset_name)
        run = mlflow.start_run(run_id=run_id,
                               run_name=None if run_id else args.name,
                               log_system_metrics=args.mlflow_system_metrics)
        self.run_id = run.info.run_id
        if run_id is None:
            mlflow.log_params({k: str(v)[:500] for k, v in vars(args).items()})
            mlflow.set_tag("output_dir", args.output_dir)
        logger.info("MLflow run %s (%s) at %s", self.run_id,
                    "resumed" if run_id else "new", args.mlflow_uri)

    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        if self.enabled:
            self._mlflow.log_metrics(_flatten(prefix, metrics), step=step)

    def set_tags(self, tags: Dict):
        if self.enabled:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: str):
        if self.enabled and os.path.exists(path):
            self._mlflow.log_artifact(path)

    def close(self, status: str = "FINISHED"):
        """status: FINISHED when training completed, KILLED when paused."""
        if self.enabled:
            self._mlflow.end_run(status=status)
