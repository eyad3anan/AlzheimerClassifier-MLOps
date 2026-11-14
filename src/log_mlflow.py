"""Log a finished `dvc repro` run (params, history, metrics, model) to MLflow.

Uses MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD from the
environment (DagsHub); without them it logs to ./mlruns.
"""
import json
import os
import subprocess
from pathlib import Path

import mlflow
import yaml

from src.train import HISTORY_PATH, MODEL_PATH


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())
    commit = os.environ.get("GITHUB_SHA") or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()

    mlflow.set_experiment("alzheimer-resnet50")
    with mlflow.start_run(run_name=os.environ.get("RUN_NAME")) as run:
        mlflow.set_tag("git_commit", commit)
        mlflow.log_params({f"{section}.{k}": v for section in ("data", "train") for k, v in params[section].items()})
        for name, values in json.loads(HISTORY_PATH.read_text()).items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(name, value, step=epoch)
        test_metrics = json.loads(Path("metrics.json").read_text())
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        for path in (MODEL_PATH, Path("metrics.json"), Path("reports/confusion_matrix.json")):
            mlflow.log_artifact(str(path))
        print(f"MLflow run {run.info.run_id} -> {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
