import mlflow
from pathlib import Path

def setup_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0")

def log_run_artifacts(run_dir):
    for p in Path(run_dir).iterdir():
        if p.is_file():
            mlflow.log_artifact(str(p), artifact_path="run")
