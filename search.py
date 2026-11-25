from __future__ import annotations
import argparse, math, json
from pathlib import Path
import yaml
import optuna
import mlflow
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from src.lr0.config import load_config, validate_config
from src.lr0.training import train_core, flatten_params
from src.lr0.tracking import setup_mlflow
from src.lr0.utils import git_info, utc_now

def main():
    p = argparse.ArgumentParser(description="Optuna TPE search для ЛР0")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--n-trials", type=int, default=40, help="Целевое общее число trial в study")
    p.add_argument("--allow-dirty", action="store_true", help="Только отладка")
    p.add_argument("--export-best", default=None, help="Сохранить лучшую конфигурацию в YAML")
    a = p.parse_args()
    git = git_info(require_clean=not a.allow_dirty)
    cfg = load_config(a.config)
    cfg["seed"] = 0
    cfg["split_seed"] = 0
    cfg["train"]["epochs"] = 30
    setup_mlflow()
    study = optuna.create_study(
        study_name="lr0_search", storage="sqlite:///optuna.db", load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    with mlflow.start_run(run_name="lr0_search", nested=False):
        mlflow.set_tags({"commit": str(git["commit"]), "series": "optuna_search"})
        def objective(trial):
            params = {
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "hidden": trial.suggest_int("hidden", 32, 512, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            }
            trial_cfg = json.loads(json.dumps(cfg))
            trial_cfg["model"].update({k: params[k] for k in ("n_layers", "hidden", "dropout")})
            trial_cfg["train"].update({k: params[k] for k in ("lr", "weight_decay", "batch_size")})
            validate_config(trial_cfg)
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.set_tag("trial_number", trial.number)
                last = float("nan")
                elapsed = 0.0
                def report(epoch, val_acc, row):
                    nonlocal last, elapsed
                    last = val_acc
                    elapsed += float(row["epoch_seconds"])
                    mlflow.log_metric("val_acc", val_acc, step=epoch)
                    mlflow.log_metric("epochs_completed", epoch, step=epoch)
                    mlflow.log_metric("elapsed_epoch_seconds", elapsed, step=epoch)
                    mlflow.log_metric("epoch_seconds", float(row["epoch_seconds"]), step=epoch)
                    trial.report(val_acc, step=epoch)
                    if trial.should_prune():
                        mlflow.set_tag("state", "PRUNED")
                        raise optuna.TrialPruned()
                try:
                    result = train_core(trial_cfg, on_epoch=report)
                    if math.isnan(last):
                        mlflow.set_tag("state", "FAIL")
                        return float("nan")
                    mlflow.set_tag("state", "COMPLETE")
                    mlflow.log_metric("final_val_acc", last)
                    return last
                except optuna.TrialPruned:
                    raise
                except Exception as exc:
                    mlflow.set_tag("state", "FAIL")
                    mlflow.set_tag("error", str(exc)[:450])
                    raise
        study.optimize(objective, n_trials=max(0, a.n_trials - len(study.trials)),
                       n_jobs=1, callbacks=[MaxTrialsCallback(a.n_trials, states=None)])
    states = {s.name: sum(t.state == s for t in study.trials) for s in TrialState}
    print("Trial states:", states)
    print("Trials total:", len(study.trials))
    if a.export_best:
        complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)]
        if not complete:
            raise SystemExit("Нет COMPLETE trial с конечным objective; best.yaml не создан.")
        best = max(complete, key=lambda t: t.value)
        out = json.loads(json.dumps(cfg))
        out["series"] = "final_best"
        out["model"].update({k: best.params[k] for k in ("n_layers", "hidden", "dropout")})
        out["train"].update({k: best.params[k] for k in ("lr", "weight_decay", "batch_size")})
        Path(a.export_best).write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print("Best trial", best.number, "value", best.value, "saved to", a.export_best)

if __name__ == "__main__": main()
