from __future__ import annotations
import csv, json, time, traceback, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from .data import load_split, DigitsDataset
from .model import MLP
from .utils import seed_everything, utc_now, sha256_file, package_versions, json_dump, git_info

def _predict(model, loader, device):
    model.eval()
    ys, preds, losses = [], [], []
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            losses.extend(criterion(logits, yb).cpu().numpy().tolist())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(yb.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(preds), float(np.mean(losses))

def train_core(cfg, on_epoch=None):
    """Обучение без создания run-каталога; используется обычным запуском и Optuna."""
    split = load_split()
    seed = int(cfg["seed"])
    threads = int(cfg["runtime"].get("num_threads", 1))
    seed_everything(seed, threads)
    device = torch.device(cfg["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен CUDA device, но CUDA недоступна.")
    train_ds, val_ds = DigitsDataset(split["train"]), DigitsDataset(split["val"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                              generator=generator, num_workers=0)
    train_eval_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    model = MLP(**cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]),
                                 weight_decay=float(cfg["train"]["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        started = time.perf_counter()
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_n += len(yb)
        train_loss = total_loss / total_n
        _, _, train_eval_loss = _predict(model, train_eval_loader, device)
        _, _, val_loss = _predict(model, val_loader, device)
        y_val, pred_val, _ = _predict(model, val_loader, device)
        val_acc = float(np.mean(y_val == pred_val))
        epoch_seconds = time.perf_counter() - started
        row = {"epoch": epoch, "train_loss": train_eval_loss, "val_loss": val_loss,
               "val_acc": val_acc, "epoch_seconds": epoch_seconds}
        history.append(row)
        if on_epoch is not None:
            on_epoch(epoch, val_acc, row)
    y_val, pred_val, val_loss = _predict(model, val_loader, device)
    result = {
        "history": history,
        "val": {"y_true": y_val, "y_pred": pred_val, "loss": val_loss},
        "model": model,
    }
    return result

def _write_predictions(path, split_name, indices, y_true, y_pred):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["split", "index", "y_true", "y_pred"])
        for idx, yt, yp in zip(indices, y_true, y_pred):
            w.writerow([split_name, int(idx), int(yt), int(yp)])

def run_experiment(config_path, seed=None, eval_test=False, allow_dirty=False, series=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if seed is not None: cfg["seed"] = int(seed)
    if series is not None: cfg["series"] = series
    from .config import validate_config
    validate_config(cfg)
    git = git_info(require_clean=not allow_dirty)
    split = load_split()
    split_hash = sha256_file("data/split.json")
    run_id = uuid.uuid4().hex[:12]
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id, "experiment": cfg["experiment"], "series": cfg.get("series", "manual"),
        "status": "running", "started_at": utc_now(), "ended_at": None,
        "commit": git["commit"], "dirty_worktree_allowed": bool(allow_dirty),
        "versions": package_versions(), "device": cfg["runtime"].get("device", "cpu"),
        "num_threads": int(cfg["runtime"].get("num_threads", 1)), "seed": int(cfg["seed"]),
        "split_seed": int(cfg["split_seed"]), "split_sha256": split_hash,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    json_dump(metadata, run_dir / "metadata.json")
    from .tracking import setup_mlflow, log_run_artifacts
    setup_mlflow()
    try:
        with mlflow_start_run(cfg, metadata) as _:
            result = train_core(cfg, on_epoch=lambda e, a, row: __import__("mlflow").log_metric("val_acc", a, step=e))
            pd.DataFrame(result["history"]).to_csv(run_dir / "history.csv", index=False)
            with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["split", "index", "y_true", "y_pred"])
                for idx, yt, yp in zip(split["val"], result["val"]["y_true"], result["val"]["y_pred"]):
                    w.writerow(["val", int(idx), int(yt), int(yp)])
            if eval_test:
                device = torch.device(cfg["runtime"].get("device", "cpu"))
                test_ds = DigitsDataset(split["test"])
                test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
                yt, yp, _ = _predict(result["model"], test_loader, device)
                with open(run_dir / "predictions.csv", "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for idx, a, b in zip(split["test"], yt, yp):
                        w.writerow(["test", int(idx), int(a), int(b)])
            pred = pd.read_csv(run_dir / "predictions.csv")
            val = pred[pred.split == "val"]
            hist = pd.read_csv(run_dir / "history.csv")
            metrics = {
                "val_acc": float((val.y_true == val.y_pred).mean()),
                "val_loss": float(hist.iloc[-1].val_loss),
                "training_seconds": float(hist.epoch_seconds.sum()),
                "epochs_completed": int(len(hist)),
            }
            if eval_test:
                test = pred[pred.split == "test"]
                metrics["test_acc"] = float((test.y_true == test.y_pred).mean())
            json_dump(metrics, run_dir / "metrics.json")
            metadata["status"] = "completed"
            metadata["ended_at"] = utc_now()
            json_dump(metadata, run_dir / "metadata.json")
            import mlflow
            mlflow.log_params(flatten_params(cfg))
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"commit": str(git["commit"]), "series": str(cfg.get("series", "manual")),
                             "seed": str(cfg["seed"]), "run_id": run_id})
            log_run_artifacts(run_dir)
        return run_dir
    except Exception:
        metadata["status"] = "failed"
        metadata["ended_at"] = utc_now()
        json_dump(metadata, run_dir / "metadata.json")
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

def flatten_params(cfg):
    out = {"seed": cfg["seed"], "split_seed": cfg["split_seed"]}
    for section in ("model", "train", "runtime"):
        for k, v in cfg[section].items():
            out[f"{section}.{k}"] = v
    return out

class mlflow_start_run:
    def __init__(self, cfg, metadata): self.cfg, self.metadata = cfg, metadata
    def __enter__(self):
        import mlflow
        self.ctx = mlflow.start_run(run_name=f"{self.cfg.get('series','manual')}-{self.metadata['run_id']}")
        self.ctx.__enter__()
        mlflow.log_params(flatten_params(self.cfg))
        return self
    def __exit__(self, *args): return self.ctx.__exit__(*args)
