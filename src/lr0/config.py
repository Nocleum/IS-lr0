from __future__ import annotations
from pathlib import Path
import yaml

REQUIRED_TOP = {"experiment", "split_seed", "seed", "model", "train", "runtime"}

def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Конфигурация должна быть YAML-объектом.")
    missing = REQUIRED_TOP - set(cfg)
    if missing:
        raise ValueError(f"В конфигурации отсутствуют поля: {sorted(missing)}")
    cfg.setdefault("series", "manual")
    cfg["model"] = dict(cfg["model"])
    cfg["train"] = dict(cfg["train"])
    cfg["runtime"] = dict(cfg["runtime"])
    validate_config(cfg)
    return cfg

def validate_config(c):
    m, t, r = c["model"], c["train"], c["runtime"]
    for key in ("n_layers", "hidden", "dropout"):
        if key not in m: raise ValueError(f"Не задан model.{key}")
    for key in ("lr", "weight_decay", "batch_size", "epochs"):
        if key not in t: raise ValueError(f"Не задан train.{key}")
    if not isinstance(m["n_layers"], int) or not 1 <= m["n_layers"] <= 3:
        raise ValueError("model.n_layers должен быть целым числом от 1 до 3.")
    if not isinstance(m["hidden"], int) or m["hidden"] < 1:
        raise ValueError("model.hidden должен быть положительным целым числом.")
    if not 0 <= float(m["dropout"]) <= 0.5:
        raise ValueError("model.dropout должен быть в диапазоне [0, 0.5].")
    if not 0 < float(t["lr"]) <= 0.1:
        raise ValueError("train.lr должен быть в диапазоне (0, 0.1].")
    if float(t["weight_decay"]) < 0:
        raise ValueError("train.weight_decay не может быть отрицательным.")
    if int(t["batch_size"]) not in (32, 64, 128):
        raise ValueError("train.batch_size должен быть одним из [32, 64, 128].")
    if int(t["epochs"]) != 30:
        raise ValueError("По ТЗ число эпох фиксировано и должно быть равно 30.")
    if int(r.get("num_threads", 1)) < 1:
        raise ValueError("runtime.num_threads должен быть >= 1.")
    if r.get("device", "cpu") != "cpu" and not str(r["device"]).startswith("cuda"):
        raise ValueError("Поддерживаются device=cpu или cuda[:N].")
