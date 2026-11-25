from __future__ import annotations
import hashlib, json, os, random, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_info(require_clean: bool = True):
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if require_clean:
            raise RuntimeError("Git-репозиторий/коммит не найден. Инициализируйте Git и закоммитьте проект.")
        return {"commit": None, "dirty": None}
    if require_clean and dirty:
        raise RuntimeError("Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty)
    return {"commit": commit, "dirty": bool(dirty)}

def seed_everything(seed: int, num_threads: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)

def json_dump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

def package_versions():
    import sklearn, optuna, mlflow
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "optuna": optuna.__version__,
        "mlflow": mlflow.__version__,
        "numpy": np.__version__,
    }
