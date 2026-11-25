from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from .utils import sha256_file, json_dump

SPLIT_PATH = Path("data/split.json")
MANIFEST_PATH = Path("data/manifest.json")

def prepare_split(split_seed=0):
    X, y = load_digits(return_X_y=True)
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=split_seed)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=split_seed)
    payload = {
        "split_seed": int(split_seed),
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json_dump(payload, SPLIT_PATH)
    import sklearn
    manifest = {
        "datasets": [{
            "name": "sklearn_load_digits",
            "source": "scikit-learn.datasets.load_digits",
            "source_package": "scikit-learn",
            "source_version": sklearn.__version__,
            "selection_rule": "Stratified split: test=20% first; val=25% of remaining; random_state=split_seed=0",
            "original_records": int(len(y)),
            "records": {"train": len(train), "val": len(val), "test": len(test)},
            "sha256": {"split.json": sha256_file(SPLIT_PATH)},
            "feature_shape": [8, 8],
            "feature_range": [0, 16],
            "classes": list(range(10)),
        }]
    }
    json_dump(manifest, MANIFEST_PATH)
    validate_split(SPLIT_PATH)
    return payload

def load_split(path=SPLIT_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} не найден. Сначала выполните python prepare_data.py")
    return json.loads(Path(path).read_text(encoding="utf-8"))

def validate_split(path=SPLIT_PATH):
    X, y = load_digits(return_X_y=True)
    split = load_split(path)
    parts = {k: np.asarray(split[k], dtype=int) for k in ("train", "val", "test")}
    assert [len(parts[k]) for k in ("train", "val", "test")] == [1077, 360, 360]
    all_idx = np.concatenate(list(parts.values()))
    assert len(np.unique(all_idx)) == len(y) == 1797
    assert set(all_idx.tolist()) == set(range(len(y)))
    for name, ids in parts.items():
        assert len(np.unique(y[ids])) == 10, f"В {name} отсутствует один из классов"
    return True

class DigitsDataset(Dataset):
    def __init__(self, indices):
        X, y = load_digits(return_X_y=True)
        self.X = (X[np.asarray(indices)] / 16.0).astype(np.float32)
        self.y = y[np.asarray(indices)].astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)
