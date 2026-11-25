import copy, yaml, pytest
from src.lr0.config import load_config, validate_config

def test_base_config_valid():
    c = load_config("configs/base.yaml")
    validate_config(c)

def test_negative_lr_rejected():
    c = load_config("configs/base.yaml")
    c["train"]["lr"] = -1
    with pytest.raises(ValueError, match="train.lr"):
        validate_config(c)

def test_accuracy_known_example():
    import numpy as np
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 3])
    assert float(np.mean(y_true == y_pred)) == 0.75
