import copy
from pathlib import Path
import yaml
from src.lr0.data import prepare_split
from src.lr0.training import train_core


def test_same_seed_repeats_and_different_seed_changes():
    if not Path('data/split.json').exists():
        prepare_split(0)
    cfg = yaml.safe_load(Path('configs/base.yaml').read_text(encoding='utf-8'))
    # Короткий тест механизма детерминизма; полноценные 30 эпох выполняются в опытах.
    cfg['train']['epochs'] = 3
    a = train_core(copy.deepcopy(cfg))
    b = train_core(copy.deepcopy(cfg))
    # Wall-clock epoch_seconds закономерно различается между физическими запусками.
    # Сравниваем детерминированные поля истории, а время проверяем отдельно как измерение.
    stable = lambda rows: [{k: v for k, v in row.items() if k != 'epoch_seconds'} for row in rows]
    assert stable(a['history']) == stable(b['history'])
    assert (a['val']['y_pred'] == b['val']['y_pred']).all()
    cfg_other = copy.deepcopy(cfg)
    cfg_other['seed'] = 1
    c = train_core(cfg_other)
    assert stable(a['history']) != stable(c['history']) or not (a['val']['y_pred'] == c['val']['y_pred']).all()
