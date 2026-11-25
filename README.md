# ЛР0 — воспроизводимый эксперимент: подбор гиперпараметров MLP

Каркас для ЛР0 по приложенным «Общим правилам» и ТЗ. Проект готовит данные, запускает отдельные эксперименты, сохраняет артефакты, ведёт MLflow и Optuna, проверяет воспроизводимость и строит отчётные графики.

**Экспериментальные серии намеренно не запускались.** После установки зависимостей нужно выполнить команды из раздела «Порядок работы».

## Требования

- Python 3.10+
- Git
- CPU достаточно; GPU не требуется
- Зависимости указаны в `requirements.txt` (диапазоны совместимости; после установки сохраните фактическое окружение командой `pip freeze > requirements-lock.txt` и закоммитьте lock-файл).

## Быстрый старт

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# Создать фиксированное разбиение и manifest.json
python prepare_data.py

# Проверить реализацию (это тесты каркаса, не экспериментальные серии)
pytest -q

# Зафиксировать исходный код перед итоговыми запусками
git init
git add .
git commit -m "Prepare LR0 experiment harness"
```

## Структура

```text
.
├── configs/
│   ├── base.yaml              # базовая конфигурация из ТЗ
│   ├── best.yaml              # сюда сохранить параметры лучшего Optuna trial
│   └── search_space.yaml       # пространство поиска (документация)
├── data/
│   ├── split.json             # генерируется prepare_data.py
│   └── manifest.json          # генерируется вместе с split.json
├── reports/
│   ├── forecast.md            # прогноз ДО экспериментов
│   └── report_template.md     # шаблон отчёта
├── runs/                      # создаётся запусками; не коммитить
├── mlartifacts/               # артефакты MLflow; не коммитить
├── train.py                   # один запуск
├── prepare_data.py            # разбиение и манифест
├── run_series.py              # заранее заданная серия запусков
├── search.py                  # Optuna study lr0_search
├── make_report.py             # таблицы и графики из сохранённых результатов
├── src/lr0/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   ├── tracking.py
│   └── utils.py
└── tests/
```

## Порядок работы по ТЗ

### 0. Подготовка и проверки

1. Выполнить `python prepare_data.py`.
2. Проверить `data/split.json` и `data/manifest.json`.
3. Выполнить `pytest -q`.
4. Создать Git-коммит. Итоговые запуски по умолчанию отказываются стартовать при незакоммиченном коде. Для локальной отладки можно явно указать `--allow-dirty`; такие запуски не следует включать в итоговые серии.

### 1. Эксперимент 1 — повторяемость и разброс

Сначала дважды повторить один seed:

```bash
python train.py --config configs/base.yaml --seed 0
python train.py --config configs/base.yaml --seed 0
```

Сравнить `history.csv` и `predictions.csv` двух каталогов `runs/`. Затем серия seed 1–5:

```bash
python run_series.py --config configs/base.yaml --seeds 1 2 3 4 5 --series base_seeds
```

Сводку из сохранённых запусков можно сформировать командой:

```bash
python make_report.py --output reports/generated
```

Для сравнения ошибок двух разных seed используйте `predictions.csv`: в каждом файле есть `split,index,y_true,y_pred`.

### 2. Эксперимент 2 — Optuna

```bash
python search.py --n-trials 40
```

Чтобы выполнить требуемое прерывание примерно после 10 trial, сначала запустите:

```bash
python search.py --n-trials 10
```

Затем продолжите:

```bash
python search.py --n-trials 40
```

Study сохраняется в `sqlite:///optuna.db`, `load_if_exists=True`, имя `lr0_search`. `n-trials` — целевое общее число trial, а не количество дополнительных trial. Поиск последовательный (`n_jobs=1`).

После завершения:

```bash
python search.py --export-best configs/best.yaml
# Сохранить выбранную конфигурацию отдельным коммитом до финальной серии
git add configs/best.yaml && git commit -m "Record best LR0 configuration"
python make_report.py --output reports/generated
```

Важно: как оговорено в ТЗ, продолжение прерванного study сохраняет нумерацию и результаты, но seed TPE не восстанавливает прежнюю последовательность предложений. Полностью повторяемым является поиск, выполненный без прерывания.

### 3. Эксперимент 3 — честное сравнение

После сохранения лучшей конфигурации:

```bash
python run_series.py --config configs/base.yaml --seeds 101 102 103 104 105 --series final_base --eval-test
python run_series.py --config configs/best.yaml --seeds 101 102 103 104 105 --series final_best --eval-test
python make_report.py --output reports/generated
```

Test используется только в этой финальной серии. Не меняйте конфигурации/код после просмотра test-результатов, иначе test перестаёт быть отложенным.

## Команды

### Один запуск

```bash
python train.py --config configs/base.yaml
python train.py --config configs/base.yaml --seed 7
python train.py --config configs/base.yaml --seed 7 --eval-test
```

CLI-параметр `--seed` переопределяет seed из YAML. `--eval-test` включает однократную оценку на test. Без флага test не читается функцией обучения и не попадает в `predictions.csv`.

### Серия

```bash
python run_series.py --config configs/base.yaml --seeds 1 2 3 4 5 --series base_seeds
```

Каждый указанный seed запускается отдельным процессом с указанной меткой серии. Неудачные запуски сохраняются; при продолжении прерванной серии запускайте только недостающие seed вручную, чтобы не перезаписать уже собранные повторы.

### MLflow

Tracking URI: `sqlite:///mlflow.db`. Эксперимент: `lr0`. Обычный запуск — отдельный MLflow run; Optuna создаёт родительский run и вложенные записи trial.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

В Colab/Kaggle можно анализировать журнал через `mlflow.search_runs()` либо использовать сохранённые артефакты.

## Формат каталога запуска

`runs/<run_id>/`:

- `config.yaml` — итоговая конфигурация;
- `metadata.json` — run_id, серия, статус, timestamps, commit, версии библиотек, seed, split hash, число потоков;
- `history.csv` — epoch, train_loss, val_loss, val_acc, epoch_seconds;
- `predictions.csv` — split, index, y_true, y_pred для val и, только с `--eval-test`, test;
- `metrics.json` — показатели, пересчитанные из CSV;
- `error.txt` — traceback при ошибке.

Метрики не берутся из промежуточных переменных модели: итоговая accuracy пересчитывается из `predictions.csv`, время и потери — из `history.csv`.

## Воспроизводимость и ограничения

- random, NumPy, PyTorch и DataLoader получают seed;
- `num_workers=0`, `num_threads` фиксируется;
- включён `torch.use_deterministic_algorithms(True)`;
- разбиение фиксировано отдельным `split_seed=0`;
- признаки делятся на 16; статистики по val/test не вычисляются;
- для CPU детерминизм ожидается в одинаковом окружении; значения `epoch_seconds` — реальные wall-clock замеры и могут различаться даже при одинаковом seed. При проверке повторяемости сравнивайте train/val losses, accuracy и predictions, а время анализируйте отдельно. Это техническое исключение из буквального требования побайтно сравнить `history.csv`: физическое время выполнения не является детерминированной величиной;
- версии библиотек, commit и SHA-256 split записываются в metadata;
- итоговый запуск требует чистого Git working tree.

## Что осталось сделать студенту

- заполнить `reports/forecast.md` и закоммитить прогноз до экспериментов;
- выполнить серии 1–3;
- проверить все артефакты и изучить неудачные/прерванные trial;
- перенести реальные результаты в `reports/report_template.md`;
- ответить на вопросы к защите собственными числами.

Никакие численные результаты в шаблоны не подставлены, поскольку серии не запускались.
