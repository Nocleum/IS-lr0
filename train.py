import argparse
from src.lr0.training import run_experiment

def main():
    p = argparse.ArgumentParser(description="Один воспроизводимый запуск ЛР0")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--allow-dirty", action="store_true", help="Только отладка; не включать в итоговые серии")
    p.add_argument("--series", help="Имя заранее определённой серии")
    a = p.parse_args()
    path = run_experiment(a.config, seed=a.seed, eval_test=a.eval_test, allow_dirty=a.allow_dirty, series=a.series)
    print(path)

if __name__ == "__main__": main()
