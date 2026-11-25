import argparse, subprocess, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--series", required=True)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()
    for seed in a.seeds:
        cmd = [sys.executable, "train.py", "--config", a.config, "--seed", str(seed), "--series", a.series]
        if a.eval_test: cmd.append("--eval-test")
        if a.allow_dirty: cmd.append("--allow-dirty")
        print(">", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

if __name__ == "__main__": main()
