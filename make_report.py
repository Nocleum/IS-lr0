from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/generated")
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(Path("runs").glob("*")) if Path("runs").exists() else []:
        mp = d / "metrics.json"
        meta = d / "metadata.json"
        if mp.exists() and meta.exists():
            m, md = json.loads(mp.read_text()), json.loads(meta.read_text())
            rows.append({"run_id": md["run_id"], "series": md.get("series"), "seed": md.get("seed"),
                         "status": md.get("status"), **m})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "runs_summary.csv", index=False)
        for series, group in df.groupby("series"):
            g = group.dropna(subset=["val_acc"])
            if len(g):
                plt.figure()
                plt.scatter(g["seed"], g["val_acc"])
                plt.xlabel("seed"); plt.ylabel("validation accuracy")
                plt.title(f"Validation accuracy — {series}")
                plt.tight_layout(); plt.savefig(out / f"val_acc_{series}.png", dpi=160); plt.close()
    try:
        import optuna
        study = optuna.load_study(study_name="lr0_search", storage="sqlite:///optuna.db")
        tdf = study.trials_dataframe()
        tdf.to_csv(out / "optuna_trials.csv", index=False)
        complete = tdf[tdf["state"] == "COMPLETE"]
        if len(complete):
            vals = [t.value if t.state.name == "COMPLETE" else float("nan") for t in study.trials]
            best_so_far, current = [], float("-inf")
            for v in vals:
                if pd.notna(v): current = max(current, v)
                best_so_far.append(current if current != float("-inf") else float("nan"))
            plt.figure()
            plt.plot(range(len(vals)), vals, marker="o", linestyle="none", label="trial val_acc")
            plt.plot(range(len(vals)), best_so_far, label="best so far")
            plt.xlabel("trial number"); plt.ylabel("validation accuracy")
            plt.legend(); plt.tight_layout(); plt.savefig(out / "optuna_trials.png", dpi=160); plt.close()
            try:
                import optuna.visualization.matplotlib as ovm
                ax = ovm.plot_param_importances(study)
                ax.figure.tight_layout(); ax.figure.savefig(out / "optuna_importance.png", dpi=160)
                plt.close(ax.figure)
            except Exception as e:
                (out / "importance_unavailable.txt").write_text(str(e), encoding="utf-8")
        states = {s.name: sum(t.state == s for t in study.trials) for s in optuna.trial.TrialState}
        cost = {"trial_states": states, "planned_epochs": len(study.trials) * 30}
        try:
            import mlflow
            mlf = mlflow.search_runs(experiment_names=["lr0"], output_format="pandas")
            if not mlf.empty and "tags.trial_number" in mlf.columns:
                trial_runs = mlf[mlf["tags.trial_number"].notna()].copy()
                epoch_col = "metrics.epochs_completed"
                elapsed_col = "metrics.elapsed_epoch_seconds"
                state_col = "tags.state"
                if epoch_col in trial_runs.columns:
                    trial_runs[epoch_col] = pd.to_numeric(trial_runs[epoch_col], errors="coerce").fillna(0)
                    cost["epochs_actually_run"] = int(trial_runs[epoch_col].sum())
                    cost["epochs_saved"] = int(max(0, cost["planned_epochs"] - cost["epochs_actually_run"]))
                if elapsed_col in trial_runs.columns and state_col in trial_runs.columns:
                    completed = trial_runs[trial_runs[state_col] == "COMPLETE"]
                    rates = completed[elapsed_col] / completed[epoch_col].replace(0, pd.NA)
                    rates = pd.to_numeric(rates, errors="coerce").dropna()
                    if len(rates):
                        sec_per_epoch = float(rates.median())
                        pruned = trial_runs[trial_runs[state_col] == "PRUNED"]
                        cost["estimated_pruner_saved_seconds"] = round(float(sum(max(0, 30 - n) for n in pruned[epoch_col]) * sec_per_epoch), 3)
                        cost["median_seconds_per_epoch_completed_trials"] = round(sec_per_epoch, 5)
        except Exception as e:
            cost["mlflow_cost_extraction_error"] = str(e)
        (out / "optuna_states.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    except Exception as e:
        (out / "optuna_not_available.txt").write_text(str(e), encoding="utf-8")
    print(f"Отчётные файлы сохранены в {out}")


if __name__ == "__main__": main()
