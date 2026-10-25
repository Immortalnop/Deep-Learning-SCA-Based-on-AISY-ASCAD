# Phase 0: compare AISY's MLP and our CNN under one protocol, over several seeds.
#
# Process: generate seeds -> for each seed, train every model with the same
# protocol -> read guessing entropy (GE) from the AISY database -> summary table.
#
# Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.phase0_compare
#     ..\aisy_env\Scripts\python.exe -m scripts.phase0_compare --n-seeds 5 --epochs 50 --batch-size 128
#     ..\aisy_env\Scripts\python.exe -m scripts.phase0_compare --seeds 616134 4242
#
# Output in resources/experiments/phase0_<timestamp>/:
#     runs.csv    one row per (model, seed) run
#     summary.md  mean ± std per model
#     log.txt     everything printed during the run
# GE curves for each run are in the AISY webapp under the printed analysis ids.
# One row per run is added to resources/experiments/index.csv.
import argparse
import csv
import json
import os
import random
import sqlite3
import statistics
import traceback
from datetime import datetime

import aisy_sca
import tensorflow as tf
from custom.custom_datasets.datasets import datasets_dict
from custom.custom_models.neural_networks import cnn_sca_light, mlp
from scripts.experiment_log import log_to
from scripts.experiment_output import configure_aisy, finish_run, start_run
from scripts.experiment_seed import SEED_RANGE, set_seed

MODELS = {
    "mlp": mlp,             # AISY basic MLP: 4 x 200 selu
    "cnn": cnn_sca_light,   # group 30 CNN
}

# Same for every model. Early stopping keeps the epoch with the lowest validation GE;
# it needs a validation set, so AISY splits the 10,000 attack traces into 5,000 + 5,000.
PROTOCOL = {
    "dataset": "ASCAD.h5",
    "leakage_model": "HW",
    "byte": 2,
    "profiling_traces": 50000,
    "attack_traces": 10000,
    "key_rank_executions": 100,
    "key_rank_attack_traces": 1000,
    "key_rank_report_interval": 10,
}

EARLY_STOPPING = {
    "metrics": {
        "guessing_entropy": {
            "direction": "min",
            "class": "custom.custom_metrics.guessing_entropy",
            "parameters": []
        }
    }
}

DATABASE_FOLDER = "databases/"
DATABASE_NAME = "database_ascad.sqlite"
GE_THRESHOLD = 2  # AISY counts a key as found when GE < 2 (rank 1 = best)
LABEL_FINAL = "Attack Set"                          # weights after the last epoch
LABEL_BEST = "ES Attack Set guessing_entropy"       # weights from the best epoch


def run_one(model_name, seed, epochs, batch_size, folder):
    tf.keras.backend.clear_session()
    set_seed(seed)

    aisy = aisy_sca.Aisy()
    configure_aisy(aisy, folder, DATABASE_NAME)
    aisy.set_dataset(datasets_dict[PROTOCOL["dataset"]])
    aisy.set_number_of_profiling_traces(PROTOCOL["profiling_traces"])
    aisy.set_number_of_attack_traces(PROTOCOL["attack_traces"])
    aisy.set_aes_leakage_model(leakage_model=PROTOCOL["leakage_model"], byte=PROTOCOL["byte"])
    aisy.set_batch_size(batch_size)
    aisy.set_epochs(epochs)
    aisy.add_neural_network(MODELS[model_name], name=model_name, seed=seed)
    aisy.run(
        key_rank_executions=PROTOCOL["key_rank_executions"],
        key_rank_attack_traces=PROTOCOL["key_rank_attack_traces"],
        key_rank_report_interval=PROTOCOL["key_rank_report_interval"],
        early_stopping=EARLY_STOPPING,
    )
    return aisy.settings["analysis_id"]


def read_curve(connection, table, analysis_id, label):
    row = connection.execute(f'select "values", report_interval from {table} where analysis_id = ? and label = ?',
                             (analysis_id, label)).fetchone()
    if row is None:
        return None, None
    values = json.loads(row[0])
    if isinstance(values, str):  # AISY stores the JSON twice encoded
        values = json.loads(values)
    return [values[str(i)] for i in range(len(values))], row[1]


def traces_to_threshold(ge, interval):
    # Number of attack traces after which GE stays below the threshold; None if never.
    above = [i for i, value in enumerate(ge) if value >= GE_THRESHOLD]
    if not above:
        return interval
    if above[-1] == len(ge) - 1:
        return None
    return (above[-1] + 2) * interval


def results_for(analysis_id):
    connection = sqlite3.connect(DATABASE_FOLDER + DATABASE_NAME)
    result = {}
    for key, label in (("final", LABEL_FINAL), ("best", LABEL_BEST)):
        ge, interval = read_curve(connection, "guessing_entropy", analysis_id, label)
        sr, _ = read_curve(connection, "success_rate", analysis_id, label)
        if ge is None:
            continue
        result[f"{key}_ge"] = round(ge[-1], 2)
        result[f"{key}_sr"] = round(sr[-1], 2) if sr else None
        result[f"{key}_traces_ge_lt_2"] = traces_to_threshold(ge, interval)
    connection.close()
    return result


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def write_summary(path, rows, args, seeds):
    lines = [
        f"# Phase 0: MLP vs CNN ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}. Epochs: {args.epochs} (best epoch kept by validation GE). "
        f"Batch size: {args.batch_size}.",
        f"Data: {PROTOCOL['dataset']}, byte {PROTOCOL['byte']}, {PROTOCOL['leakage_model']} leakage model, "
        f"{PROTOCOL['profiling_traces']} profiling traces, attack set split into 5,000 validation + 5,000 attack.",
        f"GE: mean of {PROTOCOL['key_rank_executions']} key ranks on {PROTOCOL['key_rank_attack_traces']} attack traces. "
        f"'Traces to GE < 2' averages only runs that reach it.",
        "",
        "| Model | Runs OK | Reached GE < 2 (best epoch) | Traces to GE < 2 (best epoch) | Final GE (best epoch) "
        "| Final SR (best epoch) | Final GE (last epoch) |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_name in args.models:
        ok = [r for r in rows if r["model"] == model_name and r["status"] == "ok"]
        reached = [r for r in ok if r.get("best_traces_ge_lt_2") is not None]
        lines.append(
            f"| {model_name} | {len(ok)}/{len(seeds)} | {len(reached)}/{len(ok)} "
            f"| {mean_std([r.get('best_traces_ge_lt_2') for r in ok])} "
            f"| {mean_std([r.get('best_ge') for r in ok])} | {mean_std([r.get('best_sr') for r in ok])} "
            f"| {mean_std([r.get('final_ge') for r in ok])} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 0: MLP vs CNN under one protocol.")
    parser.add_argument("--seeds", type=int, nargs="+", help="seeds to use (default: generate --n-seeds)")
    parser.add_argument("--n-seeds", type=int, default=3, help="number of seeds to generate (default 3)")
    parser.add_argument("--epochs", type=int, default=50, help="epochs per run (default 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for every model (default 128)")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [random.randrange(SEED_RANGE) for _ in range(args.n_seeds)]
    folder = start_run("phase0")

    # Everything printed from here on also goes to <folder>/log.txt
    with log_to(os.path.join(folder, "log.txt")):
        rows = run_all(args, seeds, folder)
        ok = [r for r in rows if r["status"] == "ok"]
        finish_run(folder, "phase0", " ".join(str(s) for s in seeds), [r["analysis_id"] for r in ok],
                   status="ok" if len(ok) == len(rows) else f"{len(ok)}/{len(rows)} ok")


def run_all(args, seeds, folder):
    print(f"Phase 0: models {args.models}, seeds {seeds}, epochs {args.epochs}, batch size {args.batch_size}")
    print(f"Protocol: {PROTOCOL}")

    columns = ["model", "seed", "status", "analysis_id", "best_ge", "best_sr", "best_traces_ge_lt_2",
               "final_ge", "final_sr", "final_traces_ge_lt_2", "error"]
    rows = []
    with open(os.path.join(folder, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for seed in seeds:
            for model_name in args.models:
                print(f"\n===== {model_name}, seed {seed} =====")
                row = {"model": model_name, "seed": seed}
                try:
                    analysis_id = run_one(model_name, seed, args.epochs, args.batch_size, folder)
                    row.update(status="ok", analysis_id=analysis_id, **results_for(analysis_id))
                except Exception as e:
                    traceback.print_exc()
                    row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                rows.append(row)
                writer.writerow(row)
                f.flush()

    print("\n" + write_summary(os.path.join(folder, "summary.md"), rows, args, seeds))
    print(f"\nResults: {folder}")
    return rows


if __name__ == "__main__":
    main()
