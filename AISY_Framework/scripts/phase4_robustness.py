# Phase 4: robustness to desynchronisation, under the Phase 0 protocol.
#
# Process: one study folder, filled in stages. A stage can be rerun on the same folder: finished runs are skipped,
# failed runs are retried.
#   baseline   Experiment A: every variant trained on desync 0, then attacked at desync 0 and at every --levels value
#              (the same trained model, traces normalised with the desync 0 profiling statistics)
#   direct     Experiment B: every variant trained and attacked on the same desync level (--levels)
#   report     tables, GE curves per desync level, summary.md
#   all        baseline, direct, report
#
# Variants (the CNN is the best Phase 3 search model on the validation set: hp and loss are read from --phase3):
#   mlp        AISY MLP, the model expected to collapse under desync
#   cnn        the Phase 3 CNN (Flatten before the dense layers)
#   cnn_aug    cnn trained on randomly shifted, noisy mini-batches (shift up to 100 samples when training on desync 0,
#              else up to the desync level; custom/custom_data_augmentation)
#   cnn_wide   cnn with doubled kernels (wider receptive field)
#   cnn_gap    cnn with global average pooling instead of Flatten
#
# Every run: AISY trains with early stopping on the validation GE area (custom/custom_metrics/ge_area.py), stops after
# --patience epochs without improvement (PlateauStop), and keeps the best epoch. Models are chosen on the validation set
# of the training desync level only. GE / SR use scripts/sca_protocol.py on the same attack subsets as Phases 2 and 3.
# A run whose predictions are not finite (diverged training) is recorded as failed, never as a result.
#
# Data: data/ASCAD.h5 (desync 0), data/ascad-variable-desync50.h5, data/ascad-variable-desync100.h5 (the ASCAD v1
# variable-key files; links in the ASCAD repository README). Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.phase4_robustness all
#     ..\aisy_env\Scripts\python.exe -m scripts.phase4_robustness all --seeds 1 --variants mlp cnn cnn_aug
#     ..\aisy_env\Scripts\python.exe -m scripts.phase4_robustness baseline --levels   (desync 0 only, no desync files)
#     ..\aisy_env\Scripts\python.exe -m scripts.phase4_robustness report --study resources/experiments/phase4_<timestamp>
#
# Output in resources/experiments/phase4_<timestamp>/:
#     runs.csv               one row per training run: variant, training desync, settings, analysis id, validation metrics
#     eval.csv               one row per (run, attack desync): attack metrics
#     predictions/<run>.npz  best-epoch log-probabilities per attack desync; plaintexts_d<level>.npz once per level
#     tensorboard/<run>/     loss and accuracy per epoch
#     results.csv, ge_desync.png, summary.md  (report)
#     log_<stage>_<timestamp>.txt
# One row per stage is added to resources/experiments/index.csv.
import argparse
import csv
import json
import os
import time
import traceback
from datetime import datetime

import h5py
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

import aisy_sca
import tensorflow as tf
from custom.custom_data_augmentation.data_augmentation import make_shift_augmentation
from custom.custom_datasets.datasets import datasets_dict
from custom.custom_metrics.ge_area import HISTORY_KEY, VALIDATION_SEED
from scripts.experiment_log import log_to
from scripts.experiment_output import EXPERIMENTS_FOLDER, configure_aisy, finish_run, resume_run, start_run
from scripts.experiment_seed import set_seed
from scripts.phase0_compare import mean_std
from scripts.phase3_study import EARLY_STOPPING, FIXED_BATCH_SIZE, build_model, markdown_table, number, write_csv
from scripts.sca_protocol import (GE_THRESHOLD, PROTOCOL, attack_subsets, ge_area, ge_sr, key_ranks, key_scores,
                                  traces_to_threshold)

STAGES = ["baseline", "direct", "report"]
VARIANTS = ["mlp", "cnn", "cnn_aug", "cnn_wide", "cnn_gap"]
DESYNC_FILES = {0: "ASCAD.h5", 50: "ascad-variable-desync50.h5", 100: "ascad-variable-desync100.h5"}
DATA_FOLDER = "data/"
WIDE_KERNEL_CAP = 101
NORMALISATION_TOLERANCE = 1e-4  # own z-score against AISY's, per sample

RUN_COLUMNS = ["run", "stage", "variant", "model", "hp", "train_desync", "loss", "seed", "batch_size", "status",
               "analysis_id", "parameters", "epochs_run", "best_epoch", "val_ge_area", "val_ge_final", "seconds", "error"]
EVAL_COLUMNS = ["run", "variant", "train_desync", "attack_desync", "seed", "att_ge_area", "att_ge_final",
                "att_median_rank", "att_sr_final", "att_traces_ge_lt_2"]


# ---------------------------------------------------------------------------------------------------------------
# Study folder and data
# ---------------------------------------------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def check_data(levels):
    missing = [DESYNC_FILES[level] for level in levels if not os.path.exists(DATA_FOLDER + DESYNC_FILES[level])]
    if missing:
        raise FileNotFoundError(f"missing in {DATA_FOLDER}: {', '.join(missing)} (ASCAD v1 variable-key files, "
                                f"download links in the ANSSI-FR/ASCAD README)")


class Study:
    def __init__(self, folder, args):
        self.folder, self.args = folder, args
        self.runs_file, self.eval_file = os.path.join(folder, "runs.csv"), os.path.join(folder, "eval.csv")
        self.predictions = os.path.join(folder, "predictions")
        os.makedirs(self.predictions, exist_ok=True)
        self.rows, self.evals = read_csv(self.runs_file), read_csv(self.eval_file)
        self.dataset = (None, None)  # (desync level, AISY dataset); one slot keeps memory low
        self.statistics, self.attack_cache = {}, {}
        self.new_analysis_ids = []

    def ok_rows(self, stage=None):
        return [r for r in self.rows if r["status"] == "ok" and (stage is None or r["stage"] == stage)]

    def done(self, run):
        return any(r["run"] == run for r in self.ok_rows())

    def save_row(self, row, evals):
        self.rows = [r for r in self.rows if r["run"] != row["run"]] + [row]
        write_rows(self.runs_file, RUN_COLUMNS, self.rows)
        if evals:
            self.evals = [e for e in self.evals if e["run"] != row["run"]] + evals
            write_rows(self.eval_file, EVAL_COLUMNS, self.evals)

    def run(self, config):
        # Train one model unless it already finished.
        if self.done(config["run"]):
            print(f"{config['run']}: already done, skipped")
            return
        print(f"\n===== {config['run']} ({config['variant']}, trained on desync {config['train_desync']}) =====")
        row = {k: config.get(k, "") for k in RUN_COLUMNS}
        row["hp"] = json.dumps(config["hp"], sort_keys=True) if config["hp"] else ""
        evals, start = [], time.time()
        try:
            result = train(config, self)
            evals = result.pop("evals")
            row.update(result)
            row["status"] = "ok"
            self.new_analysis_ids.append(row["analysis_id"])
        except Exception as e:  # one failed run must not stop the study
            traceback.print_exc()
            row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
        row["seconds"] = round(time.time() - start, 1)
        self.save_row(row, evals)
        shown = "; ".join(f"desync {e['attack_desync']}: GE {e['att_ge_final']}, traces to GE < 2 "
                          f"{e['att_traces_ge_lt_2'] or 'not reached'}" for e in evals)
        print(f"{config['run']}: {row['status']}, validation GE area {row.get('val_ge_area')}; {shown}; "
              f"{row['seconds']} s")

    def normalisation(self, level):
        # Per-sample mean and standard deviation of the profiling traces, as AISY's z-score (DatasetControls).
        if level not in self.statistics:
            info = datasets_dict[DESYNC_FILES[level]]
            first, end, n = info["first_sample"], info["first_sample"] + info["number_of_samples"], \
                PROTOCOL["profiling_traces"]
            with h5py.File(DATA_FOLDER + info["filename"], "r") as f:
                traces = f["Profiling_traces/traces"]
                chunks = [(a, min(a + 5000, n)) for a in range(0, n, 5000)]
                mean = sum(traces[a:b, first:end].astype(np.float64).sum(axis=0) for a, b in chunks) / n
                var = sum(((traces[a:b, first:end].astype(np.float64) - mean) ** 2).sum(axis=0) for a, b in chunks) / n
            self.statistics[level] = (mean, np.sqrt(var))
        return self.statistics[level]

    def attack(self, level):
        # Raw attack traces of a desync level in AISY's split: validation = first half, attack = second half.
        if level not in self.attack_cache:
            info = datasets_dict[DESYNC_FILES[level]]
            first, end, n = info["first_sample"], info["first_sample"] + info["number_of_samples"], \
                PROTOCOL["attack_traces"]
            with h5py.File(DATA_FOLDER + info["filename"], "r") as f:
                x = f["Attack_traces/traces"][:n, first:end].astype(np.float64)
                plaintext = f["Attack_traces/metadata"]["plaintext"][:n]
                keys = np.unique(f["Attack_traces/metadata"]["key"][:n, PROTOCOL["byte"]])
            expected = bytearray.fromhex(info["key"])[PROTOCOL["byte"]]
            if len(keys) != 1 or keys[0] != expected:
                raise ValueError(f"attack key byte {PROTOCOL['byte']} in {info['filename']} is {keys[:5]}, "
                                 f"datasets.py gives 0x{expected:02X}")
            half = n // 2
            self.attack_cache = {level: {"x": x, "plain_val": plaintext[:half], "plain_att": plaintext[half:],
                                         "key": int(expected)}}  # one level in memory at a time
        return self.attack_cache[level]

    def plaintexts(self, level):
        data = np.load(os.path.join(self.predictions, f"plaintexts_d{level}.npz"))
        return data["validation"], data["attack"], int(data["key"])


# ---------------------------------------------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------------------------------------------
def log_probabilities(model, x):
    if len(model.input_shape) == 3:
        x = x[:, :, None]
    return np.log(model.predict(x, batch_size=1000, verbose=0) + 1e-36).astype(np.float32)


def evaluate(study, level, log_val, log_att):
    plain_val, plain_att, key = study.plaintexts(level)
    byte, lm = PROTOCOL["byte"], PROTOCOL["leakage_model"]
    val = key_scores(log_val, plain_val, byte, lm)
    att = key_scores(log_att, plain_att, byte, lm)
    val_ge, _ = ge_sr(key_ranks(val, key, attack_subsets(len(val), VALIDATION_SEED)))
    att_ranks = key_ranks(att, key, attack_subsets(len(att), study.args.subset_seed))
    att_ge, att_sr = ge_sr(att_ranks)
    return {"val_ge_area": round(ge_area(val_ge), 2), "val_ge_final": round(float(val_ge[-1]), 2),
            "att_ge_area": round(ge_area(att_ge), 2), "att_ge_final": round(float(att_ge[-1]), 2),
            "att_median_rank": float(np.median(att_ranks[:, -1])), "att_sr_final": round(float(att_sr[-1]), 2),
            "att_traces_ge_lt_2": traces_to_threshold(list(att_ge), PROTOCOL["key_rank_report_interval"])}


def train(config, study):
    args, level = study.args, config["train_desync"]
    tf.keras.backend.clear_session()
    set_seed(config["seed"])
    aisy = aisy_sca.Aisy()
    configure_aisy(aisy, study.folder)
    # Reuse the loaded, normalised and split dataset when the next run trains on the same desync level.
    aisy.set_dataset(datasets_dict[DESYNC_FILES[level]], study.dataset[1] if study.dataset[0] == level else None)
    aisy.set_number_of_profiling_traces(PROTOCOL["profiling_traces"])
    aisy.set_number_of_attack_traces(PROTOCOL["attack_traces"])
    aisy.set_aes_leakage_model(leakage_model=PROTOCOL["leakage_model"], byte=PROTOCOL["byte"])
    aisy.set_batch_size(config["batch_size"])
    aisy.set_epochs(args.epochs)
    model = build_model(config, aisy.settings["classes"], aisy.settings["number_of_samples"])
    parameters = model.count_params()
    aisy.add_neural_network(model, name=config["run"], seed=config["seed"])

    augmentation = None
    if config["variant"] == "cnn_aug":
        shift = args.aug_shift if level == 0 else level
        print(f"Augmentation: random shifts up to {shift} samples, Gaussian noise {args.aug_noise}")
        augmentation = [make_shift_augmentation(shift, args.aug_noise), PROTOCOL["profiling_traces"] // config["batch_size"]]
    aisy.run(
        key_rank_executions=PROTOCOL["key_rank_executions"],
        key_rank_attack_traces=PROTOCOL["key_rank_attack_traces"],
        key_rank_report_interval=PROTOCOL["key_rank_report_interval"],
        early_stopping=EARLY_STOPPING,
        data_augmentation=augmentation,
        callbacks=[
            {"class": "custom.custom_callbacks.callbacks.PlateauStop", "name": "plateau_stop",
             "parameters": {"history_key": HISTORY_KEY, "patience": args.patience}},
            {"class": "custom.custom_callbacks.callbacks.TensorBoardLog", "name": "tensorboard",
             "parameters": {"log_dir": os.path.join(study.folder, "tensorboard", config["run"])}},
        ],
    )
    study.dataset = (level, aisy.dataset)

    # AISY loads the best-epoch weights to compute its early-stopping metrics and keeps them in the model.
    trained, dataset = aisy.models["0"]["model"], aisy.dataset
    mean, std = study.normalisation(level)
    levels = [0] + args.levels if level == 0 else [level]
    arrays, evals, own = {}, [], None
    for attack_level in levels:
        data = study.attack(attack_level)
        half = len(data["x"]) // 2
        x = ((data["x"] - mean) / std).astype(np.float32)
        if attack_level == level:  # own z-score against AISY's: the transfer to other levels relies on it
            difference = float(np.abs(x[half:] - dataset.x_attack.reshape(half, -1)).max())
            if difference > NORMALISATION_TOLERANCE:
                raise RuntimeError(f"own normalisation differs from AISY's by {difference:.2g}")
        log_val, log_att = log_probabilities(trained, x[:half]), log_probabilities(trained, x[half:])
        if not (np.isfinite(log_val).all() and np.isfinite(log_att).all()):
            raise RuntimeError(f"non-finite predictions at desync {attack_level} (diverged training)")
        plaintexts = os.path.join(study.predictions, f"plaintexts_d{attack_level}.npz")
        if not os.path.exists(plaintexts):
            np.savez(plaintexts, validation=data["plain_val"], attack=data["plain_att"], key=data["key"])
        arrays[f"validation_d{attack_level}"], arrays[f"attack_d{attack_level}"] = log_val, log_att
        metrics = evaluate(study, attack_level, log_val, log_att)
        evals.append({"run": config["run"], "variant": config["variant"], "train_desync": level,
                      "attack_desync": attack_level, "seed": config["seed"],
                      **{k: v for k, v in metrics.items() if k.startswith("att_")}})
        if attack_level == level:
            own = metrics
    np.savez(os.path.join(study.predictions, f"{config['run']}.npz"), **arrays)
    if config.get("weights_file"):  # Phase 5 needs the trained model; AISY deletes its own best-epoch files
        trained.save_weights(config["weights_file"])

    history = aisy.settings[HISTORY_KEY]
    return {"analysis_id": aisy.settings["analysis_id"], "parameters": parameters, "epochs_run": len(history),
            "best_epoch": int(np.argmin(history)) + 1, "val_ge_area": own["val_ge_area"],
            "val_ge_final": own["val_ge_final"], "evals": evals}


# ---------------------------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------------------------
def best_phase3_cnn(args):
    # Best search-family model of a Phase 3 study on the validation GE area (search and loss stages only; a value
    # below 1 cannot be a real GE).
    folder = args.phase3
    if not folder:
        studies = sorted(f for f in os.listdir(EXPERIMENTS_FOLDER)
                         if f.startswith("phase3_") and os.path.exists(os.path.join(EXPERIMENTS_FOLDER, f, "runs.csv")))
        if not studies:
            raise RuntimeError("no Phase 3 study found: run phase3_study first or pass --phase3 <folder>")
        folder = f"{EXPERIMENTS_FOLDER}/{studies[-1]}"
    rows = [r for r in read_csv(os.path.join(folder, "runs.csv")) if r["status"] == "ok"
            and r["stage"] in ("search", "loss") and float(r["val_ge_area"]) >= 1]
    if not rows:
        raise RuntimeError(f"no finished search runs in {folder}")
    best = min(rows, key=lambda r: float(r["val_ge_area"]))
    print(f"Phase 3 CNN: {best['run']} from {folder} (validation GE area {best['val_ge_area']}, loss {best['loss']}, "
          f"hp {best['hp']})")
    return {"hp": json.loads(best["hp"]), "loss": best["loss"], "source": f"{best['run']} ({folder})"}


def make_config(variant, level, seed, base, stage):
    hp, loss = dict(base["hp"]), base["loss"]
    model, batch_size = "cnn_search", hp["batch_size"]
    if variant == "mlp":
        model, hp, loss, batch_size = "mlp", None, "cce", FIXED_BATCH_SIZE
    elif variant == "cnn_wide":
        hp["kernel"] = min(2 * hp["kernel"], WIDE_KERNEL_CAP)
    elif variant == "cnn_gap":
        hp["gap"] = True
    return {"run": f"d{level}_{variant}_s{seed}", "stage": stage, "variant": variant, "model": model, "hp": hp,
            "train_desync": level, "loss": loss, "seed": seed, "batch_size": batch_size}


def stage_baseline(study):
    check_data(study.args.levels)
    base = best_phase3_cnn(study.args)
    for variant in study.args.variants:
        for seed in study.args.seeds:
            study.run(make_config(variant, 0, seed, base, "baseline"))


def stage_direct(study):
    if not study.args.levels:
        print("No --levels: nothing to train on desync 50 / 100")
        return
    check_data(study.args.levels)
    base = best_phase3_cnn(study.args)
    for level in study.args.levels:  # one desync level at a time, so the loaded dataset is reused
        for variant in study.args.variants:
            for seed in study.args.seeds:
                study.run(make_config(variant, level, seed, base, "direct"))


# ---------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------
def group_evals(study):
    # (variant, training desync, attack desync) -> eval rows of the finished runs, in study order
    groups = {}
    for e in study.evals:
        if study.done(e["run"]):
            groups.setdefault((e["variant"], int(e["train_desync"]), int(e["attack_desync"])), []).append(e)
    return groups


def summarise(study):
    parameters = {r["variant"]: r["parameters"] for r in study.ok_rows()}
    stats = []
    for (variant, train, attack), runs in group_evals(study).items():
        traces = [number(e["att_traces_ge_lt_2"]) for e in runs]
        reached = [t for t in traces if t is not None]
        stats.append({"variant": variant, "train_desync": train, "attack_desync": attack, "seeds": len(runs),
                      "reached_ge_lt_2": f"{len(reached)}/{len(runs)}", "n_reached": len(reached),
                      "traces_ge_lt_2": mean_std(traces),
                      "mean_traces": float(np.mean(reached)) if reached else float("inf"),
                      "att_ge_final": mean_std([number(e["att_ge_final"]) for e in runs]),
                      "mean_ge_final": float(np.mean([number(e["att_ge_final"]) for e in runs])),
                      "att_ge_area": mean_std([number(e["att_ge_area"]) for e in runs]),
                      "parameters": parameters.get(variant, "")})
    return stats


def run_ge(study, run, level):
    attack = np.load(os.path.join(study.predictions, f"{run}.npz"))[f"attack_d{level}"]
    _, plain_att, key = study.plaintexts(level)
    scores = key_scores(attack, plain_att, PROTOCOL["byte"], PROTOCOL["leakage_model"])
    return ge_sr(key_ranks(scores, key, attack_subsets(len(scores), study.args.subset_seed)))[0]


def plot_ge(path, study, groups):
    interval = PROTOCOL["key_rank_report_interval"]
    levels = sorted({attack for _, _, attack in groups})
    colors = {v: plt.cm.tab10(i) for i, v in enumerate(VARIANTS)}
    fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4.4), sharey=True, squeeze=False)
    for ax, level in zip(axes[0], levels):
        for (variant, train, attack), runs in groups.items():
            if attack != level:
                continue
            ge = np.mean([run_ge(study, e["run"], level) for e in runs], axis=0)
            label = variant if train == level else f"{variant} (trained on desync {train})"
            ax.plot(interval * np.arange(1, len(ge) + 1), ge, color=colors[variant], linewidth=1.2,
                    linestyle="-" if train == level else "--", label=label)
        ax.axhline(GE_THRESHOLD, color="gray", linewidth=0.7, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylim(0.9, 256)
        ax.set_xlabel("attack traces")
        ax.set_title(f"attack on desync {level}", fontsize=9)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("guessing entropy (rank of the correct key)")
    fig.suptitle(f"GE over {PROTOCOL['key_rank_executions']} attack subsets (seed {study.args.subset_seed}), mean over "
                 f"seeds; dashed = trained on desync 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage_report(study):
    folder, args = study.folder, study.args
    if not study.evals:
        raise RuntimeError("no finished runs to report")
    stats = summarise(study)
    columns = ["variant", "train_desync", "attack_desync", "seeds", "reached_ge_lt_2", "traces_ge_lt_2",
               "att_ge_final", "att_ge_area", "parameters"]
    write_csv(os.path.join(folder, "results.csv"), [{c: s[c] for c in columns} for s in stats])
    plot_ge(os.path.join(folder, "ge_desync.png"), study, group_evals(study))

    def table(selected):
        return markdown_table(sorted(selected, key=lambda s: (s["attack_desync"], VARIANTS.index(s["variant"]))),
                              columns)

    transfer = [s for s in stats if s["train_desync"] == 0]
    direct = [s for s in stats if s["train_desync"] > 0]
    seeds = sorted({int(e["seed"]) for e in study.evals})
    lines = [
        f"# Phase 4: robustness to desynchronisation ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Data: ASCAD v1 variable key, byte {PROTOCOL['byte']}, {PROTOCOL['profiling_traces']} profiling traces, attack "
        f"traces {PROTOCOL['attack_traces'] // 2}-{PROTOCOL['attack_traces'] - 1}, validation traces "
        f"0-{PROTOCOL['attack_traces'] // 2 - 1}; desync 0 / 50 / 100 files {', '.join(DESYNC_FILES.values())}.",
        f"Training: up to {args.epochs} epochs, early stopping on the validation GE area of the training desync level, "
        f"stop after {args.patience} epochs without improvement; best epoch kept. Seeds {', '.join(map(str, seeds))}.",
        f"Attack metrics: the same key rank and {PROTOCOL['key_rank_executions']} attack subsets (seed "
        f"{args.subset_seed}) as Phases 2 and 3. The spread over seeds (independent trainings) is the one to trust.",
        "",
        "## Experiment A: trained on desync 0, attacked at each desync level",
        "",
        *table(transfer),
        "",
        "## Experiment B: trained and attacked on the same desync level",
        "",
        *table(direct),
        "",
        "## Meaning",
        "",
    ]
    for level in sorted({s["attack_desync"] for s in stats}):
        for title, selected in (("trained on desync 0", [s for s in transfer if s["attack_desync"] == level]),
                                ("trained on the same level", [s for s in direct if s["attack_desync"] == level])):
            if not selected:
                continue
            best = min(selected, key=lambda s: (-s["n_reached"] / s["seeds"], s["mean_traces"], s["mean_ge_final"]))
            others = ", ".join(f"{s['variant']} {s['reached_ge_lt_2']}" for s in selected)
            lines.append(f"- Attack on desync {level}, {title}: best `{best['variant']}` (GE < 2 in "
                         f"{best['reached_ge_lt_2']} seeds, traces {best['traces_ge_lt_2']}). Seeds reaching GE < 2: "
                         f"{others}.")
    lines += ["", "Files: `results.csv`, `eval.csv`, `ge_desync.png`, `runs.csv`, `predictions/`, `tensorboard/`."]
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: robustness to desynchronisation.")
    parser.add_argument("stage", choices=STAGES + ["all"])
    parser.add_argument("--study", help="existing study folder (default: a new phase4_<timestamp>/ for baseline / all)")
    parser.add_argument("--phase3", help="Phase 3 study folder for the CNN (default: latest)")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS, help=f"default {' '.join(VARIANTS)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="training seeds (default 1 2 3)")
    parser.add_argument("--levels", type=int, nargs="*", default=[50, 100], choices=[50, 100],
                        help="desync levels besides 0 (default 50 100; none = desync 0 only)")
    parser.add_argument("--epochs", type=int, default=50, help="maximum epochs per run (default 50)")
    parser.add_argument("--patience", type=int, default=10,
                        help="stop after this many epochs without a better validation GE area (default 10)")
    parser.add_argument("--aug-shift", type=int, default=100, help="cnn_aug: maximum shift when training on desync 0")
    parser.add_argument("--aug-noise", type=float, default=0.1, help="cnn_aug: Gaussian noise, in z-scored units")
    parser.add_argument("--subset-seed", type=int, default=0, help="attack subsets, as Phase 2 / 3 (default 0)")
    args = parser.parse_args()
    if args.stage in ("direct", "report") and not args.study:
        parser.error(f"stage {args.stage} needs --study <phase4 folder>")
    if args.epochs < 1 or args.patience < 1 or args.aug_shift < 1:
        parser.error("--epochs, --patience and --aug-shift must be >= 1")

    folder = resume_run(args.study) if args.study else start_run("phase4")
    stages = STAGES if args.stage == "all" else [args.stage]
    study = Study(folder, args)
    for stage in stages:
        study.new_analysis_ids = []
        with log_to(os.path.join(folder, f"log_{stage}_{datetime.now():%Y%m%d_%H%M%S}.txt")):
            print(f"Phase 4, stage {stage}: {vars(args)}")
            print(f"Protocol: {PROTOCOL}")
            try:
                globals()[f"stage_{stage}"](study)
            except BaseException as e:
                finish_run(folder, f"phase4_{stage}", args.subset_seed, study.new_analysis_ids,
                           status=f"failed: {type(e).__name__}")
                raise
            failed = [r["run"] for r in study.rows if r["status"] != "ok" and r["stage"] == stage]
            status = "ok" if not failed else f"{len(failed)} failed: {' '.join(failed)}"
            print(f"\nStage {stage}: {status}. Results: {folder}")
            finish_run(folder, f"phase4_{stage}", args.subset_seed, study.new_analysis_ids, status=status)
            if stage != stages[-1]:
                resume_run(folder)  # finish_run closed the timer; the next stage gets its own index row


if __name__ == "__main__":
    main()
