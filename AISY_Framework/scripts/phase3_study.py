# Phase 3: systematic CNN study with AISY, under the Phase 0 protocol.
#
# Process: one study folder, filled in stages. A stage can be rerun on the same folder: finished runs are skipped,
# failed runs are retried.
#   search   random search over the cnn_search family (HW classes, 50,000 profiling traces, CCE loss)
#   loss     the --top best search models retrained with the CER loss, same seed: CCE vs CER, paired
#   ablate   the best search model over --seeds: leakage model (HW / ID), profiling-set size (--sizes), and
#            architecture (best CNN vs AISY MLP vs cnn_sca_light, optionally ASCAD CNN_best)
#   report   ranked runs, ensembles of the top-N search models, ablation tables, GE curves next to Phase 2
#   all      search, loss, ablate, report
#
# Every run: AISY trains with early stopping on the validation GE area (custom/custom_metrics/ge_area.py) and stops
# after --patience epochs without improvement (PlateauStop). AISY logs it to the database (webapp) and TensorBoard
# logs loss and accuracy. Then the log-probabilities of the best-epoch model on the validation and attack sets are
# saved, and GE / SR are computed with scripts/sca_protocol.py on the same attack subsets as Phase 2.
# Models are chosen on the validation set only (GE area); attack-set numbers are only reported, never used to choose.
#
# Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.phase3_study all --trials 20
#     ..\aisy_env\Scripts\python.exe -m scripts.phase3_study search --trials 20
#     ..\aisy_env\Scripts\python.exe -m scripts.phase3_study loss --study resources/experiments/phase3_<timestamp>
#     ..\aisy_env\Scripts\python.exe -m scripts.phase3_study ablate --study ... --seeds 1 2 3
#     ..\aisy_env\Scripts\python.exe -m scripts.phase3_study report --study ...
#
# Output in resources/experiments/phase3_<timestamp>/:
#     runs.csv               one row per training run: stage, model, settings, analysis id, validation and attack metrics
#     predictions/<run>.npz  best-epoch log-probabilities (validation, attack); plaintexts.npz once
#     tensorboard/<run>/     loss and accuracy per epoch
#     ranking.csv, ensembles.csv, ablations.csv, loss.csv, ge_best.png, ablation_profiling.png, summary.md  (report)
#     log_<stage>_<timestamp>.txt
# One row per stage is added to resources/experiments/index.csv.
import argparse
import csv
import json
import os
import random
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

import aisy_sca
import tensorflow as tf
from custom.custom_datasets.datasets import datasets_dict
from custom.custom_loss.cer import cer_loss
from custom.custom_metrics.ge_area import HISTORY_KEY, VALIDATION_SEED
from custom.custom_models.neural_networks import ASCAD_cnn, cnn_sca_light, cnn_search, mlp
from scripts.experiment_log import log_to
from scripts.experiment_output import EXPERIMENTS_FOLDER, configure_aisy, finish_run, resume_run, start_run
from scripts.experiment_seed import set_seed
from scripts.phase0_compare import mean_std
from scripts.sca_protocol import (GE_THRESHOLD, PROTOCOL, attack_subsets, ge_area, ge_sr, key_ranks, key_scores,
                                  traces_to_threshold)

STAGES = ["search", "loss", "ablate", "report"]

# Random search space (cnn_search in custom/custom_models/neural_networks.py). Kernels up to 51 and pooling up to 8
# cover receptive fields from tens to hundreds of samples; Flatten lets the dense layers combine the shares anyway.
SEARCH_SPACE = {
    "conv_blocks": [1, 2, 3],
    "filters": [8, 16, 32],
    "kernel": [11, 25, 51],
    "pool": [2, 4, 8],
    "dense_layers": [1, 2, 3],
    "dense_width": [64, 128, 256],
    "activation": ["relu", "selu"],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [128, 256],
}
FIXED_MODELS = {"mlp": mlp, "cnn_sca_light": cnn_sca_light, "ascad_cnn_best": ASCAD_cnn}
FIXED_BATCH_SIZE = 128  # Phase 0 default, for the fixed architectures
CER_SHUFFLES = 10       # n in custom/custom_loss/cer.py

EARLY_STOPPING = {"metrics": {"ge_area": {"direction": "min", "class": "custom.custom_metrics.ge_area",
                                          "parameters": []}}}
RUN_COLUMNS = ["run", "stage", "variant", "model", "hp", "leakage_model", "profiling_traces", "loss", "seed",
               "batch_size", "status", "analysis_id", "parameters", "epochs_run", "best_epoch", "val_ge_area",
               "val_ge_final", "att_ge_area", "att_ge_final", "att_median_rank", "att_sr_final", "att_traces_ge_lt_2",
               "seconds", "error"]


# ---------------------------------------------------------------------------------------------------------------
# Study folder
# ---------------------------------------------------------------------------------------------------------------
class Study:
    def __init__(self, folder, args):
        self.folder, self.args = folder, args
        self.runs_file = os.path.join(folder, "runs.csv")
        self.predictions = os.path.join(folder, "predictions")
        os.makedirs(self.predictions, exist_ok=True)
        self.rows = []
        if os.path.exists(self.runs_file):
            with open(self.runs_file, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
        self.dataset = (None, None)  # (leakage model, profiling traces) -> AISY dataset; one slot keeps memory low
        self.new_analysis_ids = []

    def ok_rows(self, stage=None):
        return [r for r in self.rows if r["status"] == "ok" and (stage is None or r["stage"] == stage)]

    def done(self, run):
        return any(r["run"] == run for r in self.ok_rows())

    def save_row(self, row):
        self.rows = [r for r in self.rows if r["run"] != row["run"]] + [row]
        with open(self.runs_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

    def run(self, config):
        # Train one model unless it already finished; returns its row.
        if self.done(config["run"]):
            print(f"{config['run']}: already done, skipped")
            return next(r for r in self.ok_rows() if r["run"] == config["run"])
        print(f"\n===== {config['run']} ({config['variant']}) =====")
        row = {k: config.get(k, "") for k in RUN_COLUMNS}
        row["hp"] = json.dumps(config["hp"], sort_keys=True) if config["hp"] else ""
        start = time.time()
        try:
            row.update(train(config, self))
            row["status"] = "ok"
            self.new_analysis_ids.append(row["analysis_id"])
        except Exception as e:  # one failed run must not stop the study
            traceback.print_exc()
            row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
        row["seconds"] = round(time.time() - start, 1)
        self.save_row(row)
        print(f"{config['run']}: {row['status']}, validation GE area {row.get('val_ge_area')}, attack GE "
              f"{row.get('att_ge_final')}, traces to GE < 2 {row.get('att_traces_ge_lt_2') or 'not reached'}, "
              f"{row['seconds']} s")
        return row

    def load_predictions(self, run):
        data = np.load(os.path.join(self.predictions, f"{run}.npz"))
        return data["validation"], data["attack"]

    def plaintexts(self):
        data = np.load(os.path.join(self.predictions, "plaintexts.npz"))
        return data["validation"], data["attack"], int(data["key"])


# ---------------------------------------------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------------------------------------------
def build_model(config, classes, samples):
    if config["model"] != "cnn_search":
        return FIXED_MODELS[config["model"]](classes, samples)
    loss = (cer_loss({"classes": classes}, parameters={"n": CER_SHUFFLES}) if config["loss"] == "cer"
            else "categorical_crossentropy")
    return cnn_search(classes, samples, config["hp"], loss=loss, name=config["run"])


def train(config, study):
    args = study.args
    tf.keras.backend.clear_session()
    set_seed(config["seed"])
    data_key = (config["leakage_model"], config["profiling_traces"])
    aisy = aisy_sca.Aisy()
    configure_aisy(aisy, study.folder)
    # Reuse the loaded, normalised and split dataset when the next run needs the same one (AISY reads the whole file).
    aisy.set_dataset(datasets_dict[PROTOCOL["dataset"]], study.dataset[1] if study.dataset[0] == data_key else None)
    aisy.set_number_of_profiling_traces(config["profiling_traces"])
    aisy.set_number_of_attack_traces(PROTOCOL["attack_traces"])
    aisy.set_aes_leakage_model(leakage_model=config["leakage_model"], byte=PROTOCOL["byte"])
    aisy.set_batch_size(config["batch_size"])
    aisy.set_epochs(args.epochs)
    model = build_model(config, aisy.settings["classes"], aisy.settings["number_of_samples"])
    parameters = model.count_params()
    aisy.add_neural_network(model, name=config["run"], seed=config["seed"])
    aisy.run(
        key_rank_executions=PROTOCOL["key_rank_executions"],
        key_rank_attack_traces=PROTOCOL["key_rank_attack_traces"],
        key_rank_report_interval=PROTOCOL["key_rank_report_interval"],
        early_stopping=EARLY_STOPPING,
        callbacks=[
            {"class": "custom.custom_callbacks.callbacks.PlateauStop", "name": "plateau_stop",
             "parameters": {"history_key": HISTORY_KEY, "patience": args.patience}},
            {"class": "custom.custom_callbacks.callbacks.TensorBoardLog", "name": "tensorboard",
             "parameters": {"log_dir": os.path.join(study.folder, "tensorboard", config["run"])}},
        ],
    )
    study.dataset = (data_key, aisy.dataset)

    # AISY loads the best-epoch weights to compute its early-stopping metrics and keeps them in the model.
    trained, dataset = aisy.models["0"]["model"], aisy.dataset
    log_val = np.log(trained.predict(dataset.x_validation, batch_size=1000, verbose=0) + 1e-36).astype(np.float32)
    log_att = np.log(trained.predict(dataset.x_attack, batch_size=1000, verbose=0) + 1e-36).astype(np.float32)
    np.savez(os.path.join(study.predictions, f"{config['run']}.npz"), validation=log_val, attack=log_att)
    plaintexts = os.path.join(study.predictions, "plaintexts.npz")
    if not os.path.exists(plaintexts):
        np.savez(plaintexts, validation=dataset.plaintext_validation, attack=dataset.plaintext_attack,
                 key=aisy.settings["good_key"])

    history = aisy.settings[HISTORY_KEY]
    metrics = evaluate(study, [(log_val, log_att, config["leakage_model"])])
    return {"analysis_id": aisy.settings["analysis_id"], "parameters": parameters, "epochs_run": len(history),
            "best_epoch": int(np.argmin(history)) + 1, **metrics["row"]}


# ---------------------------------------------------------------------------------------------------------------
# Metrics (same rules as Phase 2: scripts/sca_protocol.py)
# ---------------------------------------------------------------------------------------------------------------
def evaluate(study, models):
    # models: [(validation log-p, attack log-p, leakage model)]; several = ensemble (log-probabilities of every
    # key guess added over the models, then over the traces).
    plain_val, plain_att, key = study.plaintexts()
    byte = PROTOCOL["byte"]
    val = sum(key_scores(v, plain_val, byte, lm) for v, _, lm in models)
    att = sum(key_scores(a, plain_att, byte, lm) for _, a, lm in models)
    val_ge, _ = ge_sr(key_ranks(val, key, attack_subsets(len(val), VALIDATION_SEED)))
    att_ranks = key_ranks(att, key, attack_subsets(len(att), study.args.subset_seed))
    att_ge, att_sr = ge_sr(att_ranks)
    row = {"val_ge_area": round(ge_area(val_ge), 2), "val_ge_final": round(float(val_ge[-1]), 2),
           "att_ge_area": round(ge_area(att_ge), 2), "att_ge_final": round(float(att_ge[-1]), 2),
           "att_median_rank": float(np.median(att_ranks[:, -1])), "att_sr_final": round(float(att_sr[-1]), 2),
           "att_traces_ge_lt_2": traces_to_threshold(list(att_ge), PROTOCOL["key_rank_report_interval"])}
    return {"row": row, "att_ge": att_ge}


def run_metrics(study, row):
    log_val, log_att = study.load_predictions(row["run"])
    return evaluate(study, [(log_val, log_att, row["leakage_model"])])


def by_validation(rows):
    return sorted(rows, key=lambda r: float(r["val_ge_area"]))


def number(value):
    return None if value in ("", None) else float(value)


# ---------------------------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------------------------
def sample_hp(seed, trial):
    rng = random.Random(f"{seed}-{trial}")  # the same trial number always gives the same model
    return {name: rng.choice(values) for name, values in SEARCH_SPACE.items()}


def stage_search(study):
    args = study.args
    for trial in range(args.trials):
        hp = sample_hp(args.seed, trial)
        study.run({"run": f"search_{trial:02d}", "stage": "search", "variant": "search", "model": "cnn_search",
                   "hp": hp, "leakage_model": "HW", "profiling_traces": PROTOCOL["profiling_traces"], "loss": "cce",
                   "seed": args.seed, "batch_size": hp["batch_size"]})


def stage_loss(study):
    top = by_validation(study.ok_rows("search"))[:study.args.top]
    if not top:
        raise RuntimeError("no finished search runs: run the search stage first")
    for source in top:
        hp = json.loads(source["hp"])
        study.run({"run": f"cer_{source['run']}", "stage": "loss", "variant": f"cer ({source['run']})",
                   "model": "cnn_search", "hp": hp, "leakage_model": "HW", "profiling_traces": source["profiling_traces"],
                   "loss": "cer", "seed": source["seed"], "batch_size": hp["batch_size"]})


def best_search_config(study):
    # Best search-family model on the validation GE area, over the search and loss stages.
    candidates = study.ok_rows("search") + study.ok_rows("loss")
    if not candidates:
        raise RuntimeError("no finished search runs: run the search stage first")
    best = by_validation(candidates)[0]
    print(f"Best model on the validation set: {best['run']} (GE area {best['val_ge_area']}, loss {best['loss']}, "
          f"hp {best['hp']})")
    return best


def stage_ablate(study):
    args = study.args
    best = best_search_config(study)
    hp = json.loads(best["hp"])
    base = {"model": "cnn_search", "hp": hp, "leakage_model": "HW", "profiling_traces": PROTOCOL["profiling_traces"],
            "loss": best["loss"], "batch_size": hp["batch_size"]}
    variants = [("best_cnn", {})]
    variants += [(arch, {"model": arch, "hp": None, "loss": "cce", "batch_size": FIXED_BATCH_SIZE})
                 for arch in args.archs]  # fixed architectures compile their own CCE loss
    variants += [("leakage_ID", {"leakage_model": "ID"})]
    variants += [(f"profiling_{n}", {"profiling_traces": n}) for n in args.sizes if n != PROTOCOL["profiling_traces"]]
    # Variants in data order (same dataset back to back), seeds inside, so the loaded dataset is reused.
    for variant, change in variants:
        for seed in args.seeds:
            study.run({**base, **change, "run": f"ablate_{variant}_s{seed}", "stage": "ablate", "variant": variant,
                       "seed": seed})


# ---------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------
def phase2_runs(subset_seed):
    # Finished Phase 2 runs whose attack subsets are the ones used here (index.csv seed = Phase 2 --seed), newest first.
    index = os.path.join(EXPERIMENTS_FOLDER, "index.csv")
    if not os.path.exists(index):
        return []
    with open(index, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "phase2" and r["status"] == "ok"
                and r["seed"] == str(subset_seed)]
    return [r["folder"] for r in sorted(rows, key=lambda r: r["started"], reverse=True)]


def phase2_baseline(args):
    # Phase 2 run on the same attack subsets (or --phase2): GE curves and the best real classic attack.
    folders = [args.phase2] if args.phase2 else phase2_runs(args.subset_seed)
    for folder in folders:
        folder = folder.replace("\\", "/")
        results, curves = os.path.join(folder, "results.csv"), os.path.join(folder, "curves.npz")
        if os.path.exists(results) and os.path.exists(curves):
            with open(results, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["status"] == "ok" and r["traces_ge_lt_2"]
                        and r["method"] != "cpa_known_mask"]
            best = min(rows, key=lambda r: int(r["traces_ge_lt_2"])) if rows else None
            return {"folder": folder, "curves": np.load(curves), "best": best}
    print(f"No Phase 2 run with --seed {args.subset_seed} found: report without the classic baseline")
    return None


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def markdown_table(rows, columns):
    def cell(row, column):
        value = row.get(column, "")
        if value in ("", None, "None"):
            return "not reached" if "traces_ge_lt_2" in column else ""
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    return lines + ["| " + " | ".join(cell(r, c) for c in columns) + " |" for r in rows]


def ensembles(study):
    # Top-N search models by validation GE area; N chosen on the validation set too.
    ranked = by_validation(study.ok_rows("search"))
    rows, curves = [], {}
    for n in sorted({min(n, len(ranked)) for n in study.args.ensemble_sizes}):
        members = ranked[:n]
        models = [(*study.load_predictions(r["run"]), r["leakage_model"]) for r in members]
        result = evaluate(study, models)
        rows.append({"ensemble": f"top {n}", "members": " ".join(r["run"] for r in members), **result["row"]})
        curves[f"top {n}"] = result["att_ge"]
    return rows, curves


def ablation_summary(study):
    rows = []
    variants = []
    for r in study.ok_rows("ablate"):
        if r["variant"] not in variants:
            variants.append(r["variant"])
    for variant in variants:
        runs = [r for r in study.ok_rows("ablate") if r["variant"] == variant]
        traces = [number(r["att_traces_ge_lt_2"]) for r in runs]
        rows.append({
            "variant": variant, "seeds": len(runs),
            "reached_ge_lt_2": f"{sum(t is not None for t in traces)}/{len(runs)}",
            "traces_ge_lt_2": mean_std(traces),
            "att_ge_final": mean_std([number(r["att_ge_final"]) for r in runs]),
            "att_ge_area": mean_std([number(r["att_ge_area"]) for r in runs]),
            "val_ge_area": mean_std([number(r["val_ge_area"]) for r in runs]),
            "epochs_run": mean_std([number(r["epochs_run"]) for r in runs]),
            "parameters": runs[0]["parameters"],
        })
    return rows


def plot_ge(path, study, curves, baseline):
    interval = PROTOCOL["key_rank_report_interval"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, ge in curves.items():
        ax.plot(interval * np.arange(1, len(ge) + 1), ge, label=name, linewidth=1.2)
    if baseline:
        for attack in ("template_poi_HW", "template_pca_HW"):
            key = f"{attack}_ge"
            if key in baseline["curves"]:
                ge = baseline["curves"][key]
                ax.plot(interval * np.arange(1, len(ge) + 1), ge, label=f"Phase 2 {attack}", linewidth=1.0,
                        linestyle="--")
    ax.axhline(GE_THRESHOLD, color="gray", linewidth=0.7, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 256)
    ax.set_xlabel("attack traces")
    ax.set_ylabel("guessing entropy (rank of the correct key)")
    ax.set_title(f"{PROTOCOL['dataset']}, byte {PROTOCOL['byte']}: GE over {PROTOCOL['key_rank_executions']} attack "
                 f"subsets (seed {study.args.subset_seed}, same as Phase 2)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiling(path, study):
    # Attack GE area against the profiling-set size: mean and spread over the seeds.
    runs = [r for r in study.ok_rows("ablate") if r["variant"].startswith("profiling_") or r["variant"] == "best_cnn"]
    sizes = sorted({int(r["profiling_traces"]) for r in runs})
    if len(sizes) < 2:
        return False
    values = [[float(r["att_ge_area"]) for r in runs if int(r["profiling_traces"]) == n] for n in sizes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sizes, [np.mean(v) for v in values], yerr=[np.std(v) for v in values], marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("profiling traces")
    ax.set_ylabel("attack GE area (mean GE over 10-1,000 traces)")
    ax.set_title(f"Best CNN, HW classes: mean ± std over {len(study.args.seeds)} seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def stage_report(study):
    folder, args = study.folder, study.args
    if not study.ok_rows():
        raise RuntimeError("no finished runs to report")
    baseline = phase2_baseline(args)
    columns = ["run", "variant", "loss", "parameters", "epochs_run", "best_epoch", "val_ge_area", "att_ge_area",
               "att_ge_final", "att_median_rank", "att_sr_final", "att_traces_ge_lt_2", "analysis_id"]
    ranking = [{c: r[c] for c in columns} for r in by_validation(study.ok_rows())]
    write_csv(os.path.join(folder, "ranking.csv"), ranking)

    ensemble_rows, ensemble_curves = ensembles(study) if study.ok_rows("search") else ([], {})
    write_csv(os.path.join(folder, "ensembles.csv"), ensemble_rows)
    ablation_rows = ablation_summary(study)
    write_csv(os.path.join(folder, "ablations.csv"), ablation_rows)
    loss_rows = []
    for cer in study.ok_rows("loss"):
        cce = next((r for r in study.ok_rows("search") if f"cer_{r['run']}" == cer["run"]), None)
        if cce:
            loss_rows.append({"model": cce["run"], "cce_val_ge_area": cce["val_ge_area"],
                              "cer_val_ge_area": cer["val_ge_area"], "cce_att_ge_area": cce["att_ge_area"],
                              "cer_att_ge_area": cer["att_ge_area"], "cce_traces_ge_lt_2": cce["att_traces_ge_lt_2"],
                              "cer_traces_ge_lt_2": cer["att_traces_ge_lt_2"]})
    write_csv(os.path.join(folder, "loss.csv"), loss_rows)

    # GE curves: best single run on validation, the ensemble with the best validation GE area, and the mean curve of
    # each architecture over the ablation seeds.
    curves = {}
    best = by_validation(study.ok_rows())[0]
    curves[f"best single: {best['run']}"] = run_metrics(study, best)["att_ge"]
    if ensemble_rows:
        chosen = min(ensemble_rows, key=lambda r: r["val_ge_area"])
        curves[f"ensemble {chosen['ensemble']} (chosen on validation)"] = ensemble_curves[chosen["ensemble"]]
    for variant in ["best_cnn"] + list(args.archs):
        runs = [r for r in study.ok_rows("ablate") if r["variant"] == variant]
        if runs:
            label = f"{variant}, mean of {len(runs)} seeds" if len(runs) > 1 else f"{variant}, 1 seed"
            curves[label] = np.mean([run_metrics(study, r)["att_ge"] for r in runs], axis=0)
    plot_ge(os.path.join(folder, "ge_best.png"), study, curves, baseline)
    profiling_plot = plot_profiling(os.path.join(folder, "ablation_profiling.png"), study)

    lines = [
        f"# Phase 3: systematic CNN study ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Data: {PROTOCOL['dataset']}, byte {PROTOCOL['byte']}, attack traces {PROTOCOL['attack_traces'] // 2}-"
        f"{PROTOCOL['attack_traces'] - 1}, validation traces 0-{PROTOCOL['attack_traces'] // 2 - 1}.",
        f"Training: up to {args.epochs} epochs, early stopping on the validation GE area (mean validation GE over "
        f"10-1,000 traces), stop after {args.patience} epochs without improvement; the best epoch's weights are kept.",
        f"Selection: validation GE area only. Attack metrics: same key rank and the same {PROTOCOL['key_rank_executions']} "
        f"attack subsets (seed {args.subset_seed}) as Phase 2.",
        f"Uncertainty: the {PROTOCOL['key_rank_executions']} subsets all come from the same "
        f"{PROTOCOL['attack_traces'] // 2} attack traces, so they overlap and the ± over subsets understates it. "
        f"The spread over the ablation seeds (independent trainings) is the one to trust; a near-random model can land "
        f"well above or below GE 128 by chance.",
        "",
        "## Ranked runs (by validation GE area)",
        "",
        *markdown_table(ranking[:15], ["run", "loss", "parameters", "best_epoch", "val_ge_area", "att_ge_area",
                                       "att_ge_final", "att_sr_final", "att_traces_ge_lt_2"]),
    ]
    if ensemble_rows:
        lines += ["", "## Ensembles of the top-N search models", "",
                  *markdown_table(ensemble_rows, ["ensemble", "val_ge_area", "att_ge_area", "att_ge_final",
                                                  "att_sr_final", "att_traces_ge_lt_2"])]
    if loss_rows:
        lines += ["", "## Loss: CCE vs CER (same model and seed)", "",
                  *markdown_table(loss_rows, list(loss_rows[0]))]
    if ablation_rows:
        lines += ["", f"## Ablations (mean ± std over seeds {', '.join(str(s) for s in args.seeds)})", "",
                  *markdown_table(ablation_rows, list(ablation_rows[0]))]
    lines += ["", "## Meaning", ""]
    best_traces = number(best["att_traces_ge_lt_2"])
    lines.append(f"- Best single model on the validation set: `{best['run']}`, attack GE < 2 after "
                 f"{int(best_traces) if best_traces else 'more than 1,000'} traces.")
    if baseline and baseline["best"]:
        bar = int(baseline["best"]["traces_ge_lt_2"])
        verdict = ("beats" if best_traces and best_traces < bar else "does not beat")
        lines.append(f"- Phase 2 bar: `{baseline['best']['attack']}` reaches GE < 2 after {bar} traces "
                     f"(`{baseline['folder']}`). The best CNN {verdict} it.")
    lines += ["", "Files: `ranking.csv`, `ensembles.csv`, `loss.csv`, `ablations.csv`, `ge_best.png`"
              + (", `ablation_profiling.png`" if profiling_plot else "") + ", `runs.csv`, `predictions/`, `tensorboard/`."]
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 3: systematic CNN study with AISY.")
    parser.add_argument("stage", choices=STAGES + ["all"])
    parser.add_argument("--study", help="existing study folder (default: a new phase3_<timestamp>/ for search / all)")
    parser.add_argument("--trials", type=int, default=20, help="random search models (default 20)")
    parser.add_argument("--epochs", type=int, default=50, help="maximum epochs per run (default 50)")
    parser.add_argument("--patience", type=int, default=10,
                        help="stop after this many epochs without a better validation GE area (default 10)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the search: sampling and training (default 0)")
    parser.add_argument("--top", type=int, default=3, help="search models retrained with CER (default 3)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="ablation seeds (default 1 2 3)")
    parser.add_argument("--archs", nargs="+", default=["mlp", "cnn_sca_light"], choices=list(FIXED_MODELS),
                        help="fixed architectures in the ablation (default mlp cnn_sca_light; ascad_cnn_best is "
                             "~90M parameters, very slow on a CPU)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 10000, 25000, 50000],
                        help="profiling-set sizes in the ablation (default 5000 10000 25000 50000)")
    parser.add_argument("--ensemble-sizes", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--subset-seed", type=int, default=0, help="attack subsets, as Phase 2 --seed (default 0)")
    parser.add_argument("--phase2", help="Phase 2 run folder for the baseline (default: latest)")
    args = parser.parse_args()
    if args.stage in ("loss", "ablate", "report") and not args.study:
        parser.error(f"stage {args.stage} needs --study <phase3 folder>")
    if args.trials < 1 or args.epochs < 1 or args.patience < 1 or args.top < 1:
        parser.error("--trials, --epochs, --patience and --top must be >= 1")
    if any(not 0 < n <= PROTOCOL["profiling_traces"] for n in args.sizes):
        parser.error(f"--sizes must be between 1 and {PROTOCOL['profiling_traces']}")

    folder = resume_run(args.study) if args.study else start_run("phase3")
    stages = STAGES if args.stage == "all" else [args.stage]
    study = Study(folder, args)
    for stage in stages:
        study.new_analysis_ids = []
        with log_to(os.path.join(folder, f"log_{stage}_{datetime.now():%Y%m%d_%H%M%S}.txt")):
            print(f"Phase 3, stage {stage}: {vars(args)}")
            print(f"Protocol: {PROTOCOL}")
            try:
                globals()[f"stage_{stage}"](study)
            except BaseException as e:
                finish_run(folder, f"phase3_{stage}", args.seed, study.new_analysis_ids,
                           status=f"failed: {type(e).__name__}")
                raise
            failed = [r["run"] for r in study.rows if r["status"] != "ok" and r["stage"] == stage]
            status = "ok" if not failed else f"{len(failed)} failed: {' '.join(failed)}"
            print(f"\nStage {stage}: {status}. Results: {folder}")
            finish_run(folder, f"phase3_{stage}", args.seed, study.new_analysis_ids, status=status)
            if stage != stages[-1]:
                resume_run(folder)  # finish_run closed the timer; the next stage gets its own index row


if __name__ == "__main__":
    main()
