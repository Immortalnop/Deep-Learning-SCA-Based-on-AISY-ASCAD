# Phase 5: explainability. Does the CNN look at the samples that leak (the Phase 1 SNR peaks)?
#
# Process: AISY deletes its best-epoch weights, so the stage `train` retrains the Phase 4 desync 0 models with the same
# recipe and seed (scripts/phase4_robustness.py) and keeps the weights -> the stage `saliency` takes the gradient of the
# log-probability of the correct class with respect to every input sample, averages its absolute value over attack
# traces, and compares that profile with the Phase 1 SNR of the masked value and of the mask.
#   train      retrain --variants on desync 0 (one run per variant and seed), save weights to models/
#   saliency   per run: saliency profile at desync 0 (and at --levels, with the desync 0 model), overlap with the
#              SNR peaks, and an occlusion test; report in summary.md
#   all        train, saliency
#
# Overlap with the SNR (the SNR is computed on the same byte; masked value = Sbox(p ^ k) ^ r, mask = r):
#   spearman_*   rank correlation of the saliency profile with the SNR of the masked value / of the mask
#   enrichment   share of the saliency inside +-window samples of the SNR peaks, divided by the share of samples
#                inside those windows (1 = no preference for the leaking samples)
# Occlusion test: set samples to 0 (the z-scored mean) and measure the attack on the same subsets as Phases 2-4.
#   peaks      the windows around the SNR peaks
#   top        the same number of samples with the highest saliency
#   random     the same number of random samples (mean of --random-draws draws)
# If the saliency is faithful, `top` hurts the attack at least as much as `random`. If the model uses the peaks,
# `peaks` hurts it clearly more than `random`.
#
# Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.phase5_saliency all
#     ..\aisy_env\Scripts\python.exe -m scripts.phase5_saliency all --variants cnn cnn_aug
#     ..\aisy_env\Scripts\python.exe -m scripts.phase5_saliency saliency --study resources/experiments/phase5_<timestamp>
#
# Output in resources/experiments/phase5_<timestamp>/:
#     runs.csv, eval.csv, predictions/, tensorboard/   from the retraining (same columns as Phase 4)
#     models/<run>.h5                                  weights of the best epoch
#     saliency.npz                                     profile per run and desync level, and the SNR curves
#     overlap.csv, occlusion.csv                       metrics
#     saliency.png, occlusion.png, summary.md
#     log_<stage>_<timestamp>.txt
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scripts import phase4_robustness as phase4
from scripts.experiment_log import log_to
from scripts.experiment_output import EXPERIMENTS_FOLDER, finish_run, resume_run, start_run
from scripts.phase3_study import build_model, markdown_table, write_csv
from scripts.sca_protocol import (HW, PROTOCOL, S_BOX, attack_subsets, ge_area, ge_sr, key_ranks,
                                  key_scores, traces_to_threshold)

STAGES = ["train", "saliency"]
SNR_TARGETS = {"masked value": "sbox_xor_r", "mask": "r"}  # keys of snr.npz are <target>_<traces>
SNR_TRACES = 50000
BATCH = 250


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------
def latest(prefix, required_file, override=None):
    if override:
        return override.replace("\\", "/").rstrip("/")
    folders = sorted(f for f in os.listdir(EXPERIMENTS_FOLDER)
                     if f.startswith(prefix) and os.path.exists(os.path.join(EXPERIMENTS_FOLDER, f, required_file)))
    if not folders:
        raise RuntimeError(f"no {prefix}* study with {required_file} found in {EXPERIMENTS_FOLDER}")
    return f"{EXPERIMENTS_FOLDER}/{folders[-1]}"


def load_snr(args):
    folder = latest("phase1_", "snr.npz", args.phase1)
    data = np.load(os.path.join(folder, "snr.npz"))
    snr = {name: data[f"{target}_{SNR_TRACES}"].astype(np.float64) for name, target in SNR_TARGETS.items()}
    peaks = {name: int(np.argmax(values)) for name, values in snr.items()}
    print(f"SNR from {folder}: peaks " + ", ".join(f"{name} at sample {sample}" for name, sample in peaks.items()))
    return snr, peaks


def run_config(study, variant, seed, base):
    config = phase4.make_config(variant, 0, seed, base, "train")
    config["weights_file"] = os.path.join(study.folder, "models", f"{config['run']}.h5")
    return config


# ---------------------------------------------------------------------------------------------------------------
# Stage train
# ---------------------------------------------------------------------------------------------------------------
def stage_train(study):
    os.makedirs(os.path.join(study.folder, "models"), exist_ok=True)
    base = phase4.best_phase3_cnn(study.args)
    for variant in study.args.variants:
        for seed in study.args.seeds:
            config = run_config(study, variant, seed, base)
            if study.done(config["run"]) and not os.path.exists(config["weights_file"]):
                raise RuntimeError(f"{config['run']} is done but {config['weights_file']} is missing: "
                                   f"delete its row in runs.csv to retrain it")
            study.run(config)


# ---------------------------------------------------------------------------------------------------------------
# Saliency
# ---------------------------------------------------------------------------------------------------------------
def load_model(row, classes, samples):
    config = {"model": row["model"], "hp": json.loads(row["hp"]) if row["hp"] else None, "loss": row["loss"],
              "run": row["run"]}
    model = build_model(config, classes, samples)
    model.load_weights(os.path.join(row["folder"], "models", f"{row['run']}.h5"))
    return model


def correct_classes(plaintext, key):
    values = S_BOX[plaintext[:, PROTOCOL["byte"]] ^ key]
    return (HW[values] if PROTOCOL["leakage_model"] == "HW" else values).astype(np.int64)


def saliency_profile(model, x, labels):
    # Mean over traces of |d log p(correct class) / d sample|, and the same for |gradient * input|.
    grad_sum, gradx_sum = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    for start in range(0, len(x), BATCH):
        batch = tf.constant(x[start:start + BATCH, :, None])
        index = tf.constant(labels[start:start + BATCH])
        with tf.GradientTape() as tape:
            tape.watch(batch)
            probabilities = model(batch, training=False)
            picked = tf.gather(probabilities, index, axis=1, batch_dims=1)
            score = tf.reduce_sum(tf.math.log(picked + 1e-36))
        gradient = tape.gradient(score, batch).numpy()[:, :, 0]
        grad_sum += np.abs(gradient).sum(axis=0)
        gradx_sum += np.abs(gradient * x[start:start + BATCH]).sum(axis=0)
    return grad_sum / len(x), gradx_sum / len(x)


def ranks_of(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def spearman(a, b):
    return float(np.corrcoef(ranks_of(a), ranks_of(b))[0, 1])


def window_mask(centres, window, size):
    mask = np.zeros(size, dtype=bool)
    for centre in centres:
        mask[max(centre - window, 0):centre + window + 1] = True
    return mask


def overlap(profile, snr, peaks, window):
    mask = window_mask(peaks.values(), window, len(profile))
    row = {"spearman_masked_value": round(spearman(profile, snr["masked value"]), 3),
           "spearman_mask": round(spearman(profile, snr["mask"]), 3),
           "enrichment": round(float(profile[mask].sum() / profile.sum() / mask.mean()), 2),
           "samples_in_windows": int(mask.sum())}
    for name, peak in peaks.items():  # percentile of the saliency at the SNR peak (100 = the most salient sample)
        row[f"percentile_at_{name.replace(' ', '_')}_peak"] = round(
            100 * float((profile <= profile[max(peak - window, 0):peak + window + 1].max()).mean()), 1)
    return row


def attack_metrics(study, model, x, plaintext, key):
    log_probabilities = phase4.log_probabilities(model, x)
    if not np.isfinite(log_probabilities).all():
        raise RuntimeError("non-finite predictions")
    scores = key_scores(log_probabilities, plaintext, PROTOCOL["byte"], PROTOCOL["leakage_model"])
    ranks = key_ranks(scores, key, attack_subsets(len(scores), study.args.subset_seed))
    ge, _ = ge_sr(ranks)
    return {"ge_area": round(ge_area(ge), 2), "ge_final": round(float(ge[-1]), 2),
            "traces_ge_lt_2": traces_to_threshold(list(ge), PROTOCOL["key_rank_report_interval"])}


def occlusion(study, model, x, plaintext, key, profile, peaks):
    args = study.args
    peak_mask = window_mask(peaks.values(), args.window, x.shape[1])
    count = int(peak_mask.sum())
    top_mask = np.zeros(x.shape[1], dtype=bool)
    top_mask[np.argsort(profile)[-count:]] = True
    rng = np.random.default_rng(args.subset_seed)
    cases = {"none": [np.zeros(x.shape[1], dtype=bool)], "peaks": [peak_mask], "top": [top_mask],
             "random": [np.isin(np.arange(x.shape[1]), rng.choice(x.shape[1], count, replace=False))
                        for _ in range(args.random_draws)]}
    rows = []
    for case, masks in cases.items():
        results = []
        for mask in masks:
            blocked = x.copy()
            blocked[:, mask] = 0
            results.append(attack_metrics(study, model, blocked, plaintext, key))
        reached = [r["traces_ge_lt_2"] for r in results if r["traces_ge_lt_2"] is not None]
        rows.append({"occluded": case, "samples": 0 if case == "none" else count,
                     "ge_area": round(float(np.mean([r["ge_area"] for r in results])), 2),
                     "ge_final": round(float(np.mean([r["ge_final"] for r in results])), 2),
                     "reached_ge_lt_2": f"{len(reached)}/{len(results)}",
                     "traces_ge_lt_2": round(float(np.mean(reached)), 1) if reached else ""})
    return rows


def stage_saliency(study):
    args = study.args
    snr, peaks = load_snr(args)
    runs = [r for r in study.ok_rows("train") if r["variant"] in args.variants and int(r["seed"]) in args.seeds]
    if not runs:
        raise RuntimeError("no finished training runs for these --variants / --seeds: run the stage train first")
    profiles, overlaps, occlusions = {"snr_masked_value": snr["masked value"], "snr_mask": snr["mask"]}, [], []
    mean, std = study.normalisation(0)
    for row in runs:
        row = dict(row, folder=study.folder)
        tf.keras.backend.clear_session()
        model = load_model(row, 9 if PROTOCOL["leakage_model"] == "HW" else 256, len(mean))
        for level in [0] + args.levels:
            data = study.attack(level)
            half = len(data["x"]) // 2
            x = ((data["x"][half:] - mean) / std).astype(np.float32)  # z-score of the desync 0 profiling traces
            plaintext = data["plain_att"]
            labels = correct_classes(plaintext, data["key"])
            n = min(args.traces, len(x))
            grad, gradx = saliency_profile(model, x[:n], labels[:n])
            profiles[f"{row['run']}_d{level}"], profiles[f"{row['run']}_d{level}_gradx"] = grad, gradx
            overlaps.append({"run": row["run"], "variant": row["variant"], "attack_desync": level, "traces": n,
                             **overlap(grad, snr, peaks, args.window)})
            print(f"{row['run']} attack desync {level}: {overlaps[-1]}")
            if level == 0:
                for result in occlusion(study, model, x, plaintext, data["key"], grad, peaks):
                    occlusions.append({"run": row["run"], "variant": row["variant"], **result})
                    print(f"{row['run']} occlusion: {occlusions[-1]}")
    np.savez(os.path.join(study.folder, "saliency.npz"), **profiles)
    write_csv(os.path.join(study.folder, "overlap.csv"), overlaps)
    write_csv(os.path.join(study.folder, "occlusion.csv"), occlusions)
    plot_saliency(study, runs, profiles, peaks)
    plot_occlusion(study, occlusions)
    write_summary(study, runs, overlaps, occlusions, peaks)


# ---------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------
def scaled(values):
    return values / values.max()


def plot_saliency(study, runs, profiles, peaks):
    levels = [0] + study.args.levels
    fig, axes = plt.subplots(len(levels), 1, figsize=(11, 3.2 * len(levels)), sharex=True, squeeze=False)
    for ax, level in zip(axes[:, 0], levels):
        ax.fill_between(range(len(profiles["snr_masked_value"])), scaled(profiles["snr_masked_value"]),
                        color="tab:red", alpha=0.25, label="SNR masked value")
        ax.fill_between(range(len(profiles["snr_mask"])), scaled(profiles["snr_mask"]), color="tab:blue", alpha=0.25,
                        label="SNR mask")
        for row in runs:
            ax.plot(scaled(profiles[f"{row['run']}_d{level}"]), lw=0.9, label=f"saliency {row['run']}")
        for name, peak in peaks.items():
            ax.axvline(peak, color="k", ls=":", lw=0.8)
        ax.set_ylabel("scaled to maximum")
        ax.set_title(f"CNN trained on desync 0, attack traces at desync {level}")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(os.path.join(study.folder, "saliency.png"), dpi=130)
    plt.close(fig)


def plot_occlusion(study, occlusions):
    runs = sorted({o["run"] for o in occlusions})
    cases = ["none", "peaks", "top", "random"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(runs), 1)
    for i, run in enumerate(runs):
        values = [next(o["ge_area"] for o in occlusions if o["run"] == run and o["occluded"] == c) for c in cases]
        ax.bar(np.arange(len(cases)) + i * width, values, width, label=run)
    ax.set_xticks(np.arange(len(cases)) + width * (len(runs) - 1) / 2)
    ax.set_xticklabels(cases)
    ax.set_ylabel("attack GE area (lower is better)")
    ax.set_xlabel("occluded samples")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(study.folder, "occlusion.png"), dpi=130)
    plt.close(fig)


def phase4_reference(args):
    # Attack metrics of the same runs in the Phase 4 study, to see whether the retraining reproduces them.
    try:
        folder = latest("phase4_", "eval.csv", args.phase4)
    except RuntimeError:
        return {}, None
    rows = phase4.read_csv(os.path.join(folder, "eval.csv"))
    return {r["run"]: r for r in rows if r["attack_desync"] == "0"}, folder


def write_summary(study, runs, overlaps, occlusions, peaks):
    args = study.args
    reference, reference_folder = phase4_reference(args)
    retrained = {e["run"]: e for e in study.evals if str(e["attack_desync"]) == "0"}
    lines = [
        f"# Phase 5: explainability ({datetime.now():%Y-%m-%d %H:%M})", "",
        f"Models: Phase 4 desync 0 recipe retrained with the same seed ({', '.join(r['run'] for r in runs)}). "
        f"Saliency = mean |d log p(correct class) / d sample| over {args.traces} attack traces. SNR peaks (Phase 1): "
        + ", ".join(f"{name} at sample {peak}" for name, peak in peaks.items()) + f"; windows of +-{args.window} "
        f"samples.", "",
        "## Retraining against Phase 4 (attack at desync 0)", "",
        *markdown_table([{"run": r["run"], "retrained_ge_area": retrained.get(r["run"], {}).get("att_ge_area", ""),
                          "phase4_ge_area": reference.get(r["run"], {}).get("att_ge_area", ""),
                          "retrained_traces_ge_lt_2": retrained.get(r["run"], {}).get("att_traces_ge_lt_2", ""),
                          "phase4_traces_ge_lt_2": reference.get(r["run"], {}).get("att_traces_ge_lt_2", "")}
                         for r in runs],
                        ["run", "retrained_ge_area", "phase4_ge_area", "retrained_traces_ge_lt_2",
                         "phase4_traces_ge_lt_2"]),
        "", f"Phase 4 reference: {reference_folder}. Small differences are expected: CPU training is not bit-exact.", "",
        "## Overlap of the saliency with the SNR peaks", "",
        *markdown_table(overlaps, list(overlaps[0])), "",
        "## Occlusion (attack at desync 0, mean over random draws for `random`)", "",
        *markdown_table(occlusions, list(occlusions[0])), "",
        "## Meaning", "",
    ]
    for run in sorted({o["run"] for o in occlusions}):
        by_case = {o["occluded"]: o for o in occlusions if o["run"] == run}
        area = {case: by_case[case]["ge_area"] for case in by_case}
        lines.append(f"- `{run}`: GE area without occlusion {area['none']}; occluding the SNR peaks {area['peaks']}, "
                     f"the same number of top-saliency samples {area['top']}, random samples {area['random']}.")
    lines += ["", "Files: `saliency.png`, `occlusion.png`, `saliency.npz`, `overlap.csv`, `occlusion.csv`, `models/`, "
              "`runs.csv`, `eval.csv`."]
    with open(os.path.join(study.folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 5: saliency of the CNN against the SNR peaks.")
    parser.add_argument("stage", choices=STAGES + ["all"])
    parser.add_argument("--study", help="existing study folder (default: a new phase5_<timestamp>/)")
    parser.add_argument("--phase1", help="Phase 1 folder with snr.npz (default: latest)")
    parser.add_argument("--phase3", help="Phase 3 study folder for the CNN (default: latest)")
    parser.add_argument("--phase4", help="Phase 4 study folder to compare the retraining with (default: latest)")
    parser.add_argument("--variants", nargs="+", default=["cnn"], choices=phase4.VARIANTS,
                        help="models to retrain and explain (default cnn)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="training seeds (default 1, as Phase 4)")
    parser.add_argument("--levels", type=int, nargs="*", default=[50, 100], choices=[50, 100],
                        help="desync levels to explain besides 0 (default 50 100; none = desync 0 only)")
    parser.add_argument("--traces", type=int, default=2000, help="attack traces for the saliency (default 2000)")
    parser.add_argument("--window", type=int, default=5, help="half width of the window around an SNR peak (default 5)")
    parser.add_argument("--random-draws", type=int, default=5, help="random occlusions to average (default 5)")
    parser.add_argument("--epochs", type=int, default=50, help="maximum epochs per run (default 50)")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience (default 10)")
    parser.add_argument("--aug-shift", type=int, default=100, help="cnn_aug: maximum shift")
    parser.add_argument("--aug-noise", type=float, default=0.1, help="cnn_aug: Gaussian noise, z-scored units")
    parser.add_argument("--subset-seed", type=int, default=0, help="attack subsets, as Phases 2-4 (default 0)")
    args = parser.parse_args()
    if args.stage == "saliency" and not args.study:
        parser.error("stage saliency needs --study <phase5 folder>")
    if min(args.traces, args.window, args.random_draws, args.epochs, args.patience, args.aug_shift) < 1:
        parser.error("--traces, --window, --random-draws, --epochs, --patience and --aug-shift must be >= 1")

    # The retraining reuses Phase 4's Study: train desync 0 only; the desync files are read in the stage saliency.
    train_args = argparse.Namespace(**vars(args))
    train_args.levels = []
    phase4.check_data(args.levels)
    folder = resume_run(args.study) if args.study else start_run("phase5")
    stages = STAGES if args.stage == "all" else [args.stage]
    study = phase4.Study(folder, train_args)
    for stage in stages:
        study.new_analysis_ids = []
        study.args = train_args if stage == "train" else args
        with log_to(os.path.join(folder, f"log_{stage}_{datetime.now():%Y%m%d_%H%M%S}.txt")):
            print(f"Phase 5, stage {stage}: {vars(args)}")
            print(f"Protocol: {PROTOCOL}")
            try:
                globals()[f"stage_{stage}"](study)
            except BaseException as e:
                finish_run(folder, f"phase5_{stage}", args.subset_seed, study.new_analysis_ids,
                           status=f"failed: {type(e).__name__}")
                raise
            failed = [r["run"] for r in study.rows if r["status"] != "ok"]
            status = "ok" if not failed else f"{len(failed)} failed: {' '.join(failed)}"
            print(f"\nStage {stage}: {status}. Results: {folder}")
            finish_run(folder, f"phase5_{stage}", args.subset_seed, study.new_analysis_ids, status=status)
            if stage != stages[-1]:
                resume_run(folder)


if __name__ == "__main__":
    main()
