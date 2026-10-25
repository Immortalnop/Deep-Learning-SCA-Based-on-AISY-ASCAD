# Phase 2: classic baselines (template attacks and CPA) under the Phase 0 GE protocol.
#
# Process: read profiling + attack traces (same split as AISY) -> SNR on the profiling traces gives the share
# leakage points (Phase 1 method) -> fit / run each attack -> key rank on the same 100 random attack subsets
# for every method -> GE and SR curves, comparable with the AISY runs of Phase 0.
#
# Methods (attacked byte b, key guess k, plaintext p):
#     template_pca    PCA on the whole trace, then a Gaussian template per class (QDA)
#     template_poi    PCA on windows around the share leakage points only, then QDA
#     cpa             first-order CPA, HW(Sbox(p ^ k)) on every sample: expected to fail (masked)
#     cpa_known_mask  first-order CPA, HW(Sbox(p ^ k) ^ r[b]) with the true mask from the metadata:
#                     not a real attack, it shows the masked share leaks on its own
#     cpa_2nd_order   CPA on the centred product of the two share points, HW(Sbox(p ^ k)): no mask needed
#
# Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.phase2_baselines
#     ..\aisy_env\Scripts\python.exe -m scripts.phase2_baselines --methods template_poi cpa_2nd_order
#     ..\aisy_env\Scripts\python.exe -m scripts.phase2_baselines --aisy-ids 31 33   (DL curves on the same axes)
#
# Output in resources/experiments/phase2_<timestamp>/:
#     ge.png               GE against attack traces for every method (+ AISY runs from --aisy-ids)
#     share_correlation.png  |correlation| per sample of HW(unmasked value) and HW(each share), profiling set
#     results.csv          one row per method: final GE, final SR, traces to GE < 2, seconds
#     curves.npz           GE and SR curves per method
#     summary.md           data, protocol, results table, what it means
#     log.txt              everything printed during the run
# One row per run is added to resources/experiments/index.csv.
import argparse
import csv
import os
import sqlite3
import time
from datetime import datetime

import h5py
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from aisy_sca.crypto.aes.aes128 import s_box
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from custom.custom_datasets.datasets import datasets_dict
from scripts.experiment_log import log_to
from scripts.experiment_output import finish_run, start_run
from scripts.phase0_compare import (DATABASE_FOLDER, DATABASE_NAME, GE_THRESHOLD, LABEL_BEST, LABEL_FINAL,
                                    PROTOCOL, read_curve, traces_to_threshold)
from scripts.phase1_snr import HW, MASK_R_OUT, ClassStats, target_values

METHODS = ["template_pca", "template_poi", "cpa", "cpa_known_mask", "cpa_2nd_order"]
SHARES = ["sbox_xor_r", "r", "sbox_xor_rout", "rout"]  # Phase 1 targets that leak
SHARE_PAIR = ("r", "sbox_xor_r")                       # the two shares combined by cpa_2nd_order
QDA_REG = 0.01  # covariance regularisation; 256 ID classes have only ~195 profiling traces each


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def load(path, byte):
    # Same traces as AISY: first n profiling traces; attack traces split in half (validation, attack)
    # when early stopping is on, so the attack set is the second half.
    n_prof, n_att = PROTOCOL["profiling_traces"], PROTOCOL["attack_traces"]
    with h5py.File(path, "r") as f:
        x_prof = f["Profiling_traces/traces"][:n_prof].astype(np.float32)
        meta_prof = f["Profiling_traces/metadata"][:n_prof]
        x_att = f["Attack_traces/traces"][n_att // 2:n_att].astype(np.float64)
        meta_att = f["Attack_traces/metadata"][n_att // 2:n_att]
    keys = np.unique(meta_att["key"][:, byte])
    if len(keys) != 1:
        raise ValueError(f"attack key byte {byte} is not fixed: {keys[:5]}")
    print(f"Profiling {x_prof.shape}, attack {x_att.shape} (attack traces {n_att // 2}-{n_att - 1}), "
          f"attack key byte {byte} = 0x{keys[0]:02X}")
    return x_prof, meta_prof, x_att, meta_att, int(keys[0])


def share_points(x_prof, meta_prof, byte, chunk=10000):
    # Peak SNR sample of every share (ID classes), as in Phase 1.
    stats = {t: ClassStats(256, x_prof.shape[1]) for t in SHARES}
    rng = np.random.default_rng(0)  # only feeds the unused "random" target
    for start in range(0, len(x_prof), chunk):
        values = target_values(meta_prof[start:start + chunk], byte, rng)
        x = x_prof[start:start + chunk].astype(np.float64)
        for t in SHARES:
            stats[t].add(x, values[t])
    points = {t: int(np.nanargmax(stats[t].snr())) for t in SHARES}
    print("Share leakage points (peak SNR sample): " + ", ".join(f"{t} {i}" for t, i in points.items()))
    return points


def guess_values(plaintext, byte):
    # Sbox(p ^ k) for every trace (rows) and every key guess k (columns).
    return s_box[plaintext[:, byte, None] ^ np.arange(256)[None, :]].astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# Key rank (same subsets for every method)
# ---------------------------------------------------------------------------------------------------------------
def additive_ranks(scores, key, subsets, interval):
    # AISY's key rank: add the log-probability of every key guess over the attack traces.
    ranks = []
    for subset in subsets:
        total = np.cumsum(scores[subset], axis=0)[interval - 1::interval]
        ranks.append((total > total[:, [key]]).sum(axis=1) + 1)
    return np.array(ranks)


def cpa_ranks(x, hyp, key, subsets, interval):
    # Pearson correlation of every key guess's hypothesis with every sample, updated every `interval` traces.
    # Score of a key guess = max |correlation| over the samples.
    ranks = []
    for subset in subsets:
        xs, hs = x[subset], hyp[subset].astype(np.float64)
        sx = np.zeros(x.shape[1]); sx2 = np.zeros(x.shape[1])
        sh = np.zeros(256); sh2 = np.zeros(256); sxh = np.zeros((256, x.shape[1]))
        row = []
        for end in range(interval, len(subset) + 1, interval):
            xb, hb = xs[end - interval:end], hs[end - interval:end]
            sx += xb.sum(axis=0); sx2 += (xb * xb).sum(axis=0)
            sh += hb.sum(axis=0); sh2 += (hb * hb).sum(axis=0); sxh += hb.T @ xb
            cov = end * sxh - np.outer(sh, sx)
            var = np.outer(end * sh2 - sh * sh, end * sx2 - sx * sx)
            corr = np.abs(cov) / np.sqrt(np.where(var > 0, var, np.inf))
            score = corr.max(axis=1)
            row.append((score > score[key]).sum() + 1)
        ranks.append(row)
    return np.array(ranks)


# ---------------------------------------------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------------------------------------------
def template_scores(x_prof, labels_prof, x_att, att_labels, components, seed):
    # PCA -> one Gaussian per class with its own covariance (QDA, uniform priors) -> log-probability of the
    # class each key guess predicts. QDA's per-class covariance can model the share-share dependence,
    # so a template attack on both share points can defeat first-order masking.
    pca = PCA(n_components=min(components, x_prof.shape[1]), random_state=seed).fit(x_prof)
    classes = int(labels_prof.max()) + 1
    qda = QuadraticDiscriminantAnalysis(priors=np.full(classes, 1 / classes), reg_param=QDA_REG)
    qda.fit(pca.transform(x_prof), labels_prof)
    log_lik = qda.decision_function(pca.transform(x_att))
    log_post = log_lik - logsumexp(log_lik, axis=1, keepdims=True)
    return np.take_along_axis(log_post, att_labels.astype(np.int64), axis=1)


def poi_window(points, width, samples):
    idx = sorted({i for p in points.values() for i in range(p - width, p + width + 1) if 0 <= i < samples})
    return np.array(idx)


def run_method(method, data, args, subsets):
    x_prof, meta_prof, x_att, meta_att, key, points = data
    interval = PROTOCOL["key_rank_report_interval"]
    labels = guess_values(meta_att["plaintext"], args.byte)               # (attack traces, 256)
    if args.leakage_model == "HW":
        labels = HW[labels]
    prof_labels = s_box[meta_prof["plaintext"][:, args.byte] ^ meta_prof["key"][:, args.byte]].astype(np.uint8)
    if args.leakage_model == "HW":
        prof_labels = HW[prof_labels]

    if method == "template_pca":
        scores = template_scores(x_prof, prof_labels, x_att, labels, args.components, args.seed)
        return additive_ranks(scores, key, subsets, interval)
    if method == "template_poi":
        idx = poi_window(points, args.poi_width, x_prof.shape[1])
        print(f"template_poi: {len(idx)} samples around {sorted(set(points.values()))}")
        scores = template_scores(x_prof[:, idx], prof_labels, x_att[:, idx], labels, args.components, args.seed)
        return additive_ranks(scores, key, subsets, interval)

    hyp_unmasked = HW[guess_values(meta_att["plaintext"], args.byte)]
    if method == "cpa":
        return cpa_ranks(x_att, hyp_unmasked, key, subsets, interval)
    if method == "cpa_known_mask":
        masked = guess_values(meta_att["plaintext"], args.byte) ^ meta_att["masks"][:, args.byte, None]
        return cpa_ranks(x_att, HW[masked], key, subsets, interval)
    if method == "cpa_2nd_order":
        a, b = (points[t] for t in SHARE_PAIR)
        mean = x_prof[:, [a, b]].mean(axis=0)  # centre with the profiling means
        product = ((x_att[:, a] - mean[0]) * (x_att[:, b] - mean[1]))[:, None]
        print(f"cpa_2nd_order: centred product of samples {a} ({SHARE_PAIR[0]}) and {b} ({SHARE_PAIR[1]})")
        return cpa_ranks(product, hyp_unmasked, key, subsets, interval)
    raise ValueError(method)


# ---------------------------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------------------------
def aisy_curves(analysis_ids):
    # GE curves of AISY runs (best epoch if early stopping was on, else last epoch).
    curves = {}
    connection = sqlite3.connect(DATABASE_FOLDER + DATABASE_NAME)
    for analysis_id in analysis_ids:
        for label in (LABEL_BEST, LABEL_FINAL):
            ge, interval = read_curve(connection, "guessing_entropy", analysis_id, label)
            if ge is not None:
                curves[f"AISY {analysis_id} ({'best' if label == LABEL_BEST else 'last'} epoch)"] = (ge, interval)
                break
        else:
            print(f"AISY analysis {analysis_id}: no GE curve in the database, skipped")
    connection.close()
    return curves


def plot_ge(path, results, dl_curves, args):
    interval = PROTOCOL["key_rank_report_interval"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, r in results.items():
        x = interval * np.arange(1, len(r["ge"]) + 1)
        ax.plot(x, r["ge"], label=method, linewidth=1.2)
    for name, (ge, dl_interval) in dl_curves.items():
        ax.plot(dl_interval * np.arange(1, len(ge) + 1), ge, label=name, linewidth=1.2, linestyle="--")
    ax.axhline(GE_THRESHOLD, color="gray", linewidth=0.7, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 256)
    ax.set_xlabel("attack traces")
    ax.set_ylabel("guessing entropy (rank of the correct key)")
    ax.set_title(f"{args.dataset}, byte {args.byte}: GE over {PROTOCOL['key_rank_executions']} attack subsets "
                 f"(dotted: GE = {GE_THRESHOLD})", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_share_correlation(path, x_prof, meta_prof, points, byte):
    # Pearson correlation per sample of HW(target) with the profiling traces (true key and masks).
    values = target_values(meta_prof, byte, np.random.default_rng(0))
    targets = ["sbox"] + SHARES
    n = len(x_prof)
    x_std = x_prof.std(axis=0, dtype=np.float64)
    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 1.8 * len(targets)), sharex=True)
    for ax, t in zip(axes, targets):
        h = HW[values[t]].astype(np.float64)
        h -= h.mean()
        corr = (h @ x_prof) / n / (h.std() * np.where(x_std > 0, x_std, np.inf))
        ax.plot(np.abs(corr), linewidth=0.7)
        if t in points:
            ax.axvline(points[t], color="tab:red", linewidth=0.7, linestyle="--")
        ax.set_title(f"HW({t})" + ("  (unmasked)" if t == "sbox" else ""), fontsize=9, loc="left")
        ax.set_ylabel("|corr|", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("sample", fontsize=8)
    fig.suptitle(f"First-order correlation, {n} profiling traces, byte {byte}. Red: SNR peak of the share.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(folder, results, points, args, key):
    interval = PROTOCOL["key_rank_report_interval"]
    np.savez(os.path.join(folder, "curves.npz"),
             **{f"{m}_{k}": r[k] for m, r in results.items() for k in ("ge", "sr")})
    rows = []
    for method, r in results.items():
        rows.append({"method": method, "final_ge": round(float(r["ge"][-1]), 2),
                     "final_sr": round(float(r["sr"][-1]), 2),
                     "traces_ge_lt_2": traces_to_threshold(list(r["ge"]), interval), "seconds": r["seconds"]})
    with open(os.path.join(folder, "results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Phase 2: classic baselines ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Data: {args.dataset}, byte {args.byte} (attack key 0x{key:02X}), {PROTOCOL['profiling_traces']} profiling "
        f"traces, attack traces {PROTOCOL['attack_traces'] // 2}-{PROTOCOL['attack_traces'] - 1} (the AISY attack "
        f"set when early stopping uses the first half for validation).",
        f"GE: mean key rank over {PROTOCOL['key_rank_executions']} random subsets of "
        f"{PROTOCOL['key_rank_attack_traces']} attack traces (same subsets for every method, seed {args.seed}), "
        f"every {interval} traces. Same protocol as Phase 0.",
        f"Templates: {args.leakage_model} classes, PCA {args.components} components, QDA (uniform priors, "
        f"reg_param {QDA_REG}). POI windows: +/- {args.poi_width} samples around "
        + ", ".join(f"{t} {i}" for t, i in points.items()) + ".",
        "",
        "| Method | Final GE | Final SR | Traces to GE < 2 | Seconds |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        reached = row["traces_ge_lt_2"] if row["traces_ge_lt_2"] is not None else "not reached"
        lines.append(f"| {row['method']} | {row['final_ge']} | {row['final_sr']} | {reached} | {row['seconds']} |")
    lines += ["", "## Meaning", ""]
    found = {row["method"]: row["traces_ge_lt_2"] is not None for row in rows}
    notes = {
        "cpa": ("First-order CPA on the masked value fails, as expected: masking removes first-order leakage.",
                "First-order CPA reaches GE < 2: check the masking (unexpected)."),
        "cpa_known_mask": ("With the true mask, first-order CPA recovers the key: the masked share leaks on its own.",
                           "Even with the true mask, CPA does not reach GE < 2 in this trace budget."),
        "cpa_2nd_order": ("Second-order CPA (centred product of the two share points) recovers the key without "
                          "the mask.", "Second-order CPA on one pair of points does not reach GE < 2 in this "
                          "trace budget."),
        "template_pca": ("Whole-trace PCA templates recover the key.",
                         "Whole-trace PCA templates do not: the leading components miss the share points."),
        "template_poi": ("Templates on the share points recover the key: QDA models the share dependence.",
                         "Templates on the share points do not reach GE < 2 in this trace budget."),
    }
    for method in results:
        lines.append(f"- {notes[method][0] if found[method] else notes[method][1]}")
    lines += ["", "Files: `ge.png`, `share_correlation.png`, `results.csv`, `curves.npz` (keys `<method>_ge`, "
                  "`<method>_sr`)."]
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 2: template attacks and CPA under the Phase 0 protocol.")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--dataset", default=PROTOCOL["dataset"], choices=list(datasets_dict))
    parser.add_argument("--byte", type=int, default=PROTOCOL["byte"], help="AES key byte (default 2)")
    parser.add_argument("--leakage-model", default="ID", choices=["ID", "HW"],
                        help="template classes: ID = 256 values, HW = 9 Hamming weights (default ID)")
    parser.add_argument("--components", type=int, default=20, help="PCA components for templates (default 20)")
    parser.add_argument("--poi-width", type=int, default=5, help="samples on each side of a share point (default 5)")
    parser.add_argument("--seed", type=int, default=0, help="seed for attack subsets and PCA (default 0)")
    parser.add_argument("--aisy-ids", type=int, nargs="+", default=[],
                        help="AISY analysis ids whose GE curves are drawn on the same axes")
    parser.add_argument("--data-folder", default="data/", help="folder with the .h5 files (default data/)")
    args = parser.parse_args()

    path = os.path.join(args.data_folder, datasets_dict[args.dataset]["filename"])
    if not os.path.isfile(path):
        parser.error(f"dataset file not found: {path}")

    folder = start_run("phase2")
    with log_to(os.path.join(folder, "log.txt")):
        print(f"Phase 2: {vars(args)}")
        print(f"Protocol: {PROTOCOL}")
        x_prof, meta_prof, x_att, meta_att, key = load(path, args.byte)
        points = share_points(x_prof, meta_prof, args.byte)
        plot_share_correlation(os.path.join(folder, "share_correlation.png"), x_prof, meta_prof, points, args.byte)

        rng = np.random.default_rng(args.seed)
        subsets = [rng.choice(len(x_att), PROTOCOL["key_rank_attack_traces"], replace=False)
                   for _ in range(PROTOCOL["key_rank_executions"])]
        data = (x_prof, meta_prof, x_att, meta_att, key, points)
        results = {}
        for method in args.methods:
            print(f"\n===== {method} =====")
            start = time.time()
            ranks = run_method(method, data, args, subsets)
            results[method] = {"ge": ranks.mean(axis=0), "sr": (ranks == 1).mean(axis=0),
                               "seconds": round(time.time() - start, 1)}
            print(f"{method}: final GE {results[method]['ge'][-1]:.2f}, SR {results[method]['sr'][-1]:.2f}, "
                  f"{results[method]['seconds']} s")

        plot_ge(os.path.join(folder, "ge.png"), results, aisy_curves(args.aisy_ids), args)
        write_outputs(folder, results, points, args, key)
        print(f"\nResults: {folder}")
        finish_run(folder, "phase2", args.seed)


if __name__ == "__main__":
    main()
