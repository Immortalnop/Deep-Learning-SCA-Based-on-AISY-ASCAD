# Phase 2: classic baselines (template attacks and CPA) under the Phase 0 GE protocol.
#
# Process: read profiling + attack traces (same split and sample window as AISY, scripts/sca_protocol.py) ->
# one pass over the profiling traces gives the share leakage points (Phase 1 SNR method) and the first-order
# correlation per sample -> fit / run each attack -> key rank on the same 100 random attack subsets for every
# method -> GE and SR curves, comparable with the AISY runs of Phase 0 and Phase 3.
#
# Methods (attacked byte b, key guess k, plaintext p):
#     template_pca    PCA on the whole trace, then a Gaussian template per class (QDA)
#     template_poi    PCA on windows around the share leakage points only, then QDA
#     cpa             first-order CPA, HW(Sbox(p ^ k)) on every sample: expected to fail (masked)
#     cpa_known_mask  first-order CPA, HW(Sbox(p ^ k) ^ r[b]) with the true mask from the metadata:
#                     not a real attack, it shows the masked share leaks on its own
#     cpa_2nd_order   CPA of HW(Sbox(p ^ k)) on the centred products of every sample pair in the windows
#                     around the two shares of each pair: no mask needed
# Templates run once per class model in --leakage-models (ID = 256 values, HW = 9 Hamming weights;
# Phase 0 trains the networks on HW). CPA always uses the HW hypothesis.
#
# Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.phase2_baselines
#     ..\aisy_env\Scripts\python.exe -m scripts.phase2_baselines --methods template_poi cpa_2nd_order
#     ..\aisy_env\Scripts\python.exe -m scripts.phase2_baselines --aisy-ids 31 33   (DL curves on the same axes)
#
# Output in resources/experiments/phase2_<timestamp>/:
#     ge.png                 GE against attack traces for every attack (+ AISY runs from --aisy-ids)
#     share_correlation.png  |correlation| per sample of HW(unmasked value) and HW(each share), profiling set
#     results.csv            one row per attack: status, final GE (+ std over subsets), final SR,
#                            traces to GE < 2, seconds
#     curves.npz             GE and SR curves per attack (keys <attack>_ge, <attack>_sr, plus interval)
#     summary.md             data, protocol, results table, what it means
#     log.txt                everything printed during the run
# One row per run is added to resources/experiments/index.csv.
import argparse
import csv
import os
import sqlite3
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from custom.custom_datasets.datasets import datasets_dict
from scripts.experiment_log import log_to
from scripts.experiment_output import finish_run, start_run
from scripts.phase1_snr import PEAK_MIN_DISTANCE, SIGNIFICANT, ClassStats, find_peaks, target_values
from scripts.sca_protocol import (DATABASE_FOLDER, DATABASE_NAME, GE_THRESHOLD, HW, LABEL_BEST, LABEL_FINAL,
                                  LOG_FLOOR, PROTOCOL, S_BOX, attack_subsets, ge_sr, guess_classes, key_ranks, load_split,
                                  rank_of, read_curve, traces_to_threshold)

METHODS = ["template_pca", "template_poi", "cpa", "cpa_known_mask", "cpa_2nd_order"]
TEMPLATE_METHODS = ("template_pca", "template_poi")
SHARES = ["sbox_xor_r", "r", "sbox_xor_rout", "rout"]       # Phase 1 targets that leak
SHARE_PAIRS = [("r", "sbox_xor_r"), ("rout", "sbox_xor_rout")]  # combined by cpa_2nd_order
CORRELATION_TARGETS = ["sbox"] + SHARES
CLASSES = {"ID": 256, "HW": 9}
PEAKS_PER_SHARE = 2
QDA_REG = 0.01  # covariance regularisation; 256 ID classes have only ~195 profiling traces each


# ---------------------------------------------------------------------------------------------------------------
# Leakage (profiling set, true key and masks)
# ---------------------------------------------------------------------------------------------------------------
def leakage_pass(x_prof, meta_prof, byte, chunk=10000):
    # One pass over the profiling traces, in float64 chunks to keep memory low:
    #   - SNR per share (ID classes, Phase 1 method) -> share leakage points: the top PEAKS_PER_SHARE peaks
    #     above SIGNIFICANT x the noise floor (random labels), top peak first. Templates use every peak
    #     (PCA + QDA weigh them); 2nd-order CPA uses only the top peak, because every extra product column is
    #     one more chance for a wrong key guess to get the max |correlation|.
    #   - Pearson correlation per sample of HW(target) for the share_correlation plot
    samples = x_prof.shape[1]
    stats = {t: ClassStats(256, samples) for t in SHARES + ["random"]}
    n, sx, sx2 = 0, np.zeros(samples), np.zeros(samples)
    sh = dict.fromkeys(CORRELATION_TARGETS, 0.0)
    sh2 = dict.fromkeys(CORRELATION_TARGETS, 0.0)
    sxh = {t: np.zeros(samples) for t in CORRELATION_TARGETS}
    rng = np.random.default_rng(0)  # random labels for the noise floor
    for start in range(0, len(x_prof), chunk):
        values = target_values(meta_prof[start:start + chunk], byte, rng)
        x = x_prof[start:start + chunk].astype(np.float64)
        n += len(x)
        sx += x.sum(axis=0)
        sx2 += (x * x).sum(axis=0)
        for t in stats:
            stats[t].add(x, values[t])
        for t in CORRELATION_TARGETS:
            h = HW[values[t]].astype(np.float64)
            sh[t] += h.sum()
            sh2[t] += h @ h
            sxh[t] += h @ x

    floor = np.nanmax(stats["random"].snr())
    points = {}
    for t in SHARES:
        snr = stats[t].snr()
        peaks = find_peaks(snr, PEAKS_PER_SHARE, PEAK_MIN_DISTANCE)
        points[t] = [peaks[0]] + [i for i in peaks[1:] if snr[i] > SIGNIFICANT * floor]  # top peak always kept
    var_x = n * sx2 - sx * sx
    correlation = {t: (n * sxh[t] - sh[t] * sx) / np.sqrt((n * sh2[t] - sh[t] ** 2) * np.where(var_x > 0, var_x, np.inf))
                   for t in CORRELATION_TARGETS}
    print(f"Share leakage points (SNR peaks > {SIGNIFICANT:g} x noise floor {floor:.5f}): {format_points(points)}")
    return points, correlation


def format_points(points):
    return ", ".join(f"{t} {'/'.join(str(i) for i in peaks)}" for t, peaks in points.items())


def window(point, width, samples):
    return np.arange(max(point - width, 0), min(point + width + 1, samples))


def classes_of(values, leakage_model):
    return HW[values] if leakage_model == "HW" else values


# ---------------------------------------------------------------------------------------------------------------
# Attacks. Each returns key ranks of shape (subsets, report points).
# ---------------------------------------------------------------------------------------------------------------
def template_ranks(x_prof, prof_classes, x_att, att_classes, leakage_model, components, seed, key, subsets):
    # PCA -> one Gaussian per class with its own covariance (QDA, uniform priors, so the score is the
    # likelihood of the trace under each class) -> log-probability of the class each key guess predicts.
    # QDA's per-class covariance can model the share-share dependence, so templates on both share points
    # can defeat first-order masking.
    pca = PCA(n_components=min(components, x_prof.shape[1]), random_state=seed).fit(x_prof)
    classes = CLASSES[leakage_model]
    present = np.unique(prof_classes)
    qda = QuadraticDiscriminantAnalysis(priors=np.full(len(present), 1 / len(present)), reg_param=QDA_REG)
    qda.fit(pca.transform(x_prof), prof_classes)
    log_lik = qda.decision_function(pca.transform(x_att))
    # Every class gets a column; a class without profiling traces gets AISY's floor probability.
    log_post = np.full((len(x_att), classes), LOG_FLOOR)
    log_post[:, qda.classes_] = log_lik - logsumexp(log_lik, axis=1, keepdims=True)
    scores = np.take_along_axis(log_post, att_classes.astype(np.int64), axis=1)
    return key_ranks(scores, key, subsets)


def cpa_ranks(x, hypothesis, key, subsets, interval=PROTOCOL["key_rank_report_interval"]):
    # Pearson correlation of every key guess's hypothesis with every column of x, updated every
    # `interval` traces from running sums. Score of a key guess = max |correlation| over the columns.
    # The variance terms are per row (key guess) and per column, so only the covariance is a full
    # (256, columns) matrix; it reuses one buffer.
    def inverse_std(sum_, sum_sq, n):
        var = n * sum_sq - sum_ * sum_
        return 1 / np.sqrt(np.where(var > 0, var, np.inf))

    ranks = []
    cov = np.empty((256, x.shape[1]))
    for subset in subsets:
        xs, hs = x[subset], hypothesis[subset].astype(np.float64)
        sx, sx2 = np.zeros(x.shape[1]), np.zeros(x.shape[1])
        sh, sh2, sxh = np.zeros(256), np.zeros(256), np.zeros((256, x.shape[1]))
        row = []
        for end in range(interval, len(subset) + 1, interval):
            xb, hb = xs[end - interval:end], hs[end - interval:end]
            sx += xb.sum(axis=0)
            sx2 += (xb * xb).sum(axis=0)
            sh += hb.sum(axis=0)
            sh2 += (hb * hb).sum(axis=0)
            sxh += hb.T @ xb
            np.multiply(sxh, end, out=cov)
            cov -= np.outer(sh, sx)
            np.abs(cov, out=cov)
            cov *= inverse_std(sx, sx2, end)
            row.append(rank_of(cov.max(axis=1) * inverse_std(sh, sh2, end), key))
        ranks.append(row)
    return np.array(ranks)


def centred_products(x_prof, x_att, points, width):
    # Second-order preprocessing (Prouff et al., 2009): centred product of every sample pair in the windows
    # around the top peaks of the two shares of each pair, centred with the profiling means. Windows
    # instead of single points keep the attack working when a peak is a few samples off (other datasets,
    # misaligned traces).
    mean = x_prof.mean(axis=0, dtype=np.float64)
    columns = []
    for share_1, share_2 in SHARE_PAIRS:
        p1, p2 = points[share_1][0], points[share_2][0]
        w1, w2 = window(p1, width, x_att.shape[1]), window(p2, width, x_att.shape[1])
        a, b = x_att[:, w1] - mean[w1], x_att[:, w2] - mean[w2]
        columns.append((a[:, :, None] * b[:, None, :]).reshape(len(x_att), -1))
        print(f"cpa_2nd_order: {len(w1)} x {len(w2)} products around {share_1} {p1} and {share_2} {p2}")
    return np.hstack(columns)


def run_attack(method, leakage_model, split, points, args, subsets):
    x_prof, meta_prof = split["x_profiling"], split["meta_profiling"]
    x_att, meta_att, key = split["x_attack"], split["meta_attack"], split["key"]
    guesses = guess_classes(meta_att["plaintext"], args.byte, "ID")  # Sbox(p ^ k), (attack traces, 256)

    if method in TEMPLATE_METHODS:
        prof_values = S_BOX[meta_prof["plaintext"][:, args.byte] ^ meta_prof["key"][:, args.byte]]
        if method == "template_poi":
            peaks = sorted({p for share_peaks in points.values() for p in share_peaks})
            idx = np.unique(np.concatenate([window(p, args.poi_width, x_prof.shape[1]) for p in peaks]))
            print(f"template_poi: {len(idx)} samples around {peaks}")
            x_prof, x_att = x_prof[:, idx], x_att[:, idx]
        return template_ranks(x_prof, classes_of(prof_values, leakage_model), x_att,
                              classes_of(guesses, leakage_model), leakage_model, args.components, args.seed,
                              key, subsets)
    if method == "cpa":
        return cpa_ranks(x_att, HW[guesses], key, subsets)
    if method == "cpa_known_mask":
        return cpa_ranks(x_att, HW[guesses ^ meta_att["masks"][:, args.byte, None]], key, subsets)
    if method == "cpa_2nd_order":
        return cpa_ranks(centred_products(x_prof, x_att, points, args.poi_width), HW[guesses], key, subsets)
    raise ValueError(method)


def attacks(args):
    # (name, method, leakage model) for every attack to run. CPA has no class model choice (HW hypothesis).
    return [(f"{m}_{lm}", m, lm) for m in args.methods for lm in
            (args.leakage_models if m in TEMPLATE_METHODS else ["HW"])]


# ---------------------------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------------------------
def aisy_curves(analysis_ids):
    # GE curves of AISY runs (best epoch if early stopping was on, else last epoch).
    curves = {}
    if not analysis_ids:
        return curves
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
    for name, r in results.items():
        if r["status"] == "ok":
            ax.plot(interval * np.arange(1, len(r["ge"]) + 1), r["ge"], label=name, linewidth=1.2)
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


def plot_share_correlation(path, correlation, points, byte, n):
    fig, axes = plt.subplots(len(CORRELATION_TARGETS), 1, figsize=(11, 1.8 * len(CORRELATION_TARGETS)), sharex=True)
    for ax, t in zip(axes, CORRELATION_TARGETS):
        ax.plot(np.abs(correlation[t]), linewidth=0.7)
        for i in points.get(t, []):
            ax.axvline(i, color="tab:red", linewidth=0.7, linestyle="--")
        ax.set_title(f"HW({t})" + ("  (unmasked)" if t == "sbox" else ""), fontsize=9, loc="left")
        ax.set_ylabel("|corr|", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("sample", fontsize=8)
    fig.suptitle(f"First-order correlation, {n} profiling traces, byte {byte}. Red: SNR peaks of the share.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


NOTES = {  # (reaches GE < 2, does not)
    "cpa": ("First-order CPA reaches GE < 2: check the masking (unexpected).",
            "First-order CPA on the unmasked value fails, as expected: masking removes first-order leakage."),
    "cpa_known_mask": ("With the true mask, first-order CPA recovers the key: the masked share leaks on its own.",
                       "Even with the true mask, CPA does not reach GE < 2 in this trace budget."),
    "cpa_2nd_order": ("Second-order CPA (centred products around the share points) recovers the key without "
                      "the mask.",
                      "Second-order CPA around the share points does not reach GE < 2 in this trace budget; "
                      "compare the median rank: a few failed subsets can hold the mean up."),
    "template_pca": ("Whole-trace PCA templates recover the key.",
                     "Whole-trace PCA templates do not: the leading components follow the largest variance, "
                     "not the share points."),
    "template_poi": ("Templates on the share points recover the key: QDA models the share dependence.",
                     "Templates on the share points do not reach GE < 2 in this trace budget."),
}


def write_outputs(folder, results, points, args, key, notes):
    interval = PROTOCOL["key_rank_report_interval"]
    ok = {name: r for name, r in results.items() if r["status"] == "ok"}
    np.savez(os.path.join(folder, "curves.npz"), interval=interval,
             **{f"{name}_{k}": r[k] for name, r in ok.items() for k in ("ge", "sr")})
    rows = []
    for name, r in results.items():
        row = {"attack": name, "method": r["method"], "leakage_model": r["leakage_model"], "status": r["status"],
               "final_ge": None, "final_ge_std": None, "final_median_rank": None, "final_sr": None,
               "traces_ge_lt_2": None, "seconds": r["seconds"], "error": r.get("error", "")}
        if r["status"] == "ok":
            row.update(final_ge=round(float(r["ge"][-1]), 2), final_ge_std=round(float(r["ge_std"]), 2),
                       final_median_rank=float(r["median_rank"]), final_sr=round(float(r["sr"][-1]), 2),
                       traces_ge_lt_2=traces_to_threshold(list(r["ge"]), interval))
        rows.append(row)
    with open(os.path.join(folder, "results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Phase 2: classic baselines ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Data: {args.dataset}, byte {args.byte} (attack key 0x{key:02X}), {PROTOCOL['profiling_traces']} profiling "
        f"traces, attack traces {PROTOCOL['attack_traces'] // 2}-{PROTOCOL['attack_traces'] - 1} (the AISY attack "
        f"set when early stopping uses the first half for validation), AISY sample window.",
        f"GE: mean key rank over {PROTOCOL['key_rank_executions']} random subsets of "
        f"{PROTOCOL['key_rank_attack_traces']} attack traces (same subsets for every attack, seed {args.seed}), "
        f"every {interval} traces; tied key guesses count half. Same protocol as Phase 0.",
        f"Templates: {' and '.join(args.leakage_models)} classes, PCA {args.components} components, QDA (uniform "
        f"priors, reg_param {QDA_REG}). Windows: +/- {args.poi_width} samples around the share SNR peaks "
        f"{format_points(points)} (templates: all peaks; 2nd-order CPA: top peak of each share). "
        f"CPA: HW hypothesis.",
        f"Phase 0 trains the networks on {PROTOCOL['leakage_model']} classes: compare them with the "
        f"`_{PROTOCOL['leakage_model']}` template rows.",
        *notes,
        "",
        "| Attack | Final GE (± std over subsets) | Final median rank | Final SR | Traces to GE < 2 | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['attack']} | failed: {row['error']} | | | | {row['seconds']} |")
            continue
        reached = row["traces_ge_lt_2"] if row["traces_ge_lt_2"] is not None else "not reached"
        lines.append(f"| {row['attack']} | {row['final_ge']} ± {row['final_ge_std']} | {row['final_median_rank']:g} "
                     f"| {row['final_sr']} | {reached} | {row['seconds']} |")

    lines += ["", "## Meaning", ""]
    for row in rows:
        if row["status"] == "ok":
            note = NOTES[row["method"]][0 if row["traces_ge_lt_2"] is not None else 1]
            lines.append(f"- `{row['attack']}`: {note}")
    found = [r for r in rows if r["status"] == "ok" and r["traces_ge_lt_2"] is not None
             and r["method"] not in ("cpa_known_mask",)]  # known mask is not a real attack
    if found:
        best = min(found, key=lambda r: r["traces_ge_lt_2"])
        lines.append(f"- Best real classic attack: `{best['attack']}`, GE < 2 after {best['traces_ge_lt_2']} traces. "
                     f"A network in Phase 3 should beat this to be worth its cost.")
    else:
        lines.append("- No real classic attack reaches GE < 2 in this trace budget.")
    lines += ["", "Files: `ge.png`, `share_correlation.png`, `results.csv`, `curves.npz` (keys `<attack>_ge`, "
                  "`<attack>_sr`, `interval`)."]
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def run(args, folder):
    print(f"Phase 2: {vars(args)}")
    print(f"Protocol: {PROTOCOL}")
    split = load_split(args.data_folder, datasets_dict[args.dataset], args.byte)
    notes = []
    if not split["meta_profiling"]["masks"][:, args.byte].any():
        notes.append(f"**NOTE:** mask r[{args.byte}] is always 0 (byte not masked): share points, "
                     f"`cpa_known_mask` and `cpa_2nd_order` are not meaningful.")
        print(notes[-1])
    points, correlation = leakage_pass(split["x_profiling"], split["meta_profiling"], args.byte)
    plot_share_correlation(os.path.join(folder, "share_correlation.png"), correlation, points, args.byte,
                           len(split["x_profiling"]))

    subsets = attack_subsets(len(split["x_attack"]), args.seed)
    results = {}
    for name, method, leakage_model in attacks(args):
        print(f"\n===== {name} =====")
        start = time.time()
        result = {"method": method, "leakage_model": leakage_model}
        try:
            ranks = run_attack(method, leakage_model, split, points, args, subsets)
            ge, sr = ge_sr(ranks)
            result.update(status="ok", ge=ge, sr=sr, ge_std=ranks[:, -1].std(), median_rank=np.median(ranks[:, -1]))
            print(f"{name}: final GE {ge[-1]:.2f} ± {ranks[:, -1].std():.2f} (median rank "
                  f"{result['median_rank']:g}), SR {sr[-1]:.2f}", end="")
        except Exception as e:  # one failed attack must not lose the others
            traceback.print_exc()
            result.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
            print(f"{name}: failed", end="")
        result["seconds"] = round(time.time() - start, 1)
        print(f", {result['seconds']} s")
        results[name] = result

    plot_ge(os.path.join(folder, "ge.png"), results, aisy_curves(args.aisy_ids), args)
    write_outputs(folder, results, points, args, split["key"], notes)
    print(f"\nResults: {folder}")
    failed = sum(r["status"] != "ok" for r in results.values())
    return "ok" if not failed else f"{len(results) - failed}/{len(results)} ok"


def main():
    parser = argparse.ArgumentParser(description="Phase 2: template attacks and CPA under the Phase 0 protocol.")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--dataset", default=PROTOCOL["dataset"], choices=list(datasets_dict))
    parser.add_argument("--byte", type=int, default=PROTOCOL["byte"], help="AES key byte (default 2)")
    parser.add_argument("--leakage-models", nargs="+", default=["ID", "HW"], choices=list(CLASSES),
                        help="template classes: ID = 256 values, HW = 9 Hamming weights (default both)")
    parser.add_argument("--components", type=int, default=20, help="PCA components for templates (default 20)")
    parser.add_argument("--poi-width", type=int, default=5,
                        help="samples on each side of a share point, templates and 2nd-order CPA (default 5)")
    parser.add_argument("--seed", type=int, default=0, help="seed for attack subsets and PCA (default 0)")
    parser.add_argument("--aisy-ids", type=int, nargs="+", default=[],
                        help="AISY analysis ids whose GE curves are drawn on the same axes")
    parser.add_argument("--data-folder", default="data/", help="folder with the .h5 files (default data/)")
    args = parser.parse_args()
    if not 0 <= args.byte < 16:
        parser.error("--byte must be 0-15")
    if args.components < 1 or args.poi_width < 0:
        parser.error("--components must be >= 1 and --poi-width >= 0")
    path = os.path.join(args.data_folder, datasets_dict[args.dataset]["filename"])
    if not os.path.isfile(path):
        parser.error(f"dataset file not found: {path}")

    folder = start_run("phase2")
    with log_to(os.path.join(folder, "log.txt")):
        try:
            status = run(args, folder)
        except BaseException as e:
            finish_run(folder, "phase2", args.seed, status=f"failed: {type(e).__name__}")
            raise
        finish_run(folder, "phase2", args.seed, status=status)


if __name__ == "__main__":
    main()
