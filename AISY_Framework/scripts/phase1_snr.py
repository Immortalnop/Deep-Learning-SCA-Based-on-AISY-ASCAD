# Phase 1: leakage analysis. Signal-to-noise ratio (SNR) per sample, before training any network.
#
# Process: read profiling traces + metadata (plaintext, key, masks) in chunks -> compute each target
# value per trace -> per-class mean and variance per sample -> SNR = Var(class means) / Mean(class variances).
#
# Targets for the attacked byte b (ASCAD v1 masking: masks[0..15] = r[0..15], masks[16] = r_in, masks[17] = r_out):
#     sbox          Sbox(p[b] ^ k[b])            unmasked value: no first-order leakage expected
#     sbox_xor_r    Sbox(p[b] ^ k[b]) ^ r[b]     masked value (share 1)
#     r             r[b]                         mask (share 2)
#     sbox_xor_rout Sbox(p[b] ^ k[b]) ^ r_out    masked S-box output (share 1, other masking step)
#     rout          r_out                        output mask (share 2)
#     random        uniform random labels        noise floor: SNR with no relation to the traces
#
# SNR uses the full trace width in the file, so it also shows whether the AISY sample window
# (first_sample, number_of_samples in custom/custom_datasets/datasets.py) covers the leakage.
#
# Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.phase1_snr
#     ..\aisy_env\Scripts\python.exe -m scripts.phase1_snr --traces 200000 --leakage-model HW
#
# Output in resources/experiments/phase1_<timestamp>/:
#     snr.png             SNR per sample for every target, AISY window shaded
#     snr_vs_traces.png   peak SNR of every target against the number of profiling traces
#     snr.npz             SNR arrays (full traces and every profiling-set size)
#     peaks.csv           top peaks per target
#     summary.md          checks, peak table and what it means for the CNN design
#     log.txt             everything printed during the run
# One row per run is added to resources/experiments/index.csv.
import argparse
import csv
import os
from datetime import datetime

import h5py
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from custom.custom_datasets.datasets import datasets_dict
from scripts.experiment_log import log_to
from scripts.experiment_output import finish_run, start_run
from scripts.sca_protocol import HW, S_BOX

TARGETS = ["sbox", "sbox_xor_r", "r", "sbox_xor_rout", "rout", "random"]
TITLES = {
    "sbox": "Sbox(p ^ k)  (unmasked)",
    "sbox_xor_r": "Sbox(p ^ k) ^ r  (masked value)",
    "r": "r  (mask)",
    "sbox_xor_rout": "Sbox(p ^ k) ^ r_out  (masked S-box output)",
    "rout": "r_out  (output mask)",
    "random": "random labels  (noise floor)",
}
MASK_R_IN, MASK_R_OUT = 16, 17  # positions of r_in and r_out in metadata['masks']
PEAKS_PER_TARGET = 3
PEAK_MIN_DISTANCE = 20  # samples between two reported peaks of one target
SIGNIFICANT = 2.0       # a peak counts as leakage when SNR > 2 x the noise-floor peak


def target_values(metadata, byte, rng):
    p = metadata["plaintext"][:, byte]
    k = metadata["key"][:, byte]
    masks = metadata["masks"]
    sbox = S_BOX[p ^ k]
    return {
        "sbox": sbox,
        "sbox_xor_r": sbox ^ masks[:, byte],
        "r": masks[:, byte],
        "sbox_xor_rout": sbox ^ masks[:, MASK_R_OUT],
        "rout": masks[:, MASK_R_OUT],
        "random": rng.integers(0, 256, len(p), dtype=np.uint8),
    }


class ClassStats:
    # Running count, sum and sum of squares per class and per sample.
    def __init__(self, classes, samples):
        self.count = np.zeros(classes, dtype=np.int64)
        self.sum = np.zeros((classes, samples), dtype=np.float64)
        self.sum_sq = np.zeros((classes, samples), dtype=np.float64)

    def add(self, traces, labels):
        order = np.argsort(labels, kind="stable")
        labels, traces = labels[order], traces[order]
        classes, starts = np.unique(labels, return_index=True)
        self.count[classes] += np.diff(np.append(starts, len(labels)))
        self.sum[classes] += np.add.reduceat(traces, starts, axis=0)
        self.sum_sq[classes] += np.add.reduceat(traces * traces, starts, axis=0)

    def snr(self):
        used = self.count >= 2  # a variance needs at least two traces
        n = self.count[used][:, None]
        mean = self.sum[used] / n
        var = self.sum_sq[used] / n - mean * mean
        noise = var.mean(axis=0)
        return mean.var(axis=0) / np.where(noise > 0, noise, np.nan)


def find_peaks(snr, count, min_distance):
    # Highest samples, at least min_distance apart.
    peaks = []
    for i in np.argsort(np.nan_to_num(snr, nan=-1.0))[::-1]:
        if all(abs(int(i) - j) >= min_distance for j in peaks):
            peaks.append(int(i))
        if len(peaks) == count:
            break
    return peaks


def checks(f, args, dataset):
    # Sanity checks on the file and the AISY dataset entry. Returns lines for the summary.
    lines = []
    n = min(args.traces, 5000)
    meta = f["Profiling_traces/metadata"][:n]
    labels = f["Profiling_traces/labels"][:n]
    sbox = S_BOX[meta["plaintext"][:, args.byte] ^ meta["key"][:, args.byte]]
    match = float(np.mean(labels == sbox))
    lines.append(f"- File labels equal Sbox(p[{args.byte}] ^ k[{args.byte}]) for {match:.1%} of the first {n} "
                 f"profiling traces.")

    if not meta["masks"][:, args.byte].any():
        lines.append(f"- **NOTE:** mask r[{args.byte}] is always 0, so byte {args.byte} is not masked in this file "
                     f"(ASCAD v1 masks bytes 2-15 only). Targets `sbox_xor_r` and `r` are not meaningful.")

    n_keys = len(np.unique(meta["key"], axis=0))
    lines.append(f"- Profiling key: {'fixed' if n_keys == 1 else f'variable ({n_keys} distinct keys in {n} traces)'}.")

    attack_keys = np.unique(f["Attack_traces/metadata"]["key"][:, args.byte])
    expected = bytearray.fromhex(dataset["key"])[args.byte]
    if len(attack_keys) == 1 and attack_keys[0] == expected:
        lines.append(f"- Attack key byte {args.byte} = 0x{expected:02X}, same as the `key` in `datasets.py`.")
    else:
        found = ", ".join(f"0x{v:02X}" for v in attack_keys[:5])
        lines.append(f"- **WARNING:** attack key byte {args.byte} in the file is {found}, but `datasets.py` gives "
                     f"0x{expected:02X}. AISY ranks the key from `datasets.py`, so GE and SR would be wrong.")

    width = f["Profiling_traces/traces"].shape[1]
    window_end = dataset["first_sample"] + dataset["number_of_samples"]
    if window_end != width or dataset["first_sample"] != 0:
        lines.append(f"- AISY window: samples {dataset['first_sample']}-{window_end - 1} of {width} in the file.")
    else:
        lines.append(f"- AISY window covers all {width} samples in the file.")

    for line in lines:
        print(line)
    return lines


def compute(f, args, rng):
    traces = f["Profiling_traces/traces"]
    metadata = f["Profiling_traces/metadata"]
    total = min(args.traces, traces.shape[0])
    sizes = sorted({s for s in args.sizes if s < total} | {total})
    classes = 9 if args.leakage_model == "HW" else 256
    stats = {t: ClassStats(classes, traces.shape[1]) for t in TARGETS}
    snr_by_size = {}

    # Chunk edges include every size, so a snapshot can be taken exactly there.
    edges = sorted(set(range(0, total, args.chunk)) | set(sizes) | {0})
    for start, end in zip(edges[:-1], edges[1:]):
        x = traces[start:end].astype(np.float64)
        values = target_values(metadata[start:end], args.byte, rng)
        for t in TARGETS:
            labels = HW[values[t]] if args.leakage_model == "HW" else values[t]
            stats[t].add(x, labels)
        if end in sizes:
            snr_by_size[end] = {t: stats[t].snr() for t in TARGETS}
            print(f"{end} traces: peak SNR " + ", ".join(f"{t} {np.nanmax(snr_by_size[end][t]):.4f}" for t in TARGETS))
    return snr_by_size, total


def plot_snr(path, snr, dataset, args, total):
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 2.0 * len(TARGETS)), sharex=True)
    floor = np.nanmax(snr["random"])
    start, end = dataset["first_sample"], dataset["first_sample"] + dataset["number_of_samples"]
    for ax, t in zip(axes, TARGETS):
        ax.plot(snr[t], linewidth=0.7, color="black" if t == "random" else "tab:blue")
        ax.axhline(floor, color="tab:red", linewidth=0.7, linestyle="--")
        ax.axvspan(start, end - 1, color="tab:green", alpha=0.08)
        ax.set_title(TITLES[t], fontsize=9, loc="left")
        ax.set_ylabel("SNR", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("sample", fontsize=8)
    fig.suptitle(f"{args.dataset}, byte {args.byte}, {args.leakage_model} classes, {total} profiling traces. "
                 f"Red: noise-floor peak. Green: AISY window.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_snr_vs_traces(path, snr_by_size, args):
    sizes = sorted(snr_by_size)
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in TARGETS:
        ax.plot(sizes, [np.nanmax(snr_by_size[s][t]) for s in sizes], marker="o", label=t,
                color="black" if t == "random" else None, linestyle="--" if t == "random" else "-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("profiling traces")
    ax.set_ylabel("peak SNR")
    ax.set_title(f"Peak SNR vs profiling-set size ({args.leakage_model} classes)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(folder, snr_by_size, total, check_lines, dataset, args):
    snr = snr_by_size[total]
    floor = float(np.nanmax(snr["random"]))
    np.savez(os.path.join(folder, "snr.npz"),
             **{f"{t}_{s}": snr_by_size[s][t] for s in snr_by_size for t in TARGETS})

    rows = []
    for t in TARGETS:
        for rank, i in enumerate(find_peaks(snr[t], PEAKS_PER_TARGET, PEAK_MIN_DISTANCE), 1):
            rows.append({"target": t, "rank": rank, "sample": i, "snr": round(float(snr[t][i]), 5),
                         "x_noise_floor": round(float(snr[t][i]) / floor, 2)})
    with open(os.path.join(folder, "peaks.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    start, end = dataset["first_sample"], dataset["first_sample"] + dataset["number_of_samples"]
    lines = [
        f"# Phase 1: SNR leakage analysis ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Data: {args.dataset} (`data/{dataset['filename']}`), byte {args.byte}, {args.leakage_model} classes, "
        f"first {total} profiling traces, all {len(snr['random'])} samples. Noise floor (peak SNR of random "
        f"labels): {floor:.5f}. A peak counts as leakage above {SIGNIFICANT:g} x the noise floor.",
        "",
        "## Checks",
        "",
        *check_lines,
        "",
        "## Peaks",
        "",
        "| Target | Peak sample | Peak SNR | x noise floor | Leaks? | Peak in AISY window? |",
        "|---|---|---|---|---|---|",
    ]
    leaking = {}
    for t in TARGETS[:-1]:
        i = int(np.nanargmax(snr[t]))
        ratio = float(snr[t][i]) / floor
        leaking[t] = (ratio > SIGNIFICANT, i)
        lines.append(f"| {t} | {i} | {snr[t][i]:.5f} | {ratio:.1f} | {'yes' if ratio > SIGNIFICANT else 'no'} "
                     f"| {'yes' if start <= i < end else 'no'} |")

    lines += ["", "## Meaning", ""]
    unmasked = "no" if not leaking["sbox"][0] else "a"
    lines.append(f"- Unmasked value: {unmasked} first-order SNR peak.")
    for share_1, share_2 in (("sbox_xor_r", "r"), ("sbox_xor_rout", "rout")):
        (leak_1, i_1), (leak_2, i_2) = leaking[share_1], leaking[share_2]
        if leak_1 and leak_2:
            lines.append(f"- Shares `{share_1}` (sample {i_1}) and `{share_2}` (sample {i_2}) both leak, "
                         f"{abs(i_1 - i_2)} samples apart. A model must combine both points to recover the key.")
        else:
            lines.append(f"- Shares `{share_1}` / `{share_2}`: not both above the noise floor.")
    lines += [
        "- CNN design: the receptive field (or the dense layers after `Flatten`) must span the distance "
        "between the two share peaks of a pair.",
        "",
        "Files: `snr.png`, `snr_vs_traces.png`, `peaks.csv`, `snr.npz` (keys `<target>_<traces>`).",
    ]
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SNR leakage analysis on the profiling traces.")
    parser.add_argument("--dataset", default="ASCAD.h5", choices=list(datasets_dict),
                        help="entry in custom/custom_datasets/datasets.py (default ASCAD.h5)")
    parser.add_argument("--byte", type=int, default=2, help="AES key byte (default 2)")
    parser.add_argument("--traces", type=int, default=50000, help="profiling traces to use (default 50000)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000],
                        help="smaller profiling-set sizes for snr_vs_traces.png (default 1000 5000 10000 25000)")
    parser.add_argument("--leakage-model", default="ID", choices=["ID", "HW"],
                        help="classes: ID = 256 values, HW = 9 Hamming weights (default ID)")
    parser.add_argument("--chunk", type=int, default=10000, help="traces read at a time (default 10000)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random-label noise floor (default 0)")
    parser.add_argument("--data-folder", default="data/", help="folder with the .h5 files (default data/)")
    args = parser.parse_args()
    if not 0 <= args.byte < 16:
        parser.error("--byte must be 0-15")
    if args.traces < 2 or args.chunk < 1:
        parser.error("--traces must be >= 2 and --chunk >= 1")

    dataset = datasets_dict[args.dataset]
    path = os.path.join(args.data_folder, dataset["filename"])
    if not os.path.isfile(path):
        parser.error(f"dataset file not found: {path}")

    folder = start_run("phase1")
    with log_to(os.path.join(folder, "log.txt")):
        print(f"Phase 1: {vars(args)}")
        rng = np.random.default_rng(args.seed)
        with h5py.File(path, "r") as f:
            check_lines = checks(f, args, dataset)
            snr_by_size, total = compute(f, args, rng)
        plot_snr(os.path.join(folder, "snr.png"), snr_by_size[total], dataset, args, total)
        plot_snr_vs_traces(os.path.join(folder, "snr_vs_traces.png"), snr_by_size, args)
        write_outputs(folder, snr_by_size, total, check_lines, dataset, args)
        print(f"\nResults: {folder}")
        finish_run(folder, "phase1", args.seed)


if __name__ == "__main__":
    main()
