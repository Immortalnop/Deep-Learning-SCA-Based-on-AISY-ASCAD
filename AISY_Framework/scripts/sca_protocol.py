# Attack protocol shared by every phase: data split, key rank, GE / SR, and reading GE curves from the AISY database.
#
# Process: load_split reads the same traces AISY trains and attacks on -> an attack gives a score per trace and
# key guess -> key_ranks ranks the correct key on the same random subsets of attack traces -> ge_sr averages
# the ranks into guessing entropy (GE) and success rate (SR) curves.
# No TensorFlow import here, so the classic attacks (Phase 2) and analysis scripts load fast.
#
# Usage:
#     from scripts.sca_protocol import PROTOCOL, attack_subsets, ge_sr, key_ranks, load_split
#     split = load_split("data/", datasets_dict["ASCAD.h5"], byte=2)
#     subsets = attack_subsets(len(split["x_attack"]), seed=0)
#     ge, sr = ge_sr(key_ranks(key_scores(log_p, plaintext, 2, "HW"), split["key"], subsets))
import json
import os

import h5py
import numpy as np

# Same for every model and every phase. Early stopping keeps the epoch with the lowest validation GE;
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

DATABASE_FOLDER = "databases/"
DATABASE_NAME = "database_ascad.sqlite"
GE_THRESHOLD = 2  # AISY counts a key as found when GE < 2 (rank 1 = best)
LABEL_FINAL = "Attack Set"                          # weights after the last epoch
LABEL_BEST = "ES Attack Set guessing_entropy"       # weights from the best epoch
LOG_FLOOR = np.log(1e-36)                           # AISY adds 1e-36 before the log of a probability
HW = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)


def aes_s_box():
    # AES S-box (FIPS-197, 5.1.1): inverse in GF(2^8), then the affine map. Same table as
    # aisy_sca.crypto.aes.aes128.s_box, built here because importing aisy_sca loads TensorFlow.
    def multiply(a, b):
        product = 0
        while b:
            if b & 1:
                product ^= a
            a = ((a << 1) ^ 0x11B) if a & 0x80 else a << 1
            b >>= 1
        return product

    table = np.zeros(256, dtype=np.uint8)
    for value in range(256):
        inverse = next((c for c in range(1, 256) if multiply(value, c) == 1), 0)
        rotations = [((inverse << i) | (inverse >> (8 - i))) & 0xFF for i in range(5)]
        table[value] = rotations[0] ^ rotations[1] ^ rotations[2] ^ rotations[3] ^ rotations[4] ^ 0x63
    return table


S_BOX = aes_s_box()


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def load_split(data_folder, dataset, byte, profiling_dtype=np.float32):
    # Same traces as AISY (aisy_sca/datasets/DatasetControls.py): first n profiling traces, the sample window
    # from datasets.py, and the attack traces split in half (validation, attack) because early stopping is on.
    # Profiling traces are float32 to halve memory; attack traces stay float64 like AISY.
    n_prof, n_att = PROTOCOL["profiling_traces"], PROTOCOL["attack_traces"]
    first, end = dataset["first_sample"], dataset["first_sample"] + dataset["number_of_samples"]
    with h5py.File(os.path.join(data_folder, dataset["filename"]), "r") as f:
        x_prof = f["Profiling_traces/traces"][:n_prof, first:end].astype(profiling_dtype)
        meta_prof = f["Profiling_traces/metadata"][:n_prof]
        x_att = f["Attack_traces/traces"][n_att // 2:n_att, first:end].astype(np.float64)
        meta_att = f["Attack_traces/metadata"][n_att // 2:n_att]

    keys = np.unique(meta_att["key"][:, byte])
    expected = bytearray.fromhex(dataset["key"])[byte]
    if len(keys) != 1 or keys[0] != expected:
        # AISY ranks the key from datasets.py: a mismatch makes every GE in Phase 0 / 3 wrong.
        raise ValueError(f"attack key byte {byte} in {dataset['filename']} is "
                         f"{', '.join(f'0x{k:02X}' for k in keys[:5])}, datasets.py gives 0x{expected:02X}")
    print(f"Profiling {x_prof.shape}, attack {x_att.shape} (attack traces {n_att // 2}-{n_att - 1}, "
          f"samples {first}-{end - 1}), attack key byte {byte} = 0x{expected:02X}")
    return {"x_profiling": x_prof, "meta_profiling": meta_prof, "x_attack": x_att, "meta_attack": meta_att,
            "key": int(expected)}


def guess_classes(plaintext, byte, leakage_model):
    # Class that every key guess k (columns) predicts for every trace (rows): Sbox(p ^ k), or its HW.
    values = S_BOX[plaintext[:, byte, None] ^ np.arange(256, dtype=np.uint8)[None, :]]
    return HW[values] if leakage_model == "HW" else values


def key_scores(log_probabilities, plaintext, byte, leakage_model):
    # (traces, classes) log-probabilities of a model -> (traces, 256) log-probability of every key guess.
    return np.take_along_axis(log_probabilities, guess_classes(plaintext, byte, leakage_model).astype(np.int64), axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Key rank, GE and SR
# ---------------------------------------------------------------------------------------------------------------
def attack_subsets(n_traces, seed):
    # key_rank_executions random subsets of key_rank_attack_traces traces. Reuse one list for every method
    # that is compared, so differences come from the attacks and not from the subsets.
    rng = np.random.default_rng(seed)
    return [rng.choice(n_traces, PROTOCOL["key_rank_attack_traces"], replace=False)
            for _ in range(PROTOCOL["key_rank_executions"])]


def rank_of(scores, key):
    # Rank of the correct key (1 = best) along the last axis. Ties count half, which is the expected rank
    # under a random tie break: early points with HW labels have many tied key guesses, and counting
    # them as wins would make GE look better than it is.
    correct = np.take(scores, [key], axis=-1)
    greater = (scores > correct).sum(axis=-1)
    tied = (scores == correct).sum(axis=-1) - 1
    return 1 + greater + tied / 2


def key_ranks(log_probabilities, key, subsets, interval=PROTOCOL["key_rank_report_interval"]):
    # AISY's key rank: add the log-probability of every key guess over the attack traces, rank every
    # `interval` traces. log_probabilities: (attack traces, 256), one column per key guess.
    # Returns (subsets, points) ranks.
    return np.array([rank_of(np.cumsum(log_probabilities[s], axis=0)[interval - 1::interval], key)
                     for s in subsets])


def ge_sr(ranks):
    # GE = mean rank of the correct key, SR = share of subsets where it ranks first (no tie).
    return ranks.mean(axis=0), (ranks == 1).mean(axis=0)


def ge_area(ge):
    # Mean of the GE curve (GE after 10, 20, ..., 1,000 traces). Lower = key found with fewer traces.
    # Unlike the final GE it still separates two attacks that both reach GE 1 within the budget.
    return float(np.mean(ge))


def traces_to_threshold(ge, interval):
    # Number of attack traces after which GE stays below the threshold; None if never.
    above = [i for i, value in enumerate(ge) if value >= GE_THRESHOLD]
    if not above:
        return interval
    if above[-1] == len(ge) - 1:
        return None
    return (above[-1] + 2) * interval


def validation_log_probabilities(dataset, settings, model):
    # Log-probabilities of a Keras model on AISY's validation set, for the early-stopping metrics.
    # Only the first-round S-box output is supported, the target of every phase.
    leakage = settings["leakage_model"]
    if (leakage["target_state"] != "Sbox" or leakage["direction"] != "Encryption" or leakage["round"] != 1
            or leakage["leakage_model"] not in ("HW", "ID")):
        raise ValueError(f"GE metric supports HW / ID of the encryption S-box output, not {leakage}")
    return np.log(model.predict(dataset.x_validation, batch_size=1000, verbose=0) + 1e-36), leakage


# ---------------------------------------------------------------------------------------------------------------
# AISY database
# ---------------------------------------------------------------------------------------------------------------
def read_curve(connection, table, analysis_id, label):
    row = connection.execute(f'select "values", report_interval from {table} where analysis_id = ? and label = ?',
                             (analysis_id, label)).fetchone()
    if row is None:
        return None, None
    values = json.loads(row[0])
    if isinstance(values, str):  # AISY stores the JSON twice encoded
        values = json.loads(values)
    return [values[str(i)] for i in range(len(values))], row[1]
