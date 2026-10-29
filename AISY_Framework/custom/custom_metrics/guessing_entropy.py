# AISY early-stopping metric: final guessing entropy on the validation set (Phase 0).
#
# Same key rank as every phase (scripts/sca_protocol.py): log-probabilities added over the traces, ties count half,
# on fixed random subsets so every epoch is compared on the same traces. The earlier version added raw
# probabilities, which is not AISY's key rank and picks epochs with a weaker rule.
from scripts.sca_protocol import attack_subsets, ge_sr, key_ranks, key_scores, validation_log_probabilities

VALIDATION_SEED = 1  # subsets of the validation set; the attack set uses the --seed of each phase script


def run(dataset, settings, model, *args):
    log_p, leakage = validation_log_probabilities(dataset, settings, model)
    scores = key_scores(log_p, dataset.plaintext_validation, leakage["byte"], leakage["leakage_model"])
    ge, _ = ge_sr(key_ranks(scores, settings["good_key"], attack_subsets(len(scores), VALIDATION_SEED)))
    print(f"GE = {ge[-1]}")
    return float(ge[-1])
