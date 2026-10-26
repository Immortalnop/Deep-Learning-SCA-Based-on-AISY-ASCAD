# AISY early-stopping metric: area of the validation GE curve (Phase 3 and later).
#
# Value = mean of the GE curve after 10, 20, ..., 1,000 validation traces (scripts/sca_protocol.ge_area). Lower = key
# found with fewer traces. The final GE stops separating epochs once several reach GE 1; the area does not.
# Every epoch's value is also appended to settings["ge_area_history"], which the PlateauStop callback reads.
from scripts.sca_protocol import attack_subsets, ge_area, ge_sr, key_ranks, key_scores, validation_log_probabilities

VALIDATION_SEED = 1  # same validation subsets as custom_metrics/guessing_entropy.py
HISTORY_KEY = "ge_area_history"


def run(dataset, settings, model, *args):
    log_p, leakage = validation_log_probabilities(dataset, settings, model)
    scores = key_scores(log_p, dataset.plaintext_validation, leakage["byte"], leakage["leakage_model"])
    ge, _ = ge_sr(key_ranks(scores, settings["good_key"], attack_subsets(len(scores), VALIDATION_SEED)))
    value = ge_area(ge)
    settings.setdefault(HISTORY_KEY, []).append(value)
    print(f"Validation GE area = {value:.2f} (final GE {ge[-1]:.2f})")
    return value
