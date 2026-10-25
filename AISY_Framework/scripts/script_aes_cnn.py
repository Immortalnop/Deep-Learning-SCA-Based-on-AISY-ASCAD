import aisy_sca
from scripts.experiment_output import configure_aisy, finish_run, start_run
from scripts.experiment_seed import apply_seed
from custom.custom_datasets.datasets import *
from custom.custom_models.neural_networks import cnn_sca_light

# -----------------------------
# SEED (shared, see scripts/experiment_seed.py)
# -----------------------------
SEED = apply_seed()

# -----------------------------
# OUTPUT FOLDER (see scripts/experiment_output.py)
# -----------------------------
RUN_FOLDER = start_run("script_aes_cnn", SEED)

# -----------------------------
# CNN MODEL DEFINITION
# -----------------------------
# cnn_sca_light lives in custom/custom_models/neural_networks.py
# so this script and scripts/phase0_compare.py use the same model.

# -----------------------------
# AISY SETUP
# -----------------------------
aisy = aisy_sca.Aisy()

# Folders set in one place (no app.py import needed)
configure_aisy(aisy, RUN_FOLDER)
# Dataset (must match datasets_dict key)
aisy.set_dataset(datasets_dict["ASCAD.h5"])
# Trace counts after set_dataset, which would otherwise overwrite them
aisy.set_number_of_profiling_traces(50000)
aisy.set_number_of_attack_traces(5000)

# Attack settings
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)

aisy.set_batch_size(64)
aisy.set_epochs(5)

# -----------------------------
# MODEL
# -----------------------------
# Input size (1400) and classes (9 for HW) come from the dataset and leakage model.
# Passing the function (not a built model) lets AISY seed and record it.
aisy.add_neural_network(cnn_sca_light, seed=SEED)

# -----------------------------
# RUN ATTACK
# -----------------------------
aisy.run()
finish_run(RUN_FOLDER, "script_aes_cnn", SEED, [aisy.settings["analysis_id"]])
