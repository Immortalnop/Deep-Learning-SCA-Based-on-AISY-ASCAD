# Directory Structure

Project: **Deep Learning-Based Side-Channel Attack on AES Using Power Traces** (group 30).
Neural networks learn from power traces in the ASCAD dataset and recover AES key byte 2, using the AISY framework.

This file maps every project folder to its files and their purpose.
For environment setup, phase status and the changelog, see `README.md`. For the evaluation protocol, see `Eval_params.md`.

---

## `/` (project root)

| File | Purpose |
|---|---|
| `README.md` | Project overview, setup and rebuild steps for the Python 3.7.6 virtualenv, dataset facts, script list, phase status and the changelog by phase. Replaces `Readme.txt`. |
| `Eval_params.md` | Evaluation protocol: GE and SR averaged over about 100 key-rank runs; headline metric is the number of attack traces to reach GE = 1; fixed seeds, splits and epoch budget. |
| `DIRECTORY_STRUCTURE.md` | This file. |
| `Deep Learning-Based Side Channel Attack.pdf` | Project brief / report document. |

---

## `aisy_env/`

Python 3.7 virtual environment. Contains only installed packages, no project code. Do not edit by hand.

| Path | Purpose |
|---|---|
| `Scripts/` | `python.exe`, `Activate.ps1` and installed command-line tools. |
| `Lib/site-packages/aisy_sca/` | The AISY framework library (v0.2.7): training, SCA metrics, hyperparameter search, ensembles, AES intermediates. |
| `Lib/site-packages/aisy_database/` | AISY database layer (v0.2.0): save, select, update and delete results in SQLite. |
| `Lib/site-packages/*` | Other dependencies (TensorFlow, h5py, numpy, Flask, Dash, and so on). |
| `Include/`, `etc/` | Headers and Jupyter config installed by packages. |

---

## `ASCAD/`

Clone of the ANSSI ASCAD repository. Reference material for the dataset; not used directly by the AISY scripts.

| Path | Purpose |
|---|---|
| `Readme.md` | ASCAD documentation: dataset format, acquisition setup, reference results. |
| `LICENSE` | ASCAD licence. |
| `ASCAD_generate.py` | Builds the labelled `.h5` dataset from raw traces. |
| `ASCAD_train_models.py` | Trains the reference ASCAD MLP/CNN models. |
| `ASCAD_test_models.py` | Evaluates trained models and plots key rank. |
| `ASCAD_data/` | Empty. The dataset used by the project is in `AISY_Framework/data/`. |
| `ATMEGA_AES_v1/` | Readme and parameter files for the ATMega8515 target, with fixed-key and variable-key sub-folders. |
| `STM32_AES_v2/` | Readme, parameter files and result figures for the STM32 target. |
| `.github/workflows/*.yml` | Upstream CI jobs for each target. |
| `.git/` | Git metadata of the clone. |

---

## `AISY_Framework/`

Main working folder. Run all scripts from here, for example:
`..\aisy_env\Scripts\python.exe -m scripts.script_aes_cnn`

| File | Purpose |
|---|---|
| `app.py` | Flask + Dash web app (AISY GUI) for browsing results, key-rank and success-rate plots, and editing scripts. Also defines the resource, database and dataset root folders imported by some scripts. |
| `readme.md` | Upstream AISY framework documentation. |
| `requirements.txt` | Python dependencies for `aisy_env`. |
| `CHANGELOG.md` | Pointer only: the changelog moved to `README.md` in the project root. |
| `LICENSE` | AISY licence. |
| `.env` | Flask environment and time-zone settings. |
| `.gitignore` | Git ignore rules. |
| `.idea/` | PyCharm project settings. |

### `AISY_Framework/data/`

| File | Purpose |
|---|---|
| `ASCAD.h5` | ASCAD v1 **variable-key** dataset (about 438 MB): 200,000 profiling traces (random keys), 100,000 attack traces (key `00112233...EEFF`), 1,400 samples, plaintexts, keys, masks and labels. |

### `AISY_Framework/databases/`

| File | Purpose |
|---|---|
| `database_ascad.sqlite` | AISY results database. Every run stores its settings, GE/SR curves and metrics here under an analysis id. |

### `AISY_Framework/resources/`

Output folder. Every run writes all its files into one folder under `experiments/` (see `scripts/experiment_output.py`).
The GE/SR curves stay in `databases/database_ascad.sqlite`; `index.csv` links each run folder to its analysis ids.

| Path | Purpose |
|---|---|
| `experiments/index.csv` | One row per run: run id, kind, seed, status, start time, duration, AISY analysis ids, folder. |
| `experiments/phase0_<timestamp>/` | Phase 0 comparison: `runs.csv` (one row per model and seed: status, analysis id, GE, SR, traces to GE < 2), `summary.md` (mean ± std per model), `log.txt`. |
| `experiments/phase1_<timestamp>/` | Phase 1 SNR analysis: `snr.png`, `snr_vs_traces.png`, `snr.npz`, `peaks.csv`, `summary.md`, `log.txt`. |
| `experiments/phase2_<timestamp>/` | Phase 2 classic baselines: `ge.png`, `share_correlation.png`, `results.csv`, `curves.npz`, `summary.md`, `log.txt`. |
| `experiments/experiment_<timestamp>_seed<N>/<script>/` | `log.txt` of each script run by `run_experiment.py`. |
| `experiments/<script>_<timestamp>_seed<N>/` | A core `script_aes*.py` run on its own. |
| `experiments/_archive/phase0_invalid/` | Phase 0 runs from 2026-10-08 and 2026-10-10. Invalid: wrong attack key (see README, Phase 1). |
| `experiments/_archive/retired_byte0/` | Old plots and `SCA_epoch_tables.xlsx` from before the evaluation protocol, computed for byte 0 (was `All_outputs-RETIRED/` in the project root). |
| `experiments/_archive/models/` | One orphan best-model `.h5` left by a crashed run. |
| `figures/` | Created by the webapp when it draws plots for an analysis. |
| `databases/`, `figures/`, `models/`, `npz/` | Created by AISY when an auxiliary or archival script runs (those still use `resources/` as AISY resources root). |

### `AISY_Framework/scripts/`

Experiment and model scripts. Each AISY script configures `aisy_sca.Aisy()` and calls `run()`.

**Experiment tooling (written by us)**

| File | Purpose |
|---|---|
| `experiment_seed.py` | Shared seed. Reads `SCA_SEED` from the environment or generates one, then seeds Python, numpy and TensorFlow and enables deterministic ops. |
| `run_experiment.py` | Runs the core scripts in sequence with one shared seed; one sub folder and one `index.csv` row per script. |
| `phase0_compare.py` | Phase 0 comparison: trains the MLP and `cnn_sca_light` with one protocol over several seeds, reads GE/SR from the database and writes `runs.csv` and `summary.md`. |
| `phase1_snr.py` | Phase 1 leakage analysis: SNR per sample for the unmasked value, masked values and masks; checks the dataset entry; writes plots, peak table and summary. |
| `phase2_baselines.py` | Phase 2 classic baselines: template attacks (whole-trace PCA, share-point PCA) and CPA (first order, known mask, second order), ranked on the same attack subsets with the Phase 0 GE protocol. |
| `experiment_log.py` | `log_to(path)`: copies everything printed to the console into a log file. Used by the phase scripts. |
| `experiment_output.py` | One output folder per run: `start_run` makes the folder, `configure_aisy` sets all AISY folders, `finish_run` removes empty AISY folders and writes the `index.csv` row. |

**Core model scripts (used for results)**

| File                                      | Purpose                                                                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `script_aes.py`                           | AISY basic MLP on ASCAD, HW leakage model, byte 2. Pre-written.                                                                 |
| `script_aes_neural_network_parameters.py` | MLP with hyperparameters passed as parameters. Pre-written.                                                                     |
| `script_aes_multi_model.py`               | Trains several models in one AISY analysis. Pre-written.                                                                        |
| `script_aes_cnn.py`                       | Our CNN (`cnn_sca_light`) through AISY. Written by us.                                                                          |
| `script_aes_cnn_new.py`                   | Standalone Keras CNN without AISY. Written by us; now retired because its per-trace HW-class rank is not a key-recovery metric. |

**Auxiliary scripts (examples of AISY plug-ins)**

| File | Purpose |
|---|---|
| `script_aes_custom_loss.py` | Runs a model with a custom loss from `custom/custom_loss/`. |
| `script_aes_custom_metrics.py` | Runs a model with custom metrics from `custom/custom_metrics/`. |
| `script_aes_custom_callback.py` | Runs a model with custom Keras callbacks from `custom/custom_callbacks/`. |
| `script_aes_custom_table.py` | Stores extra results in a custom database table from `custom/custom_tables/`. |

**Archival scripts (not used for final results)**

| File | Purpose |
|---|---|
| `script_aes_visualization.py` | Input-gradient visualisation of what the model learns. |
| `script_aes_random_search.py` | Random hyperparameter search. |
| `script_aes_grid_search.py` | Grid hyperparameter search. |
| `script_aes_ensembles.py` | Ensemble of models from a search. |
| `script_aes_data_augmentation.py` | Training with data augmentation. |
| `script_aes_new_dataset.py` | Example of adding a new dataset. |
| `script_aes_profiling_analyzer.py` | Effect of the number of profiling traces on results. |
| `Unused Scripts/` | Empty. |
| `__init__.py` | Makes `scripts` a package so `python -m scripts.<name>` works. |

### `AISY_Framework/custom/`

Plug-in modules loaded by AISY.

| Path | Purpose |
|---|---|
| `custom_models/neural_networks.py` | Model definitions: `mlp`, `cnn`, `ASCAD_mlp`, `ASCAD_cnn`, methodology CNNs, Zaid et al. models, ranking-loss CNN and our `cnn_sca_light`. |
| `custom_datasets/datasets.py` | `datasets_dict`: file name, attack key, trace counts and number of samples for each dataset. The `ASCAD.h5` entry was fixed in Phase 1 (key and 1,400 samples). |
| `custom_metrics/accuracy.py` | Accuracy metric. |
| `custom_metrics/loss.py` | Loss metric. |
| `custom_metrics/guessing_entropy.py` | Guessing entropy metric. |
| `custom_metrics/fast_guessing_entropy.py` | Faster GE estimate on fewer traces. |
| `custom_metrics/number_of_traces.py` | Number of traces needed to reach a GE target. |
| `custom_loss/cce.py` | Categorical cross-entropy loss. |
| `custom_loss/cer.py` | Cross-entropy ratio (CER) loss. |
| `custom_callbacks/callbacks.py` | Example Keras callbacks and `SaveWeights`. |
| `custom_data_augmentation/data_augmentation.py` | Random-shift and Gaussian-noise augmentation generators. |
| `custom_tables/tables.py` | Example custom SQLAlchemy table. |

### `AISY_Framework/webapp/`

GUI of the AISY framework, served by `app.py`.

| Path | Purpose |
|---|---|
| `mvc/controllers/app_controller.py` | Loads analysis data from the database for the pages. |
| `mvc/controllers/figures_controller.py` | Builds result figures. |
| `mvc/controllers/script_controller.py` | Generates script code shown for each analysis. |
| `mvc/views/views.py`, `plotly.py`, `Utils.py` | Plot and table builders for the pages. |
| `templates/base.html` | Base page layout. |
| `templates/dashboard/*.html` | Pages: index, datasets, results, script, search and documentation. |
| `static/` | Vendored SB Admin 2 theme: CSS, SCSS, JS, Bootstrap, Chart.js, DataTables, FontAwesome and Prism. Third-party; do not edit. |
