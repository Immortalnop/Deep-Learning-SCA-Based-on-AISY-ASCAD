# Deep Learning-Based Side-Channel Attack on AES Using Power Traces (group 30)

Neural networks learn from ASCAD power traces and recover AES key byte 2. Training, guessing entropy (GE),
success rate (SR) and run logging use the AISY framework.

Research question: how robust are CNN attacks on masked AES (ASCAD) to trace misalignment and limited
profiling data, and how do they compare to classic attacks?

- Folder and file map: `DIRECTORY_STRUCTURE.md`
- Evaluation protocol: `Eval_params.md`
- Changelog, by phase: [Changelog](#changelog) below

---

## Setup

The project runs in `aisy_env`, a Python 3.7.6 virtual environment.

1. Check the version (run from the project root). It must print `Python 3.7.6`:
   ```powershell
   .\aisy_env\Scripts\python.exe --version
   ```
2. Activate the environment:
   ```powershell
   .\aisy_env\Scripts\Activate.ps1
   $env:Path = "$PWD\aisy_env\Scripts;$env:Path"
   ```

If `aisy_env` does not work, rebuild it:

1. Delete it: `Remove-Item -Recurse -Force .\aisy_env`. Check with `Test-Path .\aisy_env` (must return `False`).
2. Create it again: `py -3.7 -m venv aisy_env`.
3. Check the version and activate it, as above.
4. Install the libraries: `.\aisy_env\Scripts\python.exe -m pip install -r AISY_Framework\requirements.txt`.

Do not install other analysis libraries (for example `scared`, `lascar`) into `aisy_env`. They can break AISY.
Use a separate environment for them.

## Running scripts

Run every script from `AISY_Framework`, as a module:

```powershell
cd AISY_Framework
..\aisy_env\Scripts\python.exe -m scripts.phase1_snr
..\aisy_env\Scripts\python.exe -m scripts.phase0_compare
..\aisy_env\Scripts\python.exe -m scripts.script_aes_cnn
```

All outputs of a run go to one folder: `AISY_Framework/resources/experiments/<kind>_<timestamp>[_seed<N>]/`
(for example `phase0_20261023_002735/`, or `experiment_<timestamp>_seed<N>/<script>/` from `run_experiment.py`).
Every run adds one row to `resources/experiments/index.csv`: run id, kind, seed, status, start time, duration,
AISY analysis ids and folder. Old and invalid results are in `resources/experiments/_archive/`.
AISY stores every training run (GE and SR curves, settings) in `AISY_Framework/databases/database_ascad.sqlite`; the
analysis ids in `index.csv` link a run folder to its database rows. Browse the runs in the AISY webapp:
`..\aisy_env\Scripts\python.exe app.py`.

## Dataset

`AISY_Framework/data/ASCAD.h5` is the **ASCAD v1 variable-key** dataset, not the fixed-key one:

- 200,000 profiling traces (random key per trace) and 100,000 attack traces (fixed key `00112233445566778899AABBCCDDEEFF`).
- 1,400 samples per trace, `int8`.
- Metadata: `plaintext`, `key`, `masks` (18 bytes: `r[0..15]`, `r_in`, `r_out`; `r[0]` and `r[1]` are always 0), `desync`.
- File labels are `Sbox(p[2] ^ k[2])`.

The AISY entry for it is `"ASCAD.h5"` in `AISY_Framework/custom/custom_datasets/datasets.py`. Since Phase 1 it uses
the correct attack key and all 1,400 samples (see the Phase 1 changelog).

## Scripts

**Phase scripts (written by us)**

| Script | Purpose |
|---|---|
| `scripts/phase0_compare.py` | Phase 0: AISY MLP vs our CNN under one protocol, over several seeds. |
| `scripts/phase1_snr.py` | Phase 1: SNR leakage analysis on the profiling traces. |
| `scripts/phase2_baselines.py` | Phase 2: template attacks (PCA + QDA) and CPA under the Phase 0 GE protocol. |
| `scripts/experiment_seed.py` | Shared seed (`SCA_SEED`) for Python, numpy and TensorFlow. |
| `scripts/experiment_log.py` | Writes everything printed to the console and to a `log.txt`. |
| `scripts/experiment_output.py` | One output folder per run (`start_run`), AISY folder setup (`configure_aisy`), `index.csv` row (`finish_run`). |
| `scripts/run_experiment.py` | Runs several scripts with one shared seed and logs each run. |

**Core model scripts**

| Script | Purpose |
|---|---|
| `script_aes.py` | AISY basic MLP. Pre-written. |
| `script_aes_neural_network_parameters.py` | MLP with hyperparameters as parameters. Pre-written. |
| `script_aes_multi_model.py` | Several models in one AISY analysis. Pre-written. |
| `script_aes_cnn.py` | Our CNN (`cnn_sca_light`) through AISY. Written by us. |
| `script_aes_cnn_new.py` | Retired: its per-trace HW-class rank is not a key-recovery metric. |

**Auxiliary scripts (AISY plug-in examples):** `script_aes_custom_loss.py`, `script_aes_custom_metrics.py`,
`script_aes_custom_callback.py`, `script_aes_custom_table.py`.

**Archival scripts (not used for results yet):** `script_aes_visualization.py`, `script_aes_random_search.py`,
`script_aes_grid_search.py`, `script_aes_ensembles.py`, `script_aes_data_augmentation.py`,
`script_aes_new_dataset.py`, `script_aes_profiling_analyzer.py`.

The auxiliary and archival scripts do not use `experiment_output.py` yet: AISY still uses `resources/` as their
resources root, so they create `resources/{databases,figures,models,npz}/` and add no `index.csv` row.

## Phase status

| Phase | Status |
|---|---|
| 0: Fix the baseline | Tooling done. Results from before 2026-10-21 are invalid (wrong attack key, see Phase 1). Rerun needed. |
| 1: Leakage analysis | Done. |
| 2: Classic baselines | Not started. |
| 3: Systematic CNN study | Not started. |
| 4: Robustness to misalignment | Not started. |
| 5: Explainability and harder data | Not started. |

---

## Changelog

Newest phase first. Each entry lists what changed and how it was checked.

### Tooling

#### 2026-10-23 — One output folder per run

**Added**
- `scripts/experiment_output.py`: `start_run(kind, seed)` makes `resources/experiments/<kind>_<timestamp>[_seed<N>]/`
  (or uses `SCA_RUN_FOLDER` from `run_experiment.py`); `configure_aisy(aisy, folder)` sets all AISY folders in one
  call, with the run folder as AISY resources root; `finish_run(...)` prints `ANALYSIS_ID = ...`, removes the empty
  `databases/ figures/ models/ npz/` folders AISY creates, and adds a row to `resources/experiments/index.csv`.

**Changed**
- `scripts/phase0_compare.py`, `scripts/phase1_snr.py`: run folder and index row from `experiment_output.py`.
  Output files unchanged.
- `scripts/run_experiment.py`: folder `experiment_<timestamp>_seed<N>/`, one sub folder per script with its
  `log.txt` (was `<script>.log`); one `index.csv` row per script, with the analysis ids read from the log.
  `resources/experiments/experiment_log.csv` replaced by `index.csv`.
- Core scripts `script_aes.py`, `script_aes_neural_network_parameters.py`, `script_aes_cnn.py`,
  `script_aes_multi_model.py`: `start_run` + `configure_aisy` + `finish_run` instead of 4 folder setters.
- Archived to `resources/experiments/_archive/`: invalid Phase 0 runs (`phase0_invalid/`), `All_outputs-RETIRED/`
  from the project root (`retired_byte0/`), and one orphan model `.h5` left by a crashed run (`models/`).
- Removed empty folders: `my_resources_location/` (unused AISY template), `resources/{databases,figures,models,npz}/`.
  The webapp recreates `resources/figures/` when it needs it.
- `resources/experiments/index.csv` started with the two Phase 1 runs.

**Note:** AISY saves best-epoch weights to `<resources root>/models/` only while it evaluates them, then deletes them
(`aisy_sca/sca/Profiling.py`, `delete_model`). Run folders therefore keep no `.h5` files.

**Verified** (real database backed up before and restored after; test folders and index rows removed)
- `phase1_snr --traces 2000`: same 6 files in a new `phase1_<timestamp>/`; index row written.
- `phase0_compare --seeds 4242 --epochs 1 --models mlp`: `log.txt`, `runs.csv`, `summary.md` in one folder, no empty
  AISY folders; index row has the analysis id.
- `run_experiment --seed 4242 --scripts script_aes_cnn`: `experiment_<timestamp>_seed4242/script_aes_cnn/log.txt`;
  index row has the analysis id. First try showed OneDrive marks new folders read-only, so `os.rmdir` failed;
  `remove_empty_folders` now clears the flag first and never fails the run.
- `script_aes_cnn` on its own: own `script_aes_cnn_<timestamp>_seed4242/` folder and index row.
- Nothing new written to `resources/` outside `resources/experiments/`.

### Phase 1: Leakage analysis

#### 2026-10-21 — SNR analysis, dataset entry fixed

**Added**
- `scripts/phase1_snr.py`: SNR per sample on the profiling traces, before training any network.
  - Process: read traces and metadata in chunks -> compute each target value -> per-class mean and variance per
    sample -> SNR = Var(class means) / Mean(class variances).
  - Targets for byte 2: `sbox` = Sbox(p ^ k) (unmasked), `sbox_xor_r` = Sbox(p ^ k) ^ r[2] (masked value),
    `r` = r[2] (mask), `sbox_xor_rout` = Sbox(p ^ k) ^ r_out, `rout` = r_out, and `random` labels as the noise floor.
  - Uses the full trace width of the file, and shades the AISY sample window on the plot, so a window that misses
    the leakage is visible.
  - Checks first: file labels against Sbox(p ^ k), fixed or variable profiling key, attack key in the file against
    the `key` in `datasets.py`, AISY window against the file width, and whether the byte is masked at all.
  - Options: `--dataset`, `--byte` (default 2), `--traces` (default 50,000), `--sizes` (smaller profiling-set sizes,
    default 1,000 / 5,000 / 10,000 / 25,000), `--leakage-model ID|HW` (default ID), `--chunk`, `--seed`, `--data-folder`.
  - Output in `resources/experiments/phase1_<timestamp>/`: `snr.png`, `snr_vs_traces.png`, `snr.npz`, `peaks.csv`,
    `summary.md`, `log.txt`.
- `scripts/experiment_log.py`: `log_to(path)` sends everything printed to the console and to a log file. A
  traceback that stops the run is also written to the log.

**Changed**
- `custom/custom_datasets/datasets.py`, entry `"ASCAD.h5"`:
  - `key` `4DFBE0F27221FE10A78D4ADC8E490469` -> `00112233445566778899AABBCCDDEEFF`. The old key is the fixed-key
    ASCAD key. AISY ranks the key byte from this field, so every GE and SR so far was computed for the wrong key
    byte (0xE0 instead of 0x22).
  - `number_of_samples` 700 -> 1400. The masked-value peaks are at samples 1000 and 1071, outside the old window.
  - Trace counts stay 50,000 / 10,000, so scripts without explicit counts do not load 2+ GB of traces.
- `scripts/phase0_compare.py`: uses `experiment_log.log_to` instead of its own `Tee` class. Same `log.txt` output.
- `scripts/script_aes_cnn.py`: comment updated to input size 1400.

**Results** (`resources/experiments/phase1_20261021_004650/`: ID classes, 50,000 traces;
`phase1_20261021_004619/`: HW classes)

| Target | Peak sample | Peak SNR (ID) | x noise floor |
|---|---|---|---|
| Sbox(p ^ k), unmasked | — | 0.0069 | 1.0 (no leakage) |
| Sbox(p ^ k) ^ r[2] | 1071 | 1.54 | 225 |
| r[2] | 188 | 1.15 | 168 |
| Sbox(p ^ k) ^ r_out | 1000 | 1.40 | 205 |
| r_out | 188 (also 125) | 0.19 | 28 |

- As expected: no first-order leakage of the unmasked value, clear leakage of each share. A model must combine two
  points to recover the key.
- Share pairs are 812–883 samples apart. `cnn_sca_light` has a receptive field of 66 samples (two Conv1D-11 +
  AvgPool-4 blocks), so only its dense layers after `Flatten` can combine the shares. Phase 3 should test wider
  receptive fields.
- The r_out top peak (sample 188) is at the same sample as the r[2] peak. Its own separate peak is at 125.
- Peak SNR stops changing after about 10,000 profiling traces, while the noise floor keeps falling
  (`snr_vs_traces.png`). The leakage itself is stable with small profiling sets.
- HW classes give the same peak samples.

**Consequence for Phase 0:** the Phase 0 results from 2026-10-08 and 2026-10-10 (GE ≈ 100, SR 0) are invalid: wrong
key byte, and the masked-value leakage was outside the window. Rerun Phase 0 with the fixed entry
(`..\aisy_env\Scripts\python.exe -m scripts.phase0_compare`, 3 seeds × 50 epochs).

**Verified**
- Full runs: ID and HW, 50,000 traces. All checks pass after the fix (labels 100 % equal Sbox(p ^ k), attack key
  matches, window covers all samples).
- Before the fix, the same script printed the key-mismatch warning and showed the masked-value peaks outside the window.
- Argument errors: `--byte 16`, `--traces 1`, a missing data folder all stop with a clear message.
- Edge cases: `--byte 0` (unmasked byte) prints a note; `--sizes` larger than `--traces` are skipped; `--traces`
  smaller than `--chunk` works.
- `phase0_compare.py` after both changes: `--seeds 4242 --epochs 10`, both models, HW. Both runs finished; `runs.csv`,
  `summary.md` and `log.txt` written; the 1,400-sample input works for both models. Real database restored and test
  folder removed afterwards.
  - MLP: final GE 14.8, SR 0.26 after 10 epochs (before the fix: GE ≈ 99 after 50 epochs). The key fix works.
  - CNN (`cnn_sca_light`): final GE 105, SR 0.02. It still does not learn. Its 66-sample receptive field cannot
    span the shares, so this is a design problem to fix in Phase 3, not a pipeline error.

### Phase 0: Fix the baseline

#### 2026-10-10 — Phase 0 run log

**Changed**
- `scripts/phase0_compare.py` writes `log.txt` into its experiment folder (`resources/experiments/phase0_<timestamp>/`).
  Everything printed during the run goes to both the console and the log, including errors. The log starts with the
  full `PROTOCOL`, so runs with overridden settings (for example the ID leakage model) are recorded.

**Verified**
- Test run: ID leakage model, 1 seed, 1 epoch, 2,000 traces, temp database. `log.txt`, `runs.csv` and `summary.md`
  all written to the same folder.

#### 2026-10-07 — Phase 0 orchestrator (MLP vs CNN)

**Added**
- `scripts/phase0_compare.py`: runs the Phase 0 comparison end to end.
  - Process: generate seeds (or `--seeds`) -> for each seed, train every model under one protocol -> read GE and SR
    from `databases/database_ascad.sqlite` -> summary.
  - Models: AISY `mlp` (4 × 200 selu) and `cnn_sca_light` (group 30 CNN).
  - Protocol: `ASCAD.h5`, byte 2, HW, 50,000 profiling traces, 10,000 attack traces split into 5,000 validation +
    5,000 attack, GE over 100 key ranks on 1,000 traces (every 10 traces), early stopping on validation GE.
  - Defaults: 3 seeds, 50 epochs, batch size 128 (`--n-seeds`, `--epochs`, `--batch-size`, `--models`).
  - Output: `runs.csv` (one row per model and seed, with AISY analysis id) and `summary.md` (mean ± std per model).
  - Metric: traces until GE stays below 2 (AISY's success threshold), for best-epoch and last-epoch weights.
- `set_seed(seed)` in `scripts/experiment_seed.py`.

**Changed**
- CNN moved from `scripts/script_aes_cnn.py` to `custom/custom_models/neural_networks.py` as
  `cnn_sca_light(classes, number_of_samples)`.
- `scripts/script_aes_cnn.py`: trace counts (50,000 / 5,000) set after `set_dataset`, so they take effect.
- `custom/custom_datasets/datasets.py`: `ASCAD.h5` profiling traces 500000 -> 50000.

**Results (invalid, see Phase 1; now in `resources/experiments/_archive/phase0_invalid/`):**
`phase0_20261008_003257` (HW) and `phase0_20261010_001826` (ID): 0/3 runs reach
GE < 2 for either model, final GE ≈ 98–120.

#### 2026-10-07 — Shared experiment seed and runner

**Added**
- `scripts/experiment_seed.py`: reads the seed from `SCA_SEED` or generates one (0–1048575). `apply_seed()` fixes the
  Python, numpy and TensorFlow seeds, enables deterministic ops, prints `SEED = ...` and returns the seed.
- `scripts/run_experiment.py`: runs several scripts with one shared seed. Writes
  `resources/experiments/<experiment_id>/<script>.log` and a row per script in `resources/experiments/experiment_log.csv`.

**Changed**
- All 15 `scripts/script_aes*.py` (except retired `script_aes_cnn_new.py`): `SEED = apply_seed()` before AISY starts,
  and `aisy.add_neural_network(..., seed=SEED)`.
- `scripts/script_aes_cnn.py`: `byte=2`; model passed as a function; input size and classes taken from AISY.
- `scripts/script_aes_neural_network_parameters.py`: `mlp` has default arguments and is passed as a function.
- Old results in `All_outputs/` renamed to `All_outputs-RETIRED/` (computed for byte 0). Since 2026-10-23 in
  `resources/experiments/_archive/retired_byte0/`.
- `script_aes_cnn_new.py` retired.

**Verified**
- Smoke test of every script with `SCA_SEED=4242` (1 epoch, 2,000 traces, temp folder). 11 scripts run and store
  seed 4242.

**Known issues (still open)**
- `script_aes_ensembles.py` fails with `KeyError: 'structure'` (its `grid_search` dict has no `"structure"` block).
- `script_aes_custom_callback.py`, `script_aes_random_search.py`, `script_aes_visualization.py` fail:
  `data/ascad-variable.h5` is missing. Note: `data/ASCAD.h5` is that variable-key dataset, so these scripts can use
  the `"ASCAD.h5"` entry instead.
- `script_aes_custom_table.py` writes a demo table and row into `databases/database_ascad.sqlite`.
