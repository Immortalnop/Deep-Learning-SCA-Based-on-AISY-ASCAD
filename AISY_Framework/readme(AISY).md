# AISY - Deep Learning-based Framework for Side Channel Analysis

Welcome to the first deep learning-based side-channel analysis framework.
This AISY Framework was developed by AISYLab from TU Delft.

Contributors: Guilherme Perin, Lichao Wu and Stjepan Picek.

### Installation

```
git clone https://github.com/AISyLab/AISY_Framework.git
cd AISY_framework
pip install -r requirements.txt
```

To start the webapp:

```
flask run
```

### Requirements

This framework required ```tensorflow==2.0.0```, ```keras==2.1.6``` and ```python==3.7```.

All requires python packages are listed in ```requirements.txt```.

### Documentation

See our documentation page: https://aisylab.github.io/AISY_docs/

### Main Features

- SCA Metrics
- Gradient Visualization
- Data Augmentation 
- Grid Search
- Random Search
- Early Stopping
- Ensemble
- Profiling Analyzer  
- Custom Metrics
- Custom Callbacks
- Custom Loss Functions  
- Confusion Matrix
- Easy Neural Network Definitions
- Data Augmentation
- GUI - plots, tables
- Automatically generate scripts 
- Fully reproducible script   

### Example:

```python
import aisy_sca
from app import *
from custom.custom_models.neural_networks import *

aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ascad-variable.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)
aisy.run()
```

### Running a script

After your AISY Framework project is set on your local machine, to run scripts that are placed inside ```scripts``` folder from 
command line or terminal, we recommend to write the following lines on your script (as the first lines of script code):

```python
import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
```

However, if you prefer to run scripts from inside an IDE such as PyCharm, you don't have to set the above code in your scripts.

Alternatively, you can place your script files outside ```scripts``` folder (in your project root directory) and run them from there.

If you use our framework, please consider citing:

    @misc{cryptoeprint:2021:357,
      author = {Guilherme Perin and Lichao Wu and Stjepan Picek},
      title  = {{AISY - Deep Learning-based Framework for Side-Channel Analysis}},
      howpublished = {Cryptology ePrint Archive, Report 2021/357},
      note   = {\url{https://eprint.iacr.org/2021/357}},
      year   = {2021}
    }
__________________________________________________________________________
Added By AARUSH MOHAN MATHUR: 

### Script status (group 30)

Smoke-tested on 2026-10-07 (1 epoch, 2,000 traces, `SCA_SEED=4242`). All scripts attack ASCAD byte 2 with the HW leakage model (9 classes) unless noted. All scripts read the shared seed from `scripts/experiment_seed.py`; see the changelog in `README.md` (project root). Since Phase 1 (2026-10-21) the `ASCAD.h5` entry uses the correct attack key and all 1,400 samples, so results before that date are invalid.

| Script | Purpose | Implementation | Works? |
|---|---|---|---|
| `script_aes` | Baseline attack | AISY `mlp` (4×200 selu), batch 400, 20 epochs | **Yes.** Reference baseline. |
| `script_aes_cnn` | Our CNN | 2 Conv1D + 2 AvgPool(4) + Dense 50, batch 64, 5 epochs | **Yes.** 5 epochs too few for results. 50k/5k trace setting is overwritten by `set_dataset`. |
| `script_aes_neural_network_parameters` | MLP with own size settings | 3×100 relu MLP, batch 100, 15 epochs | **Yes.** |
| `script_aes_multi_model` | Compare 5 models in one run | `cnn_architecture`, `mlp`, `noConv1_ascad_desync_0`, `methodology_cnn_ascad`, `cnn`; 50 epochs | **Yes.** Slow (5 models × 50 epochs). |
| `script_aes_custom_loss` | Different loss function | `mlp` trained with CER loss instead of cross-entropy | **Yes.** |
| `script_aes_custom_metrics` | Early stopping on guessing entropy | `mlp`, 5 epochs; keeps the epoch with the best GE | **Yes.** Ranks with only 50 attack traces, so GE is noisy. |
| `script_aes_data_augmentation` | Training with added noise | `mlp` + Gaussian-noise augmentation, 100 extra batches | **Yes.** |
| `script_aes_profiling_analyzer` | How much profiling data is needed | Retrains `mlp` on 1k / 5k / 10k / 50k traces | **Yes.** |
| `script_aes_grid_search` | CNN hyperparameter search | 4 combinations (2–3 conv layers × 1–2 dense layers), ranked by GE | **Yes.** Small search. |
| `script_aes_custom_table` | Demo: add own database table | Trains `mlp`, then writes dummy values 10/20/30 to a new table | **Runs, no research value.** Writes into the real database; keep out of experiments. |
| `script_aes_new_dataset` | Demo: plug in own dataset | Random traces and labels (10k × 100), 256 classes | **Runs, no research value.** Random data, so GE is meaningless. Writes `database_ascad_variable.sqlite`. |
| `script_aes_ensembles` | Grid search + ensemble of the 10 best | MLP grid of 16 combinations, then combines the top 10 | **No.** `KeyError: 'structure'`: grid search dict needs a `"structure"` block. |
| `script_aes_random_search` | CNN random search | 50 random CNN trials | **No.** Uses `ascad-variable.h5`, which is missing. Switch to `ASCAD.h5`. |
| `script_aes_visualization` | Which trace samples the network uses | Input-gradient maps on 4,000 traces | **No.** Same missing file. Switch to `ASCAD.h5`. |
| `script_aes_custom_callback` | Demo: own Keras callbacks | Two example callbacks that only print their parameters | **No.** Missing file; demo anyway. |
| `script_aes_cnn_new` | Standalone CNN without AISY | Own loading, training and rank code | **Retired.** Not AISY; its rank metric does not measure key recovery. |

### Running instructions (group 30)

All commands are for PowerShell on Windows. Run them from the `AISY_Framework` folder; the scripts use relative paths (`data/`, `databases/`, `resources/`).

**1. Activate the environment** (from the project root, `Deep Learning SCA group 30`):

```
.\aisy_env\Scripts\python.exe --version     # must print Python 3.7.6
.\aisy_env\Scripts\Activate.ps1
$env:Path = "$PWD\aisy_env\Scripts;$env:Path"
cd AISY_Framework
```

If the environment is broken, see `README.md` in the project root to rebuild it.

**2. Run one script.** Use `-m` so Python finds the `custom` and `scripts` packages (running the file directly gives `ModuleNotFoundError: No module named 'custom'`):

```
python -m scripts.script_aes_cnn
```

Each run prints `SEED = ...`. The seed is random unless `SCA_SEED` is set. To repeat a run or use one seed for several scripts:

```
$env:SCA_SEED = "616134"
python -m scripts.script_aes
python -m scripts.script_aes_cnn
Remove-Item Env:SCA_SEED     # back to a random seed
```

**3. Run several scripts as one experiment** with one shared seed:

```
python -m scripts.run_experiment                                   # default: script_aes, script_aes_neural_network_parameters, script_aes_cnn, script_aes_multi_model
python -m scripts.run_experiment --seed 616134 --scripts script_aes script_aes_cnn
```

Logs: `resources/experiments/<experiment_id>/<script>.log`. One row per script in `resources/experiments/experiment_log.csv`.

**4. Phase 0: compare MLP and CNN under one protocol** over several seeds.

Command: `python -m scripts.phase0_compare`, followed by any of these options. An option left out uses its default.

| Option | Meaning | Default |
|---|---|---|
| `--n-seeds N` | Generate N random seeds | 3 |
| `--seeds S1 S2 ...` | Use these seeds instead of generating them | none |
| `--epochs N` | Epochs per run | 50 |
| `--batch-size N` | Batch size for every model | 128 |
| `--models mlp cnn` | Models to compare | both |

Every seed trains every model, so the number of runs is seeds × models.

Example A, quick timing check first (1 seed, 10 epochs, about 15 minutes):

```
python -m scripts.phase0_compare --n-seeds 1 --epochs 10
```

Example B, full default run (3 random seeds, 50 epochs, batch size 128, about 2–3 hours on CPU):

```
python -m scripts.phase0_compare
```

Example C, repeat a run with known seeds (2 seeds given, so 2 seeds × 2 models = 4 runs):

```
python -m scripts.phase0_compare --seeds 616134 4242 --epochs 50 --batch-size 128
```

Output: `resources/experiments/phase0_<timestamp>/runs.csv` (one row per model and seed), `summary.md` (mean ± std per model) and `log.txt` (everything printed during the run).

In phase 0, for increased no key traces and an auto log folder  creation run: python.exe -c "import sys, scripts.phase0_compare as p; p.PROTOCOL.update(leakage_model='ID', key_rank_attack_traces=3000); sys.argv = ['phase0_compare', '--n-seeds', '3', '--epochs', '50']; p.main()" in aisy_framework 

**5. View results in the AISY webapp:**

```
flask run
```

Open http://127.0.0.1:5000. Every run is stored in `databases/database_ascad.sqlite`; the analysis id printed by `phase0_compare` matches the id in the webapp.

Do not run `script_aes_custom_table` for experiments: it writes a demo table into the real database.

- In 
