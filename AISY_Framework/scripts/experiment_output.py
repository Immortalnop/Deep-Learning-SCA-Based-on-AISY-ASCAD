# One output folder per run, and one index of all runs.
#
# Process: start_run makes resources/experiments/<kind>_<timestamp>[_seed<N>]/ -> AISY uses it as its
# resources root (configure_aisy); its best-epoch .h5 files live in models/ there and AISY deletes them
# after evaluation -> finish_run removes the empty AISY folders and adds a row to
# resources/experiments/index.csv. The AISY database stays central in databases/; the index row
# keeps the analysis ids, which link the run folder to its database rows.
# Under run_experiment.py the folder comes from the SCA_RUN_FOLDER environment variable,
# and run_experiment.py writes the index row instead.
#
# Usage in an AISY script:
#     from scripts.experiment_output import configure_aisy, finish_run, start_run
#     RUN_FOLDER = start_run("script_aes", SEED)
#     configure_aisy(aisy, RUN_FOLDER)
#     aisy.run()
#     finish_run(RUN_FOLDER, "script_aes", SEED, [aisy.settings["analysis_id"]])
import csv
import os
import stat
import time
from datetime import datetime

# Forward slashes: AISY appends "models/..." to the resources root folder.
EXPERIMENTS_FOLDER = "resources/experiments"
INDEX_FILE = f"{EXPERIMENTS_FOLDER}/index.csv"
INDEX_COLUMNS = ["run_id", "kind", "seed", "status", "started", "seconds", "analysis_ids", "folder"]
RUN_FOLDER_ENV = "SCA_RUN_FOLDER"
AISY_FOLDERS = ["databases", "figures", "models", "npz"]  # created by AISY in the resources root

_started = {}  # folder -> start time, for finish_run


def new_run_folder(kind, seed=None):
    run_id = f"{kind}_{datetime.now():%Y%m%d_%H%M%S}" + (f"_seed{seed}" if seed is not None else "")
    folder = f"{EXPERIMENTS_FOLDER}/{run_id}"
    os.makedirs(folder, exist_ok=True)
    return folder


def from_runner():
    return RUN_FOLDER_ENV in os.environ


def start_run(kind, seed=None):
    if from_runner():
        folder = os.environ[RUN_FOLDER_ENV].replace("\\", "/")
        os.makedirs(folder, exist_ok=True)
    else:
        folder = new_run_folder(kind, seed)
    _started[folder] = (datetime.now().isoformat(timespec="seconds"), time.time())
    print(f"RUN_FOLDER = {folder}")
    return folder


def resume_run(folder):
    # Continue writing into an existing run folder (a study run in several stages); finish_run adds one index row
    # per stage.
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"run folder not found: {folder}")
    folder = folder.replace("\\", "/").rstrip("/")
    _started[folder] = (datetime.now().isoformat(timespec="seconds"), time.time())
    print(f"RUN_FOLDER = {folder}")
    return folder


def configure_aisy(aisy, folder, database_name="database_ascad.sqlite"):
    aisy.set_resources_root_folder(folder + "/")
    aisy.set_database_root_folder("databases/")
    aisy.set_datasets_root_folder("data/")
    aisy.set_database_name(database_name)


def remove_empty_folders(folder):
    # OneDrive marks new folders read-only, and os.rmdir then fails on Windows -> clear the flag first.
    # Never fail the run over a leftover empty folder.
    for name in AISY_FOLDERS:
        path = os.path.join(folder, name)
        if not os.path.isdir(path) or os.listdir(path):
            continue
        try:
            os.chmod(path, stat.S_IWRITE)
            os.rmdir(path)
        except OSError as e:
            print(f"Could not remove empty folder {path}: {e}")


def append_index(row):
    new_file = not os.path.exists(INDEX_FILE)
    with open(INDEX_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def finish_run(folder, kind, seed=None, analysis_ids=(), status="ok"):
    for analysis_id in analysis_ids:
        print(f"ANALYSIS_ID = {analysis_id}")  # run_experiment.py reads this from the log
    if from_runner():
        return  # run_experiment.py cleans up and writes the index row
    remove_empty_folders(folder)
    started, start_time = _started.pop(folder, ("", time.time()))
    append_index({"run_id": os.path.basename(folder), "kind": kind, "seed": seed, "status": status,
                  "started": started, "seconds": round(time.time() - start_time, 1),
                  "analysis_ids": " ".join(str(a) for a in analysis_ids), "folder": folder})
