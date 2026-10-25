# Run a set of AISY scripts as one experiment with one shared seed.
#
# Process: generate seed -> every script reads it (scripts/experiment_seed.py).
# Run from AISY_Framework:
#     ..\aisy_env\Scripts\python.exe -m scripts.run_experiment
#     ..\aisy_env\Scripts\python.exe -m scripts.run_experiment --seed 616134
#     ..\aisy_env\Scripts\python.exe -m scripts.run_experiment --scripts script_aes script_aes_cnn
#
# Each run writes resources/experiments/experiment_<timestamp>_seed<N>/<script>/
# (log.txt; folder passed to the script in SCA_RUN_FOLDER)
# and appends one row per script to resources/experiments/index.csv.
import argparse
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime

from scripts.experiment_output import RUN_FOLDER_ENV, append_index, new_run_folder, remove_empty_folders
from scripts.experiment_seed import SEED_ENV, SEED_RANGE

# Core experiment scripts, run in this order by default.
EXPERIMENT_SCRIPTS = [
    "script_aes",
    "script_aes_neural_network_parameters",
    "script_aes_cnn",
    "script_aes_multi_model",
]

ANALYSIS_ID_LINE = re.compile(r"^ANALYSIS_ID = (\S+)")  # printed by experiment_output.finish_run


def run_script(script, env, log_path):
    start = time.time()
    analysis_ids = []
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", f"scripts.{script}"], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            match = ANALYSIS_ID_LINE.match(line)
            if match:
                analysis_ids.append(match.group(1))
        return_code = process.wait()
    return return_code, round(time.time() - start, 1), analysis_ids


def main():
    parser = argparse.ArgumentParser(description="Run AISY scripts with one shared seed.")
    parser.add_argument("--seed", type=int, help="reuse a seed (default: generate a new one)")
    parser.add_argument("--scripts", nargs="+", default=EXPERIMENT_SCRIPTS,
                        help="script names without .py (default: EXPERIMENT_SCRIPTS)")
    args = parser.parse_args()

    for script in args.scripts:
        if not os.path.exists(os.path.join("scripts", f"{script}.py")):
            sys.exit(f"Unknown script: {script} (run from AISY_Framework)")

    seed = args.seed if args.seed is not None else random.randrange(SEED_RANGE)
    experiment_folder = new_run_folder("experiment", seed)
    experiment_id = os.path.basename(experiment_folder)

    print(f"Experiment {experiment_id}: {len(args.scripts)} scripts, SEED = {seed}")
    results = []
    for script in args.scripts:
        print(f"\n===== {script} (seed {seed}) =====")
        script_folder = f"{experiment_folder}/{script}"
        os.makedirs(script_folder, exist_ok=True)
        env = dict(os.environ, **{SEED_ENV: str(seed), RUN_FOLDER_ENV: script_folder,
                                  "PYTHONPATH": os.getcwd(), "PYTHONUNBUFFERED": "1"})
        started = datetime.now().isoformat(timespec="seconds")
        return_code, seconds, analysis_ids = run_script(script, env, f"{script_folder}/log.txt")
        remove_empty_folders(script_folder)  # also when the script failed before finish_run
        status = "ok" if return_code == 0 else f"failed ({return_code})"
        results.append((script, status, seconds))
        append_index({"run_id": experiment_id, "kind": script, "seed": seed, "status": status,
                      "started": started, "seconds": seconds, "analysis_ids": " ".join(analysis_ids),
                      "folder": script_folder})

    print(f"\n===== Experiment {experiment_id} summary (seed {seed}) =====")
    for script, status, seconds in results:
        print(f"{status:12} {seconds:8.1f}s  {script}")
    print(f"Results: {experiment_folder}")


if __name__ == "__main__":
    main()
