# Shared seed for one experiment set.
#
# Process: run_experiment.py generates ONE seed -> every script reads it here.
# The seed travels in the SCA_SEED environment variable.
# Run a script on its own (SCA_SEED not set) -> a fresh random seed is generated.
#
# Usage at the top of a script:
#     from scripts.experiment_seed import apply_seed
#     SEED = apply_seed()
# and pass it to AISY so it is stored with the run:
#     aisy.add_neural_network(mlp, seed=SEED)
import os
import random

import tensorflow as tf

SEED_ENV = "SCA_SEED"
SEED_RANGE = 1048576  # same range AISY uses for its own seeds


def get_seed():
    if SEED_ENV not in os.environ:
        os.environ[SEED_ENV] = str(random.randrange(SEED_RANGE))
    return int(os.environ[SEED_ENV])


def set_seed(seed):
    tf.keras.utils.set_random_seed(seed)            # Python random, numpy and TF
    tf.config.experimental.enable_op_determinism()  # repeatable TF ops
    print(f"SEED = {seed}")


def apply_seed():
    seed = get_seed()
    set_seed(seed)
    return seed
