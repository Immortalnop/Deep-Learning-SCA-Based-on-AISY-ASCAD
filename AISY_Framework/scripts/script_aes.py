import aisy_sca
from scripts.experiment_output import configure_aisy, finish_run, start_run
from scripts.experiment_seed import apply_seed
from app import *
from custom.custom_models.neural_networks import *

SEED = apply_seed()
RUN_FOLDER = start_run("script_aes", SEED)

aisy = aisy_sca.Aisy()
configure_aisy(aisy, RUN_FOLDER)
aisy.set_dataset(datasets_dict["ASCAD.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.add_neural_network(mlp, seed=SEED)
aisy.run()
finish_run(RUN_FOLDER, "script_aes", SEED, [aisy.settings["analysis_id"]])
