import aisy_sca
from scripts.experiment_output import configure_aisy, finish_run, start_run
from scripts.experiment_seed import apply_seed
from app import *
from custom.custom_models.neural_networks import *

SEED = apply_seed()
RUN_FOLDER = start_run("script_aes_multi_model", SEED)

aisy = aisy_sca.Aisy()
configure_aisy(aisy, RUN_FOLDER)
aisy.set_dataset(datasets_dict["ASCAD.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(50)
aisy.add_neural_network(cnn_architecture, name="model_0", seed=SEED)
aisy.add_neural_network(mlp, name="model_3", seed=SEED)
aisy.add_neural_network(noConv1_ascad_desync_0, name="model_1", seed=SEED)
aisy.add_neural_network(methodology_cnn_ascad, name="model_2", seed=SEED)
aisy.add_neural_network(cnn, name="model_4", seed=SEED)
aisy.run()
finish_run(RUN_FOLDER, "script_aes_multi_model", SEED, [aisy.settings["analysis_id"]])
