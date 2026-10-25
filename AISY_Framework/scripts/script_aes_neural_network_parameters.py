import aisy_sca
from scripts.experiment_output import configure_aisy, finish_run, start_run
from scripts.experiment_seed import apply_seed
from app import *
from custom.custom_models.neural_networks import *

# Defaults hold this script's settings; AISY calls mlp(classes, number_of_samples)
def mlp(classes, number_of_samples, neuron=100, layer=3, activation="relu", learning_rate=0.001):
    model = Sequential(name="my_mlp")
    for l_i in range(layer):
        if l_i == 0:
            model.add(Dense(neuron, activation=activation, input_shape=(number_of_samples,)))
        else:
            model.add(Dense(neuron, activation=activation))
    model.add(Dense(classes, activation=None))
    model.add(Activation(activation="softmax"))
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


SEED = apply_seed()
RUN_FOLDER = start_run("script_aes_neural_network_parameters", SEED)

aisy = aisy_sca.Aisy()
configure_aisy(aisy, RUN_FOLDER)
aisy.set_dataset(datasets_dict["ASCAD.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(100)
aisy.set_epochs(15)
aisy.add_neural_network(mlp, name="my_mlp", seed=SEED)

aisy.run()
finish_run(RUN_FOLDER, "script_aes_neural_network_parameters", SEED, [aisy.settings["analysis_id"]])
