from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as backend
import tensorflow as tf
import numpy as np


class CustomCallback1(Callback):
    def __init__(self, dataset, settings, args):
        super().__init__()
        self.param1 = args["param1"]
        self.param2 = args["param2"]

    def on_epoch_end(self, epoch, logs=None):
        print("Processing epoch {}".format(epoch))

    def on_train_end(self, logs=None):
        pass

    def get_param1(self):
        return self.param1

    def get_param2(self):
        return self.param2


class CustomCallback2(Callback):
    def __init__(self, dataset, settings, args):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print("Processing epoch {}".format(epoch))

    def on_train_end(self, logs=None):
        pass

    # def get_param1(self):
    #     return self.param1
    #
    # def get_param2(self):
    #     return self.param2


class SaveWeights(Callback):
    def __init__(self, dataset, settings, *args):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.get_weights())

    def get_weights(self):
        return self.weights


class PlateauStop(Callback):
    # Stops training when an early-stopping metric (lower = better) has not improved for `patience` epochs.
    # The metric module keeps its value per epoch in settings[args["history_key"]] (custom_metrics/ge_area.py);
    # AISY creates its early-stopping callback before the custom ones, so the value of this epoch is already there.
    # AISY's early stopping keeps the best epoch's weights, so stopping only saves time.
    def __init__(self, dataset, settings, args):
        super().__init__()
        self.settings = settings
        self.history_key = args["history_key"]
        self.patience = args["patience"]
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        history = self.settings.get(self.history_key, [])
        if len(history) > self.patience and min(history[-self.patience:]) >= min(history[:-self.patience]):
            self.stopped_epoch = epoch + 1
            self.model.stop_training = True
            print(f"PlateauStop: no improvement for {self.patience} epochs, stopped after epoch {epoch + 1}")


class TensorBoardLog(tf.keras.callbacks.TensorBoard):
    # Keras TensorBoard (loss and accuracy per epoch, training and validation) in the form AISY creates custom
    # callbacks. View with: ..\aisy_env\Scripts\tensorboard.exe --logdir <run folder>/tensorboard
    def __init__(self, dataset, settings, args):
        super().__init__(log_dir=args["log_dir"], histogram_freq=0, write_graph=False, profile_batch=0)
