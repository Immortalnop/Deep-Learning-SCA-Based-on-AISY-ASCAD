# NOTE: Run ".\aisy_env\Scripts\python.exe --version" followed by ".\aisy_env\Scripts\Activate.ps1" 
# followed by  "$env:Path = "$PWD\aisy_env\Scripts;$env:Path""
#ELSE See readme.txt[not .md]

#// RETIRING THIS: the per-trace HW-class rank is not a key-recovery metric.//
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling1D, Conv1D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# -----------------------------
# HAMMING WEIGHT
# -----------------------------
def hamming_weight(x):
    return bin(int(x)).count("1")


# -----------------------------
# LOAD + PREPROCESS ASCAD
# -----------------------------
def load_ascad(filepath, n_profiling=20000, n_attack=5000):

    f = h5py.File(filepath, 'r')

    X_profiling = np.array(f['Profiling_traces/traces'][:n_profiling])
    Y_profiling = np.array(f['Profiling_traces/labels'][:n_profiling])

    X_attack = np.array(f['Attack_traces/traces'][:n_attack])
    Y_attack = np.array(f['Attack_traces/labels'][:n_attack])

    # -----------------------------
    # CONVERT LABELS → HW (0–8)
    # -----------------------------
    Y_profiling = np.array([hamming_weight(x) for x in Y_profiling])
    Y_attack = np.array([hamming_weight(x) for x in Y_attack])

    # -----------------------------
    # NORMALIZE 
    # -----------------------------
    mean = np.mean(X_profiling, axis=0)
    std = np.std(X_profiling, axis=0) + 1e-8

    X_profiling = (X_profiling - mean) / std
    X_attack = (X_attack - mean) / std

    # -----------------------------
    # RESHAPE FOR CNN
    # -----------------------------
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    return X_profiling, Y_profiling, X_attack, Y_attack


# -----------------------------
# CNN MODEL
# -----------------------------
def build_model(input_size):
    model = Sequential()

    model.add(Conv1D(16, 11, activation='relu', input_shape=(input_size, 1)))
    model.add(AveragePooling1D(2))

    model.add(Conv1D(32, 11, activation='relu'))
    model.add(AveragePooling1D(2))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(9, activation='softmax'))  # HW classes

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(0.001),
        metrics=['accuracy']
    )

    return model


# -----------------------------
# COMPUTE RANK 
# -----------------------------
def compute_rank(preds, true_labels):
    ranks = []

    for i in range(len(preds)):
        probs = preds[i]
        true = true_labels[i]

        rank = np.sum(probs > probs[true])
        ranks.append(rank)

    return np.array(ranks)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading dataset...")
    X_profiling, Y_profiling, X_attack, Y_attack = load_ascad("data/ASCAD.h5")

    print("Building model...")
    model = build_model(X_profiling.shape[1])
    model.summary()

    print("Training...")
    history = model.fit(
        X_profiling, Y_profiling,
        batch_size=64,
        epochs=10,
        validation_split=0.1,
        verbose=2
    )

    print("Predicting...")
    preds = model.predict(X_attack)

    print("Computing rank...")
    ranks = compute_rank(preds, Y_attack)

    print("\n===== RESULTS =====")
    print("Accuracy (last epoch):", history.history['accuracy'][-1])
    print("Average Rank:", np.mean(ranks))

    # Plot rank evolution
    plt.plot(ranks[:500])
    plt.title("Rank progression (first 500 traces)")
    plt.xlabel("Trace index")
    plt.ylabel("Rank")
    plt.show()