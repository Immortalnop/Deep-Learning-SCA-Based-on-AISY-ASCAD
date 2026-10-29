# Phase 4: robustness to desynchronisation (2026-10-29 22:35)

Data: ASCAD v1 variable key, byte 2, 50000 profiling traces, attack traces 5000-9999, validation traces 0-4999; desync 0 / 50 / 100 files ASCAD.h5, ascad-variable-desync50.h5, ascad-variable-desync100.h5.
Training: up to 50 epochs, early stopping on the validation GE area of the training desync level, stop after 10 epochs without improvement; best epoch kept. Seeds 1, 2, 3.
Attack metrics: the same key rank and 100 attack subsets (seed 0) as Phases 2 and 3. The spread over seeds (independent trainings) is the one to trust.

## Experiment A: trained on desync 0, attacked at each desync level

| variant | train_desync | attack_desync | seeds | reached_ge_lt_2 | traces_ge_lt_2 | att_ge_final | att_ge_area | parameters |
|---|---|---|---|---|---|---|---|---|
| mlp | 0 | 0 | 3 | 0/3 | n/a | 17.3 ± 4.3 | 46.7 ± 4.8 | 402609 |
| cnn | 0 | 0 | 3 | 3/3 | 393.3 ± 40.4 | 1.0 ± 0.0 | 8.6 ± 0.6 | 404105 |
| cnn_aug | 0 | 0 | 3 | 0/3 | n/a | 42.9 ± 66.5 | 60.2 ± 54.6 | 404105 |
| cnn_wide | 0 | 0 | 3 | 3/3 | 330.0 ± 30.0 | 1.0 ± 0.0 | 5.9 ± 1.0 | 417305 |
| cnn_gap | 0 | 0 | 3 | 0/3 | n/a | 163.9 ± 20.0 | 154.9 ± 15.1 | 51849 |
| mlp | 0 | 50 | 3 | 0/3 | n/a | 120.0 ± 17.4 | 122.3 ± 11.5 | 402609 |
| cnn | 0 | 50 | 3 | 0/3 | n/a | 127.5 ± 12.4 | 131.6 ± 8.8 | 404105 |
| cnn_aug | 0 | 50 | 3 | 0/3 | n/a | 144.5 ± 23.0 | 140.3 ± 16.6 | 404105 |
| cnn_wide | 0 | 50 | 3 | 0/3 | n/a | 102.6 ± 15.0 | 113.2 ± 9.3 | 417305 |
| cnn_gap | 0 | 50 | 3 | 0/3 | n/a | 117.7 ± 13.6 | 119.9 ± 8.1 | 51849 |
| mlp | 0 | 100 | 3 | 0/3 | n/a | 122.6 ± 9.0 | 127.5 ± 5.6 | 402609 |
| cnn | 0 | 100 | 3 | 0/3 | n/a | 119.0 ± 30.0 | 123.1 ± 18.3 | 404105 |
| cnn_aug | 0 | 100 | 3 | 0/3 | n/a | 147.3 ± 17.4 | 141.0 ± 14.9 | 404105 |
| cnn_wide | 0 | 100 | 3 | 0/3 | n/a | 91.9 ± 8.1 | 107.7 ± 4.6 | 417305 |
| cnn_gap | 0 | 100 | 3 | 0/3 | n/a | 128.3 ± 3.9 | 129.2 ± 1.6 | 51849 |

## Experiment B: trained and attacked on the same desync level

| variant | train_desync | attack_desync | seeds | reached_ge_lt_2 | traces_ge_lt_2 | att_ge_final | att_ge_area | parameters |
|---|---|---|---|---|---|---|---|---|
| mlp | 50 | 50 | 3 | 0/3 | n/a | 115.4 ± 1.8 | 123.1 ± 0.9 | 402609 |
| cnn | 50 | 50 | 3 | 0/3 | n/a | 139.5 ± 19.3 | 133.5 ± 9.7 | 404105 |
| cnn_aug | 50 | 50 | 3 | 0/3 | n/a | 149.7 ± 32.8 | 141.6 ± 20.6 | 404105 |
| cnn_wide | 50 | 50 | 3 | 0/3 | n/a | 115.6 ± 45.6 | 116.6 ± 30.6 | 417305 |
| cnn_gap | 50 | 50 | 3 | 0/3 | n/a | 113.7 ± 13.6 | 117.6 ± 9.6 | 51849 |
| mlp | 100 | 100 | 3 | 0/3 | n/a | 129.0 ± 20.9 | 132.1 ± 12.1 | 402609 |
| cnn | 100 | 100 | 3 | 0/3 | n/a | 101.4 ± 38.8 | 108.8 ± 23.0 | 404105 |
| cnn_aug | 100 | 100 | 3 | 0/3 | n/a | 137.2 ± 39.0 | 132.7 ± 24.7 | 404105 |
| cnn_wide | 100 | 100 | 3 | 0/3 | n/a | 122.2 ± 39.7 | 123.7 ± 25.4 | 417305 |
| cnn_gap | 100 | 100 | 3 | 0/3 | n/a | 118.4 ± 12.8 | 124.6 ± 10.8 | 51849 |

## Meaning

- Attack on desync 0, trained on desync 0: best `cnn_wide` (GE < 2 in 3/3 seeds, traces 330.0 ± 30.0). Seeds reaching GE < 2: mlp 0/3, cnn 3/3, cnn_aug 0/3, cnn_wide 3/3, cnn_gap 0/3.
- Attack on desync 50, trained on desync 0: best `cnn_wide` (GE < 2 in 0/3 seeds, traces n/a). Seeds reaching GE < 2: mlp 0/3, cnn 0/3, cnn_aug 0/3, cnn_wide 0/3, cnn_gap 0/3.
- Attack on desync 50, trained on the same level: best `cnn_gap` (GE < 2 in 0/3 seeds, traces n/a). Seeds reaching GE < 2: mlp 0/3, cnn 0/3, cnn_aug 0/3, cnn_wide 0/3, cnn_gap 0/3.
- Attack on desync 100, trained on desync 0: best `cnn_wide` (GE < 2 in 0/3 seeds, traces n/a). Seeds reaching GE < 2: mlp 0/3, cnn 0/3, cnn_aug 0/3, cnn_wide 0/3, cnn_gap 0/3.
- Attack on desync 100, trained on the same level: best `cnn` (GE < 2 in 0/3 seeds, traces n/a). Seeds reaching GE < 2: mlp 0/3, cnn 0/3, cnn_aug 0/3, cnn_wide 0/3, cnn_gap 0/3.

Files: `results.csv`, `eval.csv`, `ge_desync.png`, `runs.csv`, `predictions/`, `tensorboard/`.
