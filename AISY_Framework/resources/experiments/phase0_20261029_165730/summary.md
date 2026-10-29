# Phase 0: MLP vs CNN (2026-10-29 18:28)

Seeds: 120936, 55963, 519146. Epochs: 50 (best epoch kept by validation GE). Batch size: 128.
Data: ASCAD.h5, byte 2, HW leakage model, 50000 profiling traces, attack set split into 5,000 validation + 5,000 attack.
GE: mean of 100 key ranks on 1000 attack traces. 'Traces to GE < 2' averages only runs that reach it.

| Model | Runs OK | Reached GE < 2 (best epoch) | Traces to GE < 2 (best epoch) | Final GE (best epoch) | Final SR (best epoch) | Final GE (last epoch) |
|---|---|---|---|---|---|---|
| mlp | 3/3 | 0/3 | n/a | 18.1 ± 13.1 | 0.3 ± 0.2 | 17.1 ± 14.6 |
| cnn | 3/3 | 0/3 | n/a | 105.0 ± 27.1 | 0.0 ± 0.0 | 110.3 ± 35.0 |
