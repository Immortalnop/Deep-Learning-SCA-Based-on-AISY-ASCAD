# Phase 1: SNR leakage analysis (2026-10-21 00:46)

Data: ASCAD.h5 (`data/ASCAD.h5`), byte 2, HW classes, first 50000 profiling traces, all 1400 samples. Noise floor (peak SNR of random labels): 0.00466. A peak counts as leakage above 2 x the noise floor.

## Checks

- File labels equal Sbox(p[2] ^ k[2]) for 100.0% of the first 5000 profiling traces.
- Profiling key: variable (5000 distinct keys in 5000 traces).
- Attack key byte 2 = 0x22, same as the `key` in `datasets.py`.
- AISY window covers all 1400 samples in the file.

## Peaks

| Target | Peak sample | Peak SNR | x noise floor | Leaks? | Peak in AISY window? |
|---|---|---|---|---|---|
| sbox | 1012 | 0.00415 | 0.9 | no | yes |
| sbox_xor_r | 1071 | 3.22621 | 692.0 | yes | yes |
| r | 188 | 2.00343 | 429.7 | yes | yes |
| sbox_xor_rout | 1000 | 4.66517 | 1000.6 | yes | yes |
| rout | 188 | 0.58300 | 125.0 | yes | yes |

## Meaning

- Unmasked value: no first-order SNR peak.
- Shares `sbox_xor_r` (sample 1071) and `r` (sample 188) both leak, 883 samples apart. A model must combine both points to recover the key.
- Shares `sbox_xor_rout` (sample 1000) and `rout` (sample 188) both leak, 812 samples apart. A model must combine both points to recover the key.
- CNN design: the receptive field (or the dense layers after `Flatten`) must span the distance between the two share peaks of a pair.

Files: `snr.png`, `snr_vs_traces.png`, `peaks.csv`, `snr.npz` (keys `<target>_<traces>`).
