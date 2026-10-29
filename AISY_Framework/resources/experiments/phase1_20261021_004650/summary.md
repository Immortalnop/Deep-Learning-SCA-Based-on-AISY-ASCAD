# Phase 1: SNR leakage analysis (2026-10-21 00:47)

Data: ASCAD.h5 (`data/ASCAD.h5`), byte 2, ID classes, first 50000 profiling traces, all 1400 samples. Noise floor (peak SNR of random labels): 0.00685. A peak counts as leakage above 2 x the noise floor.

## Checks

- File labels equal Sbox(p[2] ^ k[2]) for 100.0% of the first 5000 profiling traces.
- Profiling key: variable (5000 distinct keys in 5000 traces).
- Attack key byte 2 = 0x22, same as the `key` in `datasets.py`.
- AISY window covers all 1400 samples in the file.

## Peaks

| Target | Peak sample | Peak SNR | x noise floor | Leaks? | Peak in AISY window? |
|---|---|---|---|---|---|
| sbox | 492 | 0.00685 | 1.0 | no | yes |
| sbox_xor_r | 1071 | 1.53856 | 224.5 | yes | yes |
| r | 188 | 1.15213 | 168.1 | yes | yes |
| sbox_xor_rout | 1000 | 1.40185 | 204.6 | yes | yes |
| rout | 188 | 0.19010 | 27.7 | yes | yes |

## Meaning

- Unmasked value: no first-order SNR peak.
- Shares `sbox_xor_r` (sample 1071) and `r` (sample 188) both leak, 883 samples apart. A model must combine both points to recover the key.
- Shares `sbox_xor_rout` (sample 1000) and `rout` (sample 188) both leak, 812 samples apart. A model must combine both points to recover the key.
- CNN design: the receptive field (or the dense layers after `Flatten`) must span the distance between the two share peaks of a pair.

Files: `snr.png`, `snr_vs_traces.png`, `peaks.csv`, `snr.npz` (keys `<target>_<traces>`).
