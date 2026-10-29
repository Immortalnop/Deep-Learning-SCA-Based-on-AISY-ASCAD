# Phase 5: explainability (2026-10-29 23:34)

Models: Phase 4 desync 0 recipe retrained with the same seed (d0_cnn_s1, d0_cnn_s2, d0_cnn_s3). Saliency = mean |d log p(correct class) / d sample| over 2000 attack traces. SNR peaks (Phase 1): masked value at sample 1071, mask at sample 188; windows of +-5 samples.

## Retraining against Phase 4 (attack at desync 0)

| run | retrained_ge_area | phase4_ge_area | retrained_traces_ge_lt_2 | phase4_traces_ge_lt_2 |
|---|---|---|---|---|
| d0_cnn_s1 | 8.3 | 9.19 | 400 | 430 |
| d0_cnn_s2 | 7.06 | 8.04 | 340 | 350 |
| d0_cnn_s3 | 7.39 | 8.5 | 360 | 400 |

Phase 4 reference: resources/experiments/phase4_20261028_233344. Small differences are expected: CPU training is not bit-exact.

## Overlap of the saliency with the SNR peaks

| run | variant | attack_desync | traces | spearman_masked_value | spearman_mask | enrichment | samples_in_windows | percentile_at_masked_value_peak | percentile_at_mask_peak |
|---|---|---|---|---|---|---|---|---|---|
| d0_cnn_s1 | cnn | 0 | 2000 | 0.235 | 0.143 | 2.91 | 22 | 100.0 | 99.9 |
| d0_cnn_s1 | cnn | 50 | 2000 | 0.198 | 0.13 | 2.26 | 22 | 99.6 | 100.0 |
| d0_cnn_s1 | cnn | 100 | 2000 | 0.179 | 0.156 | 2.19 | 22 | 97.9 | 100.0 |
| d0_cnn_s2 | cnn | 0 | 2000 | 0.242 | 0.147 | 3.63 | 22 | 99.8 | 100.0 |
| d0_cnn_s2 | cnn | 50 | 2000 | 0.197 | 0.171 | 2.81 | 22 | 99.8 | 99.6 |
| d0_cnn_s2 | cnn | 100 | 2000 | 0.2 | 0.15 | 2.68 | 22 | 99.8 | 99.4 |
| d0_cnn_s3 | cnn | 0 | 2000 | 0.216 | 0.133 | 2.61 | 22 | 99.3 | 100.0 |
| d0_cnn_s3 | cnn | 50 | 2000 | 0.177 | 0.158 | 2.09 | 22 | 99.4 | 100.0 |
| d0_cnn_s3 | cnn | 100 | 2000 | 0.205 | 0.131 | 1.81 | 22 | 99.4 | 100.0 |

## Occlusion (attack at desync 0, mean over random draws for `random`)

| run | variant | occluded | samples | ge_area | ge_final | reached_ge_lt_2 | traces_ge_lt_2 |
|---|---|---|---|---|---|---|---|
| d0_cnn_s1 | cnn | none | 0 | 8.3 | 1.0 | 1/1 | 400.0 |
| d0_cnn_s1 | cnn | peaks | 22 | 12.37 | 1.01 | 1/1 | 570.0 |
| d0_cnn_s1 | cnn | top | 22 | 18.97 | 1.14 | 1/1 | 860.0 |
| d0_cnn_s1 | cnn | random | 22 | 8.26 | 1.0 | 5/5 | 392.0 |
| d0_cnn_s2 | cnn | none | 0 | 7.06 | 1.0 | 1/1 | 340.0 |
| d0_cnn_s2 | cnn | peaks | 22 | 11.81 | 1.0 | 1/1 | 540.0 |
| d0_cnn_s2 | cnn | top | 22 | 16.76 | 1.21 | 1/1 | 790.0 |
| d0_cnn_s2 | cnn | random | 22 | 7.13 | 1.0 | 5/5 | 338.0 |
| d0_cnn_s3 | cnn | none | 0 | 7.39 | 1.0 | 1/1 | 360.0 |
| d0_cnn_s3 | cnn | peaks | 22 | 16.37 | 1.57 | 1/1 | 900.0 |
| d0_cnn_s3 | cnn | top | 22 | 17.6 | 1.76 | 1/1 | 980.0 |
| d0_cnn_s3 | cnn | random | 22 | 7.47 | 1.0 | 5/5 | 356.0 |

## Meaning

- `d0_cnn_s1`: GE area without occlusion 8.3; occluding the SNR peaks 12.37, the same number of top-saliency samples 18.97, random samples 8.26.
- `d0_cnn_s2`: GE area without occlusion 7.06; occluding the SNR peaks 11.81, the same number of top-saliency samples 16.76, random samples 7.13.
- `d0_cnn_s3`: GE area without occlusion 7.39; occluding the SNR peaks 16.37, the same number of top-saliency samples 17.6, random samples 7.47.

Files: `saliency.png`, `occlusion.png`, `saliency.npz`, `overlap.csv`, `occlusion.csv`, `models/`, `runs.csv`, `eval.csv`.
