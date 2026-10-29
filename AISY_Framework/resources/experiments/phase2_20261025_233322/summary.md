# Phase 2: classic baselines (2026-10-25 23:37)

Data: ASCAD.h5, byte 2 (attack key 0x22), 50000 profiling traces, attack traces 5000-9999 (the AISY attack set when early stopping uses the first half for validation), AISY sample window.
GE: mean key rank over 100 random subsets of 1000 attack traces (same subsets for every attack, seed 0), every 10 traces; tied key guesses count half. Same protocol as Phase 0.
Templates: ID and HW classes, PCA 20 components, QDA (uniform priors, reg_param 0.01). Windows: +/- 5 samples around the share SNR peaks sbox_xor_r 1071/1170, r 188/62, sbox_xor_rout 1000/1020, rout 188/125 (templates: all peaks; 2nd-order CPA: top peak of each share). CPA: HW hypothesis.
Phase 0 trains the networks on HW classes: compare them with the `_HW` template rows.

| Attack | Final GE (± std over subsets) | Final median rank | Final SR | Traces to GE < 2 | Seconds |
|---|---|---|---|---|---|
| template_pca_ID | 1.54 ± 1.66 | 1 | 0.78 | 860 | 5.4 |
| template_pca_HW | 1.0 ± 0.0 | 1 | 1.0 | 360 | 4.1 |
| template_poi_ID | 3.41 ± 4.91 | 1 | 0.57 | not reached | 1.9 |
| template_poi_HW | 1.0 ± 0.0 | 1 | 1.0 | 260 | 1.7 |
| cpa_HW | 116.57 ± 68.24 | 109.5 | 0.0 | not reached | 116.0 |
| cpa_known_mask_HW | 1.0 ± 0.0 | 1 | 1.0 | 30 | 96.7 |
| cpa_2nd_order_HW | 3.15 ± 5.6 | 1 | 0.59 | not reached | 15.3 |

## Meaning

- `template_pca_ID`: Whole-trace PCA templates recover the key.
- `template_pca_HW`: Whole-trace PCA templates recover the key.
- `template_poi_ID`: Templates on the share points do not reach GE < 2 in this trace budget.
- `template_poi_HW`: Templates on the share points recover the key: QDA models the share dependence.
- `cpa_HW`: First-order CPA on the unmasked value fails, as expected: masking removes first-order leakage.
- `cpa_known_mask_HW`: With the true mask, first-order CPA recovers the key: the masked share leaks on its own.
- `cpa_2nd_order_HW`: Second-order CPA around the share points does not reach GE < 2 in this trace budget; compare the median rank: a few failed subsets can hold the mean up.
- Best real classic attack: `template_poi_HW`, GE < 2 after 260 traces. A network in Phase 3 should beat this to be worth its cost.

Files: `ge.png`, `share_correlation.png`, `results.csv`, `curves.npz` (keys `<attack>_ge`, `<attack>_sr`, `interval`).
