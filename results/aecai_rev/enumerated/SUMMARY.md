# SUMMARY: main numbers of the revision experiments

Written by `python3 -m aecai_rev.summary` from the tables; values only, no interpretation. Format: window mean [window-level bootstrap 95 % CI]. A difference is `a - b`: mean [window CI]; video CI (17 videos); median; Wilcoxon p, Holm-corrected over the three pairs of one metric. Population: the enumerated windows. Area cut: `frac0.001` of the valid area unless said otherwise. SAM points_per_side 24 unless said otherwise.

## Table 1 reproduced (`00_check/`)

| SAM input | F1@.5 | F1@.75 | n_reg | oracle mIoU |
|---|---|---|---|---|
| rgb | 0.562 | 0.371 | 15.8 | 0.757 |
| depth | 0.480 | 0.291 | 6.7 | 0.469 |
| normal | 0.589 | 0.396 | 9.4 | 0.646 |

Tagged evaluator re-run equals the frozen JSONs on 81 of 81 window x modality pairs; `aecai_rev.score` (300 px) max per-frame difference 0.0e+00.

## 0. Windows (`00_windows/`)

One window per CholecSeg8k clip: 86 kept, 15 excluded (starts before the first frame of the video); 1555 annotated frames in the kept windows. The numbers below are on these enumerated windows.

## 7. Area cut (`07_filter/`)

300 px = 0.00129 of the valid area (median; range 0.00118-0.00193, 1555 frames).

| filter | metric | rgb | depth | normal |
|---|---|---|---|---|
| frac0 | f1@0.5 | 0.496 [0.470, 0.523] | 0.439 [0.404, 0.474] | 0.477 [0.442, 0.512] |
| frac0 | f1@0.75 | 0.343 [0.315, 0.370] | 0.292 [0.263, 0.324] | 0.318 [0.288, 0.348] |
| frac0 | n_gt | 11.281 [10.480, 12.072] | 11.281 [10.480, 12.072] | 11.281 [10.480, 12.072] |
| frac0 | n_pred | 14.854 [13.674, 16.099] | 6.503 [5.967, 7.043] | 8.134 [7.458, 8.812] |
| frac0.001 | f1@0.5 | 0.547 [0.518, 0.575] | 0.481 [0.445, 0.518] | 0.522 [0.484, 0.557] |
| frac0.001 | f1@0.75 | 0.379 [0.349, 0.410] | 0.321 [0.289, 0.353] | 0.348 [0.316, 0.380] |
| frac0.001 | n_gt | 9.767 [9.116, 10.425] | 9.767 [9.116, 10.425] | 9.767 [9.116, 10.425] |
| frac0.001 | n_pred | 13.634 [12.588, 14.767] | 6.160 [5.668, 6.658] | 7.743 [7.103, 8.389] |
| frac0.005 | f1@0.5 | 0.619 [0.591, 0.647] | 0.511 [0.474, 0.549] | 0.562 [0.523, 0.600] |
| frac0.005 | f1@0.75 | 0.433 [0.401, 0.464] | 0.344 [0.312, 0.378] | 0.383 [0.347, 0.418] |
| frac0.005 | n_gt | 8.733 [8.216, 9.244] | 8.733 [8.216, 9.244] | 8.733 [8.216, 9.244] |
| frac0.005 | n_pred | 10.625 [9.879, 11.406] | 5.703 [5.254, 6.147] | 6.954 [6.380, 7.517] |
| px300 | f1@0.5 | 0.557 [0.528, 0.585] | 0.488 [0.452, 0.525] | 0.529 [0.491, 0.566] |
| px300 | f1@0.75 | 0.387 [0.357, 0.417] | 0.326 [0.293, 0.359] | 0.354 [0.321, 0.386] |
| px300 | n_gt | 9.582 [8.962, 10.202] | 9.582 [8.962, 10.202] | 9.582 [8.962, 10.202] |
| px300 | n_pred | 13.332 [12.266, 14.458] | 6.097 [5.604, 6.595] | 7.646 [6.994, 8.283] |

normal - rgb and normal - depth, F1@.5, per cut:

- frac0, normal - rgb: -0.019 [-0.053, +0.016]; video CI [-0.071, 0.027]; median -0.022; p_holm = 0.229
- frac0, normal - depth: +0.038 [+0.004, +0.072]; video CI [-0.003, 0.075]; median +0.017; p_holm = 0.085
- frac0.001, normal - rgb: -0.025 [-0.061, +0.011]; video CI [-0.083, 0.023]; median -0.024; p_holm = 0.188
- frac0.001, normal - depth: +0.040 [+0.004, +0.076]; video CI [-0.002, 0.080]; median +0.020; p_holm = 0.071
- frac0.005, normal - rgb: -0.057 [-0.096, -0.020]; video CI [-0.118, -0.007]; median -0.045; p_holm = 0.009
- frac0.005, normal - depth: +0.051 [+0.013, +0.089]; video CI [0.005, 0.092]; median +0.032; p_holm = 0.010
- px300, normal - rgb: -0.028 [-0.066, +0.010]; video CI [-0.087, 0.022]; median -0.023; p_holm = 0.139
- px300, normal - depth: +0.041 [+0.005, +0.077]; video CI [-0.001, 0.081]; median +0.021; p_holm = 0.078

## 1. Granularity sweep (`01_sweep/`)

Mean GT instances per frame: 9.77. Points marked * are outside the specified grid [8, 12, 16, 24, 32].

| pps | rgb F1@.5 | depth F1@.5 | normal F1@.5 | rgb n_reg | depth n_reg | normal n_reg |
|---|---|---|---|---|---|---|
| 4* | 0.565 [0.536, 0.593] | 0.432 [0.393, 0.472] | 0.481 [0.443, 0.519] | 7.2 [6.7, 7.7] | 4.6 [4.3, 5.0] | 5.5 [5.2, 5.9] |
| 6* | 0.590 [0.562, 0.619] | 0.455 [0.416, 0.494] | 0.513 [0.471, 0.554] | 9.5 [8.9, 10.2] | 5.4 [5.0, 5.8] | 6.5 [6.0, 7.0] |
| 8 | 0.592 [0.564, 0.621] | 0.467 [0.430, 0.503] | 0.515 [0.476, 0.553] | 10.9 [10.1, 11.8] | 6.0 [5.5, 6.4] | 6.9 [6.4, 7.5] |
| 12 | 0.576 [0.550, 0.602] | 0.488 [0.451, 0.526] | 0.527 [0.492, 0.563] | 12.8 [11.8, 14.0] | 6.4 [5.9, 6.9] | 7.7 [7.1, 8.3] |
| 16 | 0.572 [0.547, 0.597] | 0.488 [0.449, 0.526] | 0.525 [0.488, 0.563] | 13.6 [12.6, 14.7] | 6.4 [5.9, 6.9] | 8.0 [7.4, 8.7] |
| 24 | 0.547 [0.518, 0.575] | 0.481 [0.445, 0.518] | 0.522 [0.484, 0.557] | 14.9 [13.7, 16.1] | 6.5 [6.0, 7.0] | 8.1 [7.5, 8.8] |
| 32 | 0.551 [0.524, 0.577] | 0.476 [0.439, 0.512] | 0.522 [0.487, 0.557] | 15.3 [14.1, 16.6] | 6.5 [6.0, 7.0] | 8.5 [7.8, 9.2] |
| 48* | 0.545 [0.518, 0.571] | 0.478 [0.440, 0.516] | 0.519 [0.482, 0.555] | 15.3 [14.1, 16.5] | 6.6 [6.1, 7.2] | 8.3 [7.7, 9.0] |

Matched granularity (basis n_reg):

- specified grid, depth, f1@0.5: nearest pps 24 (n_reg 6.50) -> 0.481; interpolated at 9.77: outside the range of the points
- specified grid, depth, f1@0.75: nearest pps 24 (n_reg 6.50) -> 0.321; interpolated at 9.77: outside the range of the points
- specified grid, normal, f1@0.5: nearest pps 32 (n_reg 8.53) -> 0.522; interpolated at 9.77: outside the range of the points
- specified grid, normal, f1@0.75: nearest pps 32 (n_reg 8.53) -> 0.346; interpolated at 9.77: outside the range of the points
- specified grid, rgb, f1@0.5: nearest pps 8 (n_reg 10.92) -> 0.592; interpolated at 9.77: outside the range of the points
- specified grid, rgb, f1@0.75: nearest pps 8 (n_reg 10.92) -> 0.427; interpolated at 9.77: outside the range of the points
- extended grid, depth, f1@0.5: nearest pps 48 (n_reg 6.65) -> 0.478; interpolated at 9.77: outside the range of the points
- extended grid, depth, f1@0.75: nearest pps 48 (n_reg 6.65) -> 0.321; interpolated at 9.77: outside the range of the points
- extended grid, normal, f1@0.5: nearest pps 32 (n_reg 8.53) -> 0.522; interpolated at 9.77: outside the range of the points
- extended grid, normal, f1@0.75: nearest pps 32 (n_reg 8.53) -> 0.346; interpolated at 9.77: outside the range of the points
- extended grid, rgb, f1@0.5: nearest pps 6 (n_reg 9.53) -> 0.590; interpolated at 9.77: 0.591 [0.563, 0.619]
- extended grid, rgb, f1@0.75: nearest pps 6 (n_reg 9.53) -> 0.432; interpolated at 9.77: 0.431 [0.399, 0.463]

normal - rgb, F1@.5, per pps:

- pps 4: -0.084 [-0.116, -0.052]; video CI [-0.130, -0.047]; median -0.060; p_holm = 9.8e-06
- pps 6: -0.078 [-0.114, -0.042]; video CI [-0.132, -0.036]; median -0.060; p_holm = 8.6e-04
- pps 8: -0.078 [-0.112, -0.043]; video CI [-0.135, -0.034]; median -0.079; p_holm = 6.2e-05
- pps 12: -0.049 [-0.081, -0.016]; video CI [-0.096, -0.011]; median -0.044; p_holm = 0.011
- pps 16: -0.047 [-0.080, -0.014]; video CI [-0.092, -0.012]; median -0.031; p_holm = 0.032
- pps 24: -0.025 [-0.061, +0.011]; video CI [-0.083, 0.023]; median -0.024; p_holm = 0.188
- pps 32: -0.028 [-0.062, +0.006]; video CI [-0.073, 0.010]; median -0.017; p_holm = 0.101
- pps 48: -0.026 [-0.063, +0.009]; video CI [-0.080, 0.024]; median -0.026; p_holm = 0.222

## 2. Statistics

Every `summary.csv` holds `ci_low`, `ci_high` (window bootstrap, 10,000, percentile) and `ci_low_video`, `ci_high_video`; every `tests.csv` holds `median_diff`, `ci_low`, `ci_high` (of the median), `mean_diff` with its intervals, `p_raw` (Wilcoxon, two-sided) and `p_holm`.

Table 1 with intervals (300 px cut, as in the paper):

| filter | metric | rgb | depth | normal |
|---|---|---|---|---|
| px300 | f1@0.5 | 0.557 [0.528, 0.585] | 0.488 [0.452, 0.525] | 0.529 [0.491, 0.566] |
| px300 | f1@0.75 | 0.387 [0.357, 0.417] | 0.326 [0.293, 0.359] | 0.354 [0.321, 0.386] |
| px300 | n_reg | 14.854 [13.674, 16.099] | 6.503 [5.967, 7.043] | 8.134 [7.458, 8.812] |
| px300 | miou | 0.733 [0.705, 0.760] | 0.493 [0.458, 0.529] | 0.559 [0.522, 0.595] |

- f1@0.5, normal - rgb: -0.028 [-0.066, +0.010]; video CI [-0.087, 0.022]; median -0.023; p_holm = 0.139
- f1@0.5, normal - depth: +0.041 [+0.005, +0.077]; video CI [-0.001, 0.081]; median +0.021; p_holm = 0.078
- f1@0.5, rgb - depth: +0.069 [+0.031, +0.108]; video CI [0.024, 0.117]; median +0.085; p_holm = 0.002
- f1@0.75, normal - rgb: -0.033 [-0.069, +0.002]; video CI [-0.100, 0.017]; median -0.020; p_holm = 0.160
- f1@0.75, normal - depth: +0.028 [-0.002, +0.059]; video CI [-0.005, 0.059]; median +0.024; p_holm = 0.160
- f1@0.75, rgb - depth: +0.061 [+0.022, +0.100]; video CI [0.007, 0.124]; median +0.079; p_holm = 0.003
- miou, normal - rgb: -0.173 [-0.208, -0.138]; video CI [-0.226, -0.132]; median -0.173; p_holm = 1.0e-11
- miou, normal - depth: +0.066 [+0.031, +0.101]; video CI [0.020, 0.102]; median +0.047; p_holm = 4.3e-04
- miou, rgb - depth: +0.240 [+0.206, +0.274]; video CI [0.207, 0.277]; median +0.251; p_holm = 3.8e-14

## 3. Merge per GT instance (`03_merge/`)

| variant | metric | rgb | depth | normal |
|---|---|---|---|---|
| merged | f1@0.5 | 0.791 [0.766, 0.816] | 0.545 [0.505, 0.585] | 0.623 [0.580, 0.664] |
| merged | f1@0.75 | 0.647 [0.611, 0.682] | 0.382 [0.343, 0.422] | 0.456 [0.415, 0.499] |
| merged | miou | 0.732 [0.704, 0.760] | 0.493 [0.458, 0.529] | 0.559 [0.522, 0.596] |
| merged | n_reg | 7.743 [7.239, 8.258] | 5.086 [4.700, 5.464] | 5.878 [5.427, 6.337] |
| raw | f1@0.5 | 0.547 [0.518, 0.575] | 0.481 [0.445, 0.518] | 0.522 [0.484, 0.557] |
| raw | f1@0.75 | 0.379 [0.349, 0.410] | 0.321 [0.289, 0.353] | 0.348 [0.316, 0.380] |
| raw | miou | 0.733 [0.705, 0.760] | 0.493 [0.458, 0.529] | 0.559 [0.522, 0.595] |
| raw | n_reg | 14.854 [13.674, 16.099] | 6.503 [5.967, 7.043] | 8.134 [7.458, 8.812] |

merged - raw, per modality:

- f1@0.5, rgb: +0.244 [+0.219, +0.271]; video CI [0.195, 0.297]; median +0.221; p_holm = 8.0e-16
- f1@0.5, depth: +0.064 [+0.049, +0.079]; video CI [0.046, 0.090]; median +0.046; p_holm = 1.7e-13
- f1@0.5, normal: +0.102 [+0.086, +0.118]; video CI [0.084, 0.123]; median +0.088; p_holm = 7.8e-15
- f1@0.75, rgb: +0.268 [+0.239, +0.299]; video CI [0.211, 0.325]; median +0.255; p_holm = 8.0e-16
- f1@0.75, depth: +0.062 [+0.044, +0.081]; video CI [0.039, 0.094]; median +0.026; p_holm = 2.6e-13
- f1@0.75, normal: +0.108 [+0.086, +0.131]; video CI [0.080, 0.139]; median +0.066; p_holm = 7.8e-15
- miou, rgb: -0.001 [-0.002, -0.000]; video CI [-0.001, -0.000]; median +0.000; p_holm = 6.3e-05
- miou, depth: -0.000 [-0.001, -0.000]; video CI [-0.000, 0.000]; median +0.000; p_holm = 0.003
- miou, normal: -0.001 [-0.001, -0.000]; video CI [-0.001, -0.000]; median +0.000; p_holm = 1.0e-04

Between modalities after merging:

- f1@0.5, normal - rgb: -0.167 [-0.206, -0.126]; video CI [-0.229, -0.119]; median -0.154; p_holm = 9.9e-11
- f1@0.5, normal - depth: +0.079 [+0.037, +0.119]; video CI [0.028, 0.125]; median +0.065; p_holm = 4.3e-04
- f1@0.5, rgb - depth: +0.246 [+0.206, +0.284]; video CI [0.196, 0.297]; median +0.261; p_holm = 3.5e-13
- f1@0.75, normal - rgb: -0.191 [-0.230, -0.150]; video CI [-0.262, -0.134]; median -0.187; p_holm = 1.3e-11
- f1@0.75, normal - depth: +0.074 [+0.033, +0.114]; video CI [0.027, 0.115]; median +0.063; p_holm = 5.0e-04
- f1@0.75, rgb - depth: +0.265 [+0.225, +0.304]; video CI [0.210, 0.324]; median +0.279; p_holm = 1.9e-13
- miou, normal - rgb: -0.173 [-0.208, -0.139]; video CI [-0.225, -0.132]; median -0.173; p_holm = 1.0e-11
- miou, normal - depth: +0.066 [+0.030, +0.101]; video CI [0.020, 0.102]; median +0.045; p_holm = 4.9e-04
- miou, rgb - depth: +0.239 [+0.206, +0.272]; video CI [0.207, 0.275]; median +0.251; p_holm = 3.7e-14

## 4. Identity, all windows, GT linking `consecutive_annotated` (`04_identity/`)

| subset | metric | rgb | depth | normal |
|---|---|---|---|---|
| all | id_switch_rate | 0.057 [0.048, 0.067] | 0.059 [0.048, 0.072] | 0.061 [0.041, 0.089] |
| all | class_switch_rate | 0.005 [0.003, 0.008] | 0.007 [0.004, 0.011] | 0.007 [0.004, 0.012] |
| all | fragmentation_mean | 1.090 [1.072, 1.107] | 1.048 [1.032, 1.065] | 1.060 [1.046, 1.076] |
| all | fragmentation_ge2 | 0.086 [0.070, 0.103] | 0.048 [0.032, 0.066] | 0.060 [0.045, 0.076] |
| all | survival_forward | 0.867 [0.841, 0.891] | 0.906 [0.878, 0.931] | 0.907 [0.882, 0.929] |
| all | survival_backward | 0.829 [0.802, 0.854] | 0.880 [0.853, 0.905] | 0.888 [0.867, 0.907] |
| all | reappearance_rate | 0.382 [0.345, 0.420] | 0.255 [0.210, 0.303] | 0.255 [0.209, 0.301] |
| all | reappear_same_gt | 0.170 [0.083, 0.269] | 0.214 [0.071, 0.375] | 0.242 [0.091, 0.417] |
| all | reappear_same_class | 0.958 [0.902, 1.000] | 0.964 [0.893, 1.000] | 0.932 [0.818, 1.000] |

- id_switch_rate, normal - rgb: +0.004 [-0.016, +0.031]; video CI [-0.014, 0.034]; median -0.008; p_holm = 0.074
- id_switch_rate, normal - depth: +0.001 [-0.019, +0.030]; video CI [-0.020, 0.033]; median -0.002; p_holm = 0.074
- id_switch_rate, rgb - depth: -0.002 [-0.013, +0.009]; video CI [-0.017, 0.013]; median +0.000; p_holm = 0.631
- class_switch_rate, normal - rgb: +0.002 [-0.002, +0.007]; video CI [-0.002, 0.009]; median +0.000; p_holm = 1.000
- class_switch_rate, normal - depth: +0.000 [-0.004, +0.005]; video CI [-0.004, 0.006]; median +0.000; p_holm = 1.000
- class_switch_rate, rgb - depth: -0.002 [-0.005, +0.000]; video CI [-0.005, 0.001]; median +0.000; p_holm = 0.482
- fragmentation_mean, normal - rgb: -0.029 [-0.051, -0.008]; video CI [-0.051, -0.009]; median +0.000; p_holm = 0.020
- fragmentation_mean, normal - depth: +0.012 [-0.007, +0.030]; video CI [-0.011, 0.033]; median +0.000; p_holm = 0.124
- fragmentation_mean, rgb - depth: +0.042 [+0.021, +0.063]; video CI [0.025, 0.059]; median +0.015; p_holm = 1.8e-04
- fragmentation_ge2, normal - rgb: -0.027 [-0.047, -0.006]; video CI [-0.048, -0.005]; median +0.000; p_holm = 0.027
- fragmentation_ge2, normal - depth: +0.011 [-0.007, +0.029]; video CI [-0.012, 0.032]; median +0.000; p_holm = 0.141
- fragmentation_ge2, rgb - depth: +0.038 [+0.018, +0.059]; video CI [0.020, 0.056]; median +0.015; p_holm = 4.2e-04
- survival_forward, normal - rgb: +0.039 [+0.020, +0.060]; video CI [0.023, 0.052]; median +0.025; p_holm = 0.003
- survival_forward, normal - depth: +0.000 [-0.019, +0.024]; video CI [-0.016, 0.012]; median +0.000; p_holm = 0.327
- survival_forward, rgb - depth: -0.039 [-0.065, -0.013]; video CI [-0.056, -0.023]; median -0.031; p_holm = 0.002
- survival_backward, normal - rgb: +0.058 [+0.034, +0.084]; video CI [0.027, 0.089]; median +0.033; p_holm = 6.7e-05
- survival_backward, normal - depth: +0.008 [-0.013, +0.030]; video CI [-0.009, 0.023]; median +0.000; p_holm = 0.960
- survival_backward, rgb - depth: -0.051 [-0.078, -0.024]; video CI [-0.079, -0.025]; median -0.049; p_holm = 5.3e-04
- reappearance_rate, normal - rgb: -0.127 [-0.166, -0.087]; video CI [-0.164, -0.081]; median -0.085; p_holm = 1.5e-07
- reappearance_rate, normal - depth: -0.000 [-0.045, +0.045]; video CI [-0.036, 0.038]; median +0.000; p_holm = 0.991
- reappearance_rate, rgb - depth: +0.126 [+0.089, +0.165]; video CI [0.090, 0.161]; median +0.111; p_holm = 1.0e-07
- reappear_same_gt, normal - rgb: +0.009 [-0.102, +0.111]; video CI [-0.070, 0.111]; median +0.000; p_holm = 1.000
- reappear_same_gt, normal - depth: -0.031 [-0.219, +0.125]; video CI [-0.214, 0.125]; median +0.000; p_holm = 1.000
- reappear_same_gt, rgb - depth: -0.071 [-0.205, +0.045]; video CI [-0.178, 0.053]; median +0.000; p_holm = 1.000
- reappear_same_class, normal - rgb: -0.009 [-0.083, +0.056]; video CI [-0.079, 0.053]; median +0.000; p_holm = 0.655
- reappear_same_class, normal - depth: -0.094 [-0.250, +0.000]; video CI [-0.250, 0.000]; median +0.000; p_holm = 0.539
- reappear_same_class, rgb - depth: -0.013 [-0.038, +0.000]; video CI [-0.045, 0.000]; median +0.000; p_holm = 0.635

## 4. Identity, frames_in_order windows, GT linking `consecutive_annotated` (`04_identity/`)

| subset | metric | rgb | depth | normal |
|---|---|---|---|---|
| frames_in_order | id_switch_rate | 0.057 [0.048, 0.067] | 0.059 [0.048, 0.072] | 0.061 [0.041, 0.089] |
| frames_in_order | class_switch_rate | 0.005 [0.003, 0.008] | 0.007 [0.004, 0.011] | 0.007 [0.004, 0.012] |
| frames_in_order | fragmentation_mean | 1.090 [1.072, 1.107] | 1.048 [1.032, 1.065] | 1.060 [1.046, 1.076] |
| frames_in_order | fragmentation_ge2 | 0.086 [0.070, 0.103] | 0.048 [0.032, 0.066] | 0.060 [0.045, 0.076] |
| frames_in_order | survival_forward | 0.867 [0.841, 0.891] | 0.906 [0.878, 0.931] | 0.907 [0.882, 0.929] |
| frames_in_order | survival_backward | 0.829 [0.802, 0.854] | 0.880 [0.853, 0.905] | 0.888 [0.867, 0.907] |
| frames_in_order | reappearance_rate | 0.382 [0.345, 0.420] | 0.255 [0.210, 0.303] | 0.255 [0.209, 0.301] |
| frames_in_order | reappear_same_gt | 0.170 [0.083, 0.269] | 0.214 [0.071, 0.375] | 0.242 [0.091, 0.417] |
| frames_in_order | reappear_same_class | 0.958 [0.902, 1.000] | 0.964 [0.893, 1.000] | 0.932 [0.818, 1.000] |

- id_switch_rate, normal - rgb: +0.004 [-0.016, +0.031]; video CI [-0.014, 0.034]; median -0.008; p_holm = 0.074
- id_switch_rate, normal - depth: +0.001 [-0.019, +0.030]; video CI [-0.020, 0.033]; median -0.002; p_holm = 0.074
- id_switch_rate, rgb - depth: -0.002 [-0.013, +0.009]; video CI [-0.017, 0.013]; median +0.000; p_holm = 0.631
- class_switch_rate, normal - rgb: +0.002 [-0.002, +0.007]; video CI [-0.002, 0.009]; median +0.000; p_holm = 1.000
- class_switch_rate, normal - depth: +0.000 [-0.004, +0.005]; video CI [-0.004, 0.006]; median +0.000; p_holm = 1.000
- class_switch_rate, rgb - depth: -0.002 [-0.005, +0.000]; video CI [-0.005, 0.001]; median +0.000; p_holm = 0.482
- fragmentation_mean, normal - rgb: -0.029 [-0.051, -0.008]; video CI [-0.051, -0.009]; median +0.000; p_holm = 0.020
- fragmentation_mean, normal - depth: +0.012 [-0.007, +0.030]; video CI [-0.011, 0.033]; median +0.000; p_holm = 0.124
- fragmentation_mean, rgb - depth: +0.042 [+0.021, +0.063]; video CI [0.025, 0.059]; median +0.015; p_holm = 1.8e-04
- fragmentation_ge2, normal - rgb: -0.027 [-0.047, -0.006]; video CI [-0.048, -0.005]; median +0.000; p_holm = 0.027
- fragmentation_ge2, normal - depth: +0.011 [-0.007, +0.029]; video CI [-0.012, 0.032]; median +0.000; p_holm = 0.141
- fragmentation_ge2, rgb - depth: +0.038 [+0.018, +0.059]; video CI [0.020, 0.056]; median +0.015; p_holm = 4.2e-04
- survival_forward, normal - rgb: +0.039 [+0.020, +0.060]; video CI [0.023, 0.052]; median +0.025; p_holm = 0.003
- survival_forward, normal - depth: +0.000 [-0.019, +0.024]; video CI [-0.016, 0.012]; median +0.000; p_holm = 0.327
- survival_forward, rgb - depth: -0.039 [-0.065, -0.013]; video CI [-0.056, -0.023]; median -0.031; p_holm = 0.002
- survival_backward, normal - rgb: +0.058 [+0.034, +0.084]; video CI [0.027, 0.089]; median +0.033; p_holm = 6.7e-05
- survival_backward, normal - depth: +0.008 [-0.013, +0.030]; video CI [-0.009, 0.023]; median +0.000; p_holm = 0.960
- survival_backward, rgb - depth: -0.051 [-0.078, -0.024]; video CI [-0.079, -0.025]; median -0.049; p_holm = 5.3e-04
- reappearance_rate, normal - rgb: -0.127 [-0.166, -0.087]; video CI [-0.164, -0.081]; median -0.085; p_holm = 1.5e-07
- reappearance_rate, normal - depth: -0.000 [-0.045, +0.045]; video CI [-0.036, 0.038]; median +0.000; p_holm = 0.991
- reappearance_rate, rgb - depth: +0.126 [+0.089, +0.165]; video CI [0.090, 0.161]; median +0.111; p_holm = 1.0e-07
- reappear_same_gt, normal - rgb: +0.009 [-0.102, +0.111]; video CI [-0.070, 0.111]; median +0.000; p_holm = 1.000
- reappear_same_gt, normal - depth: -0.031 [-0.219, +0.125]; video CI [-0.214, 0.125]; median +0.000; p_holm = 1.000
- reappear_same_gt, rgb - depth: -0.071 [-0.205, +0.045]; video CI [-0.178, 0.053]; median +0.000; p_holm = 1.000
- reappear_same_class, normal - rgb: -0.009 [-0.083, +0.056]; video CI [-0.079, 0.053]; median +0.000; p_holm = 0.655
- reappear_same_class, normal - depth: -0.094 [-0.250, +0.000]; video CI [-0.250, 0.000]; median +0.000; p_holm = 0.539
- reappear_same_class, rgb - depth: -0.013 [-0.038, +0.000]; video CI [-0.045, 0.000]; median +0.000; p_holm = 0.635

The same with GT linking only between adjacent samples is in `04_identity/summary.csv` (`link == adjacent_samples`).

## 5. Instrument-tissue merges (`05_merge_rate/`)

| level | metric | rgb | depth | normal |
|---|---|---|---|---|
| all | merge_region_rate | 0.000 [0.000, 0.000] | 0.025 [0.012, 0.040] | 0.027 [0.013, 0.043] |
| all | merge_frame | 0.002 [0.001, 0.005] | 0.115 [0.059, 0.178] | 0.145 [0.087, 0.209] |

- merge_region_rate, normal - rgb: +0.026 [+0.014, +0.044]; video CI [0.013, 0.044]; median +0.000; p_holm = 2.8e-05
- merge_region_rate, normal - depth: +0.001 [-0.015, +0.019]; video CI [-0.016, 0.022]; median +0.000; p_holm = 0.993
- merge_region_rate, rgb - depth: -0.025 [-0.039, -0.012]; video CI [-0.039, -0.011]; median +0.000; p_holm = 1.6e-04
- merge_frame, normal - rgb: +0.143 [+0.085, +0.206]; video CI [0.078, 0.207]; median +0.000; p_holm = 4.1e-05
- merge_frame, normal - depth: +0.030 [-0.034, +0.093]; video CI [-0.046, 0.114]; median +0.000; p_holm = 0.285
- merge_frame, rgb - depth: -0.113 [-0.178, -0.056]; video CI [-0.190, -0.039]; median +0.000; p_holm = 2.9e-04

## 6. Nameability (`06_naming/`)

| level | metric | rgb | depth | normal |
|---|---|---|---|---|
| all | named_one | 0.892 [0.876, 0.908] | 0.746 [0.708, 0.780] | 0.787 [0.750, 0.818] |
| all | named_many | 0.031 [0.021, 0.041] | 0.179 [0.145, 0.218] | 0.126 [0.094, 0.163] |
| all | named_none | 0.077 [0.066, 0.089] | 0.075 [0.062, 0.089] | 0.087 [0.073, 0.103] |

- named_one, normal - rgb: -0.106 [-0.142, -0.072]; video CI [-0.146, -0.068]; median -0.087; p_holm = 3.1e-09
- named_one, normal - depth: +0.041 [+0.002, +0.081]; video CI [-0.017, 0.101]; median +0.019; p_holm = 0.052
- named_one, rgb - depth: +0.147 [+0.109, +0.188]; video CI [0.095, 0.212]; median +0.125; p_holm = 4.3e-11
- named_many, normal - rgb: +0.095 [+0.062, +0.134]; video CI [0.060, 0.136]; median +0.058; p_holm = 6.1e-09
- named_many, normal - depth: -0.053 [-0.090, -0.017]; video CI [-0.101, -0.005]; median -0.043; p_holm = 5.5e-04
- named_many, rgb - depth: -0.149 [-0.187, -0.113]; video CI [-0.208, -0.107]; median -0.123; p_holm = 1.0e-12
- named_none, normal - rgb: +0.011 [-0.007, +0.028]; video CI [-0.011, 0.034]; median +0.005; p_holm = 0.432
- named_none, normal - depth: +0.012 [-0.006, +0.031]; video CI [-0.006, 0.031]; median +0.000; p_holm = 0.576
- named_none, rgb - depth: +0.002 [-0.016, +0.020]; video CI [-0.019, 0.024]; median +0.005; p_holm = 0.841

Named events (`06_naming/fig6_named_events.txt`, on an AE-CAI window; not Fig. 6's window, whose graph is not on this machine, see `00_inventory.md`):

```
# window VID12_s15_19900_crop, focus node 6, partner node 9
t=4: L-hook electrocautery (node 6) re-enters the view
t=20: L-hook electrocautery (node 6) moves from right of to above the gallbladder (node 9)
t=26: L-hook electrocautery (node 6) moves from right of to above the gallbladder (node 9)
t=28: L-hook electrocautery (node 6) moves from right of to above the gallbladder (node 9)
```
