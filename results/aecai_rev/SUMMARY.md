# SUMMARY: main numbers of the revision experiments

Written by `python3 -m aecai_rev.summary` from the tables; values only, no interpretation. Format: window mean [window-level bootstrap 95 % CI]. A difference is `a - b`: mean [window CI]; video CI (17 videos); median; Wilcoxon p, Holm-corrected over the three pairs of one metric. Population: the legacy27 windows. Area cut: `frac0.001` of the valid area unless said otherwise. SAM points_per_side 24 unless said otherwise.

## Table 1 reproduced (`00_check/`)

| SAM input | F1@.5 | F1@.75 | n_reg | oracle mIoU |
|---|---|---|---|---|
| rgb | 0.562 | 0.371 | 15.8 | 0.757 |
| depth | 0.480 | 0.291 | 6.7 | 0.469 |
| normal | 0.589 | 0.396 | 9.4 | 0.646 |

Tagged evaluator re-run equals the frozen JSONs on 81 of 81 window x modality pairs; `aecai_rev.score` (300 px) max per-frame difference 0.0e+00.

## 0. Windows (`00_windows/`)

One window per CholecSeg8k clip: 86 kept, 15 excluded (starts before the first frame of the video); 1555 annotated frames in the kept windows. Not extracted or tracked yet: the numbers below are on the AE-CAI windows.

## 7. Area cut (`07_filter/`)

300 px = 0.00125 of the valid area (median; range 0.00118-0.00177, 522 frames).

| filter | metric | rgb | depth | normal |
|---|---|---|---|---|
| frac0 | f1@0.5 | 0.496 [0.453, 0.538] | 0.433 [0.377, 0.488] | 0.530 [0.473, 0.586] |
| frac0 | f1@0.75 | 0.327 [0.277, 0.378] | 0.262 [0.220, 0.305] | 0.355 [0.308, 0.402] |
| frac0 | n_gt | 11.347 [10.030, 12.735] | 11.347 [10.030, 12.735] | 11.347 [10.030, 12.735] |
| frac0 | n_pred | 15.771 [13.875, 17.729] | 6.656 [5.886, 7.394] | 9.418 [8.460, 10.427] |
| frac0.001 | f1@0.5 | 0.552 [0.506, 0.598] | 0.472 [0.413, 0.532] | 0.580 [0.519, 0.638] |
| frac0.001 | f1@0.75 | 0.364 [0.310, 0.419] | 0.286 [0.239, 0.333] | 0.389 [0.337, 0.441] |
| frac0.001 | n_gt | 9.933 [8.774, 11.095] | 9.933 [8.774, 11.095] | 9.933 [8.774, 11.095] |
| frac0.001 | n_pred | 14.236 [12.550, 15.958] | 6.357 [5.647, 7.036] | 8.873 [7.958, 9.827] |
| frac0.005 | f1@0.5 | 0.621 [0.573, 0.668] | 0.506 [0.443, 0.571] | 0.627 [0.564, 0.686] |
| frac0.005 | f1@0.75 | 0.415 [0.354, 0.476] | 0.309 [0.259, 0.363] | 0.430 [0.373, 0.485] |
| frac0.005 | n_gt | 8.885 [7.981, 9.809] | 8.885 [7.981, 9.809] | 8.885 [7.981, 9.809] |
| frac0.005 | n_pred | 11.261 [10.121, 12.422] | 5.801 [5.197, 6.362] | 7.939 [7.197, 8.693] |
| px300 | f1@0.5 | 0.562 [0.516, 0.609] | 0.480 [0.419, 0.542] | 0.589 [0.525, 0.648] |
| px300 | f1@0.75 | 0.371 [0.318, 0.429] | 0.291 [0.244, 0.342] | 0.396 [0.342, 0.450] |
| px300 | n_gt | 9.729 [8.628, 10.860] | 9.729 [8.628, 10.860] | 9.729 [8.628, 10.860] |
| px300 | n_pred | 13.875 [12.237, 15.550] | 6.294 [5.596, 6.964] | 8.784 [7.895, 9.722] |

normal - rgb and normal - depth, F1@.5, per cut:

- frac0, normal - rgb: +0.035 [-0.011, +0.081]; video CI [-0.010, 0.079]; median +0.024; p_holm = 0.170
- frac0, normal - depth: +0.097 [+0.057, +0.139]; video CI [0.055, 0.136]; median +0.077; p_holm = 3.3e-04
- frac0.001, normal - rgb: +0.029 [-0.021, +0.077]; video CI [-0.019, 0.076]; median +0.003; p_holm = 0.290
- frac0.001, normal - depth: +0.109 [+0.065, +0.154]; video CI [0.062, 0.154]; median +0.089; p_holm = 2.5e-04
- frac0.005, normal - rgb: +0.005 [-0.045, +0.054]; video CI [-0.040, 0.049]; median -0.002; p_holm = 0.786
- frac0.005, normal - depth: +0.121 [+0.075, +0.169]; video CI [0.073, 0.166]; median +0.123; p_holm = 7.8e-05
- px300, normal - rgb: +0.027 [-0.022, +0.075]; video CI [-0.021, 0.073]; median +0.007; p_holm = 0.324
- px300, normal - depth: +0.109 [+0.065, +0.157]; video CI [0.062, 0.155]; median +0.092; p_holm = 2.5e-04

## 1. Granularity sweep (`01_sweep/`)

Mean GT instances per frame: 9.93. Points marked * are outside the specified grid [8, 12, 16, 24, 32].

| pps | rgb F1@.5 | depth F1@.5 | normal F1@.5 | rgb n_reg | depth n_reg | normal n_reg |
|---|---|---|---|---|---|---|
| 4* | 0.561 [0.517, 0.606] | 0.438 [0.370, 0.508] | 0.553 [0.502, 0.605] | 7.3 [6.5, 8.1] | 4.7 [4.1, 5.2] | 6.7 [6.2, 7.3] |
| 6* | 0.587 [0.548, 0.625] | 0.465 [0.397, 0.534] | 0.587 [0.527, 0.648] | 10.1 [8.9, 11.4] | 5.4 [4.7, 6.0] | 7.7 [7.0, 8.3] |
| 8 | 0.575 [0.530, 0.620] | 0.473 [0.406, 0.541] | 0.576 [0.510, 0.638] | 12.2 [10.8, 13.6] | 5.8 [5.1, 6.4] | 8.1 [7.4, 8.9] |
| 12 | 0.579 [0.539, 0.618] | 0.485 [0.420, 0.552] | 0.578 [0.513, 0.640] | 14.2 [12.5, 15.8] | 6.1 [5.4, 6.8] | 8.8 [8.0, 9.7] |
| 16 | 0.552 [0.514, 0.591] | 0.471 [0.412, 0.532] | 0.575 [0.514, 0.635] | 15.4 [13.5, 17.3] | 6.5 [5.8, 7.2] | 9.8 [8.9, 10.7] |
| 24 | 0.552 [0.506, 0.598] | 0.472 [0.413, 0.532] | 0.580 [0.519, 0.638] | 15.8 [13.9, 17.7] | 6.7 [5.9, 7.4] | 9.4 [8.5, 10.4] |
| 32 | 0.548 [0.501, 0.594] | 0.458 [0.402, 0.517] | 0.558 [0.491, 0.625] | 15.9 [13.9, 17.9] | 6.4 [5.6, 7.2] | 9.6 [8.7, 10.5] |
| 48* | 0.551 [0.509, 0.593] | 0.466 [0.403, 0.530] | 0.545 [0.483, 0.604] | 16.3 [14.2, 18.7] | 6.7 [5.8, 7.5] | 9.5 [8.6, 10.5] |

Matched granularity (basis n_reg):

- specified grid, depth, f1@0.5: nearest pps 24 (n_reg 6.66) -> 0.472; interpolated at 9.93: outside the range of the points
- specified grid, depth, f1@0.75: nearest pps 24 (n_reg 6.66) -> 0.286; interpolated at 9.93: outside the range of the points
- specified grid, normal, f1@0.5: nearest pps 16 (n_reg 9.81) -> 0.575; interpolated at 9.93: outside the range of the points
- specified grid, normal, f1@0.75: nearest pps 16 (n_reg 9.81) -> 0.383; interpolated at 9.93: outside the range of the points
- specified grid, rgb, f1@0.5: nearest pps 8 (n_reg 12.20) -> 0.575; interpolated at 9.93: outside the range of the points
- specified grid, rgb, f1@0.75: nearest pps 8 (n_reg 12.20) -> 0.382; interpolated at 9.93: outside the range of the points
- extended grid, depth, f1@0.5: nearest pps 48 (n_reg 6.66) -> 0.466; interpolated at 9.93: outside the range of the points
- extended grid, depth, f1@0.75: nearest pps 48 (n_reg 6.66) -> 0.287; interpolated at 9.93: outside the range of the points
- extended grid, normal, f1@0.5: nearest pps 16 (n_reg 9.81) -> 0.575; interpolated at 9.93: outside the range of the points
- extended grid, normal, f1@0.75: nearest pps 16 (n_reg 9.81) -> 0.383; interpolated at 9.93: outside the range of the points
- extended grid, rgb, f1@0.5: nearest pps 6 (n_reg 10.10) -> 0.587; interpolated at 9.93: 0.586 [0.546, 0.624]
- extended grid, rgb, f1@0.75: nearest pps 6 (n_reg 10.10) -> 0.404; interpolated at 9.93: 0.403 [0.357, 0.450]

normal - rgb, F1@.5, per pps:

- pps 4: -0.008 [-0.034, +0.020]; video CI [-0.035, 0.019]; median -0.014; p_holm = 0.427
- pps 6: -0.000 [-0.043, +0.047]; video CI [-0.046, 0.040]; median -0.029; p_holm = 0.841
- pps 8: +0.001 [-0.050, +0.052]; video CI [-0.047, 0.048]; median +0.018; p_holm = 0.934
- pps 12: -0.001 [-0.054, +0.051]; video CI [-0.048, 0.043]; median +0.004; p_holm = 0.897
- pps 16: +0.023 [-0.026, +0.069]; video CI [-0.019, 0.058]; median +0.032; p_holm = 0.301
- pps 24: +0.029 [-0.021, +0.077]; video CI [-0.019, 0.076]; median +0.003; p_holm = 0.290
- pps 32: +0.010 [-0.041, +0.061]; video CI [-0.039, 0.057]; median +0.029; p_holm = 0.645
- pps 48: -0.007 [-0.054, +0.039]; video CI [-0.052, 0.036]; median +0.011; p_holm = 0.972

## 2. Statistics

Every `summary.csv` holds `ci_low`, `ci_high` (window bootstrap, 10,000, percentile) and `ci_low_video`, `ci_high_video`; every `tests.csv` holds `median_diff`, `ci_low`, `ci_high` (of the median), `mean_diff` with its intervals, `p_raw` (Wilcoxon, two-sided) and `p_holm`.

Table 1 with intervals (300 px cut, as in the paper):

| filter | metric | rgb | depth | normal |
|---|---|---|---|---|
| px300 | f1@0.5 | 0.562 [0.516, 0.609] | 0.480 [0.419, 0.542] | 0.589 [0.525, 0.648] |
| px300 | f1@0.75 | 0.371 [0.318, 0.429] | 0.291 [0.244, 0.342] | 0.396 [0.342, 0.450] |
| px300 | n_reg | 15.771 [13.875, 17.729] | 6.656 [5.886, 7.394] | 9.418 [8.460, 10.427] |
| px300 | miou | 0.757 [0.709, 0.803] | 0.469 [0.415, 0.523] | 0.646 [0.590, 0.701] |

- f1@0.5, normal - rgb: +0.027 [-0.022, +0.075]; video CI [-0.021, 0.073]; median +0.007; p_holm = 0.324
- f1@0.5, normal - depth: +0.109 [+0.065, +0.157]; video CI [0.062, 0.155]; median +0.092; p_holm = 2.5e-04
- f1@0.5, rgb - depth: +0.083 [+0.027, +0.139]; video CI [0.024, 0.130]; median +0.107; p_holm = 0.016
- f1@0.75, normal - rgb: +0.025 [-0.023, +0.070]; video CI [-0.030, 0.077]; median +0.044; p_holm = 0.170
- f1@0.75, normal - depth: +0.105 [+0.059, +0.154]; video CI [0.054, 0.156]; median +0.073; p_holm = 3.3e-04
- f1@0.75, rgb - depth: +0.080 [+0.020, +0.141]; video CI [0.014, 0.148]; median +0.059; p_holm = 0.060
- miou, normal - rgb: -0.111 [-0.166, -0.054]; video CI [-0.179, -0.053]; median -0.135; p_holm = 0.001
- miou, normal - depth: +0.176 [+0.123, +0.230]; video CI [0.115, 0.234]; median +0.193; p_holm = 9.1e-06
- miou, rgb - depth: +0.288 [+0.229, +0.346]; video CI [0.236, 0.338]; median +0.287; p_holm = 8.9e-08

## 3. Merge per GT instance (`03_merge/`)

| variant | metric | rgb | depth | normal |
|---|---|---|---|---|
| merged | f1@0.5 | 0.804 [0.769, 0.838] | 0.533 [0.465, 0.600] | 0.714 [0.653, 0.772] |
| merged | f1@0.75 | 0.647 [0.587, 0.705] | 0.362 [0.311, 0.415] | 0.538 [0.473, 0.602] |
| merged | miou | 0.756 [0.709, 0.801] | 0.469 [0.414, 0.525] | 0.645 [0.589, 0.700] |
| merged | n_reg | 8.047 [7.093, 9.010] | 5.237 [4.663, 5.784] | 6.572 [5.803, 7.300] |
| raw | f1@0.5 | 0.552 [0.506, 0.598] | 0.472 [0.413, 0.532] | 0.580 [0.519, 0.638] |
| raw | f1@0.75 | 0.364 [0.310, 0.419] | 0.286 [0.239, 0.333] | 0.389 [0.337, 0.441] |
| raw | miou | 0.757 [0.709, 0.803] | 0.469 [0.415, 0.523] | 0.646 [0.590, 0.701] |
| raw | n_reg | 15.771 [13.875, 17.729] | 6.656 [5.886, 7.394] | 9.418 [8.460, 10.427] |

merged - raw, per modality:

- f1@0.5, rgb: +0.253 [+0.214, +0.293]; video CI [0.207, 0.300]; median +0.257; p_holm = 1.5e-08
- f1@0.5, depth: +0.061 [+0.042, +0.083]; video CI [0.044, 0.075]; median +0.047; p_holm = 1.2e-05
- f1@0.5, normal: +0.133 [+0.100, +0.170]; video CI [0.096, 0.172]; median +0.115; p_holm = 1.5e-08
- f1@0.75, rgb: +0.283 [+0.233, +0.338]; video CI [0.224, 0.347]; median +0.274; p_holm = 1.5e-08
- f1@0.75, depth: +0.076 [+0.049, +0.105]; video CI [0.054, 0.096]; median +0.064; p_holm = 1.2e-05
- f1@0.75, normal: +0.149 [+0.106, +0.197]; video CI [0.104, 0.197]; median +0.115; p_holm = 1.5e-08
- miou, rgb: -0.001 [-0.002, -0.000]; video CI [-0.002, -0.000]; median +0.000; p_holm = 0.001
- miou, depth: -0.001 [-0.001, -0.000]; video CI [-0.001, -0.000]; median +0.000; p_holm = 0.012
- miou, normal: -0.001 [-0.002, -0.000]; video CI [-0.002, -0.000]; median +0.000; p_holm = 0.010

Between modalities after merging:

- f1@0.5, normal - rgb: -0.091 [-0.155, -0.029]; video CI [-0.175, -0.026]; median -0.049; p_holm = 0.005
- f1@0.5, normal - depth: +0.181 [+0.126, +0.236]; video CI [0.122, 0.235]; median +0.200; p_holm = 3.3e-06
- f1@0.5, rgb - depth: +0.271 [+0.201, +0.341]; video CI [0.203, 0.339]; median +0.281; p_holm = 4.5e-07
- f1@0.75, normal - rgb: -0.109 [-0.174, -0.042]; video CI [-0.199, -0.025]; median -0.102; p_holm = 0.002
- f1@0.75, normal - depth: +0.176 [+0.126, +0.228]; video CI [0.121, 0.228]; median +0.183; p_holm = 2.6e-06
- f1@0.75, rgb - depth: +0.285 [+0.218, +0.351]; video CI [0.208, 0.361]; median +0.316; p_holm = 1.5e-06
- miou, normal - rgb: -0.111 [-0.167, -0.052]; video CI [-0.179, -0.053]; median -0.135; p_holm = 0.001
- miou, normal - depth: +0.176 [+0.121, +0.229]; video CI [0.113, 0.234]; median +0.193; p_holm = 9.1e-06
- miou, rgb - depth: +0.288 [+0.229, +0.345]; video CI [0.235, 0.339]; median +0.287; p_holm = 8.9e-08

## 4. Identity, all windows, GT linking `consecutive_annotated` (`04_identity/`)

| subset | metric | rgb | depth | normal |
|---|---|---|---|---|
| all | id_switch_rate | 0.050 [0.038, 0.063] | 0.042 [0.024, 0.063] | 0.040 [0.024, 0.060] |
| all | class_switch_rate | 0.003 [0.000, 0.007] | 0.006 [0.001, 0.012] | 0.005 [0.001, 0.010] |
| all | fragmentation_mean | 1.098 [1.066, 1.139] | 1.087 [1.043, 1.139] | 1.085 [1.051, 1.128] |
| all | fragmentation_ge2 | 0.096 [0.064, 0.138] | 0.085 [0.041, 0.138] | 0.085 [0.052, 0.127] |
| all | survival_forward | 0.851 [0.797, 0.899] | 0.905 [0.859, 0.945] | 0.897 [0.858, 0.931] |
| all | survival_backward | 0.867 [0.836, 0.896] | 0.916 [0.880, 0.948] | 0.898 [0.869, 0.924] |
| all | reappearance_rate | 0.366 [0.295, 0.435] | 0.251 [0.185, 0.322] | 0.299 [0.231, 0.370] |
| all | reappear_same_gt | 0.119 [0.000, 0.286] | 0.143 [0.000, 0.429] | 0.222 [0.000, 0.556] |
| all | reappear_same_class | 0.857 [0.679, 1.000] | 0.929 [0.786, 1.000] | 0.917 [0.750, 1.000] |

- id_switch_rate, normal - rgb: -0.010 [-0.027, +0.009]; video CI [-0.028, 0.010]; median -0.007; p_holm = 0.626
- id_switch_rate, normal - depth: -0.002 [-0.023, +0.022]; video CI [-0.023, 0.025]; median +0.000; p_holm = 0.952
- id_switch_rate, rgb - depth: +0.008 [-0.011, +0.028]; video CI [-0.012, 0.030]; median +0.000; p_holm = 0.952
- class_switch_rate, normal - rgb: +0.002 [-0.003, +0.008]; video CI [-0.003, 0.008]; median +0.000; p_holm = 1.000
- class_switch_rate, normal - depth: -0.001 [-0.008, +0.006]; video CI [-0.007, 0.006]; median +0.000; p_holm = 1.000
- class_switch_rate, rgb - depth: -0.003 [-0.008, +0.002]; video CI [-0.008, 0.003]; median +0.000; p_holm = 1.000
- fragmentation_mean, normal - rgb: -0.013 [-0.039, +0.014]; video CI [-0.044, 0.018]; median +0.000; p_holm = 0.867
- fragmentation_mean, normal - depth: -0.002 [-0.048, +0.041]; video CI [-0.050, 0.047]; median +0.000; p_holm = 0.970
- fragmentation_mean, rgb - depth: +0.011 [-0.031, +0.051]; video CI [-0.037, 0.050]; median +0.000; p_holm = 0.867
- fragmentation_ge2, normal - rgb: -0.011 [-0.038, +0.016]; video CI [-0.043, 0.020]; median +0.000; p_holm = 0.789
- fragmentation_ge2, normal - depth: +0.000 [-0.044, +0.042]; video CI [-0.046, 0.048]; median +0.000; p_holm = 0.852
- fragmentation_ge2, rgb - depth: +0.012 [-0.028, +0.049]; video CI [-0.033, 0.051]; median +0.000; p_holm = 0.764
- survival_forward, normal - rgb: +0.046 [+0.022, +0.073]; video CI [0.019, 0.078]; median +0.052; p_holm = 0.006
- survival_forward, normal - depth: -0.008 [-0.033, +0.020]; video CI [-0.034, 0.015]; median -0.009; p_holm = 0.313
- survival_forward, rgb - depth: -0.054 [-0.089, -0.023]; video CI [-0.090, -0.023]; median -0.049; p_holm = 0.006
- survival_backward, normal - rgb: +0.030 [-0.001, +0.060]; video CI [0.000, 0.058]; median +0.027; p_holm = 0.077
- survival_backward, normal - depth: -0.019 [-0.042, +0.007]; video CI [-0.040, 0.002]; median -0.041; p_holm = 0.077
- survival_backward, rgb - depth: -0.049 [-0.086, -0.012]; video CI [-0.089, -0.010]; median -0.042; p_holm = 0.049
- reappearance_rate, normal - rgb: -0.066 [-0.110, -0.021]; video CI [-0.105, -0.024]; median -0.086; p_holm = 0.017
- reappearance_rate, normal - depth: +0.048 [-0.010, +0.105]; video CI [-0.011, 0.111]; median +0.033; p_holm = 0.167
- reappearance_rate, rgb - depth: +0.114 [+0.055, +0.174]; video CI [0.055, 0.173]; median +0.111; p_holm = 0.007
- reappear_same_gt, normal - rgb: +0.062 [+0.000, +0.188]; video CI [0.000, 0.214]; median +0.000; p_holm = 1.000
- reappear_same_gt, normal - depth: +0.000 [+0.000, +0.000]; video CI [0.000, 0.000]; median +0.000; p_holm = n/a
- reappear_same_gt, rgb - depth: +0.000 [+0.000, +0.000]; video CI [0.000, 0.000]; median +0.000; p_holm = n/a
- reappear_same_class, normal - rgb: +0.031 [-0.094, +0.188]; video CI [-0.094, 0.188]; median +0.000; p_holm = 1.000
- reappear_same_class, normal - depth: -0.062 [-0.188, +0.000]; video CI [-0.188, 0.000]; median +0.000; p_holm = 1.000
- reappear_same_class, rgb - depth: +0.000 [+0.000, +0.000]; video CI [0.000, 0.000]; median +0.000; p_holm = n/a

## 4. Identity, frames_in_order windows, GT linking `consecutive_annotated` (`04_identity/`)

| subset | metric | rgb | depth | normal |
|---|---|---|---|---|
| frames_in_order | id_switch_rate | 0.044 [0.029, 0.061] | 0.043 [0.020, 0.072] | 0.028 [0.016, 0.041] |
| frames_in_order | class_switch_rate | 0.004 [0.000, 0.010] | 0.005 [0.000, 0.013] | 0.004 [0.000, 0.010] |
| frames_in_order | fragmentation_mean | 1.065 [1.040, 1.092] | 1.059 [1.020, 1.109] | 1.051 [1.024, 1.082] |
| frames_in_order | fragmentation_ge2 | 0.065 [0.040, 0.091] | 0.059 [0.019, 0.110] | 0.051 [0.024, 0.081] |
| frames_in_order | survival_forward | 0.862 [0.811, 0.909] | 0.916 [0.870, 0.955] | 0.903 [0.865, 0.935] |
| frames_in_order | survival_backward | 0.871 [0.829, 0.911] | 0.928 [0.889, 0.961] | 0.907 [0.874, 0.938] |
| frames_in_order | reappearance_rate | 0.350 [0.272, 0.431] | 0.227 [0.167, 0.289] | 0.286 [0.211, 0.361] |
| frames_in_order | reappear_same_gt | 0.019 [0.000, 0.056] | 0.000 [0.000, 0.000] | 0.167 [0.000, 0.500] |
| frames_in_order | reappear_same_class | 0.833 [0.611, 1.000] | 0.917 [0.750, 1.000] | 0.875 [0.625, 1.000] |

- id_switch_rate, normal - rgb: -0.016 [-0.033, -0.001]; video CI [-0.029, 0.000]; median -0.009; p_holm = 0.091
- id_switch_rate, normal - depth: -0.015 [-0.041, +0.008]; video CI [-0.039, 0.012]; median -0.005; p_holm = 0.443
- id_switch_rate, rgb - depth: +0.001 [-0.022, +0.025]; video CI [-0.024, 0.027]; median +0.000; p_holm = 0.975
- class_switch_rate, normal - rgb: +0.000 [-0.008, +0.008]; video CI [-0.008, 0.007]; median +0.000; p_holm = 1.000
- class_switch_rate, normal - depth: -0.001 [-0.010, +0.007]; video CI [-0.007, 0.006]; median +0.000; p_holm = 1.000
- class_switch_rate, rgb - depth: -0.001 [-0.008, +0.006]; video CI [-0.008, 0.008]; median +0.000; p_holm = 1.000
- fragmentation_mean, normal - rgb: -0.014 [-0.055, +0.029]; video CI [-0.063, 0.040]; median -0.003; p_holm = 0.984
- fragmentation_mean, normal - depth: -0.008 [-0.062, +0.043]; video CI [-0.066, 0.049]; median +0.000; p_holm = 0.984
- fragmentation_mean, rgb - depth: +0.006 [-0.038, +0.043]; video CI [-0.051, 0.042]; median +0.012; p_holm = 0.984
- fragmentation_ge2, normal - rgb: -0.014 [-0.055, +0.028]; video CI [-0.062, 0.038]; median -0.003; p_holm = 0.984
- fragmentation_ge2, normal - depth: -0.008 [-0.061, +0.041]; video CI [-0.068, 0.048]; median +0.000; p_holm = 0.984
- fragmentation_ge2, rgb - depth: +0.006 [-0.039, +0.043]; video CI [-0.051, 0.041]; median +0.012; p_holm = 0.984
- survival_forward, normal - rgb: +0.041 [+0.015, +0.069]; video CI [0.018, 0.069]; median +0.040; p_holm = 0.064
- survival_forward, normal - depth: -0.013 [-0.046, +0.020]; video CI [-0.056, 0.020]; median -0.014; p_holm = 0.528
- survival_forward, rgb - depth: -0.054 [-0.102, -0.013]; video CI [-0.111, -0.016]; median -0.040; p_holm = 0.067
- survival_backward, normal - rgb: +0.035 [+0.008, +0.063]; video CI [0.013, 0.055]; median +0.035; p_holm = 0.077
- survival_backward, normal - depth: -0.021 [-0.052, +0.014]; video CI [-0.052, 0.009]; median -0.044; p_holm = 0.100
- survival_backward, rgb - depth: -0.057 [-0.094, -0.020]; video CI [-0.099, -0.015]; median -0.048; p_holm = 0.033
- reappearance_rate, normal - rgb: -0.064 [-0.119, -0.004]; video CI [-0.102, -0.011]; median -0.096; p_holm = 0.094
- reappearance_rate, normal - depth: +0.059 [-0.016, +0.135]; video CI [-0.035, 0.147]; median +0.050; p_holm = 0.175
- reappearance_rate, rgb - depth: +0.123 [+0.056, +0.192]; video CI [0.040, 0.199]; median +0.146; p_holm = 0.028
- reappear_same_gt, normal - rgb: +0.000 [+0.000, +0.000]; video CI [0.000, 0.000]; median +0.000; p_holm = n/a
- reappear_same_gt, normal - depth: +0.000 [+0.000, +0.000]; video CI [0.000, 0.000]; median +0.000; p_holm = n/a
- reappear_same_gt, rgb - depth: +0.000 [+0.000, +0.000]; video CI [0.000, 0.000]; median +0.000; p_holm = n/a
- reappear_same_class, normal - rgb: -0.050 [-0.150, +0.000]; video CI [-0.188, 0.000]; median +0.000; p_holm = 1.000
- reappear_same_class, normal - depth: -0.062 [-0.188, +0.000]; video CI [-0.188, 0.000]; median +0.000; p_holm = 1.000
- reappear_same_class, rgb - depth: +0.000 [+0.000, +0.000]; video CI [0.000, 0.000]; median +0.000; p_holm = n/a

The same with GT linking only between adjacent samples is in `04_identity/summary.csv` (`link == adjacent_samples`).

## 5. Instrument-tissue merges (`05_merge_rate/`)

| level | metric | rgb | depth | normal |
|---|---|---|---|---|
| all | merge_region_rate | 0.000 [0.000, 0.000] | 0.048 [0.015, 0.088] | 0.006 [0.000, 0.014] |
| all | merge_frame | 0.000 [0.000, 0.000] | 0.195 [0.072, 0.343] | 0.059 [0.000, 0.138] |

- merge_region_rate, normal - rgb: +0.006 [+0.000, +0.014]; video CI [0.000, 0.013]; median +0.000; p_holm = 0.109
- merge_region_rate, normal - depth: -0.042 [-0.082, -0.011]; video CI [-0.078, -0.012]; median +0.000; p_holm = 0.035
- merge_region_rate, rgb - depth: -0.048 [-0.089, -0.015]; video CI [-0.088, -0.014]; median +0.000; p_holm = 0.035
- merge_frame, normal - rgb: +0.059 [+0.000, +0.138]; video CI [0.000, 0.124]; median +0.000; p_holm = 0.109
- merge_frame, normal - depth: -0.135 [-0.256, -0.036]; video CI [-0.247, -0.038]; median +0.000; p_holm = 0.050
- merge_frame, rgb - depth: -0.195 [-0.338, -0.070]; video CI [-0.340, -0.060]; median +0.000; p_holm = 0.034

## 6. Nameability (`06_naming/`)

| level | metric | rgb | depth | normal |
|---|---|---|---|---|
| all | named_one | 0.907 [0.885, 0.927] | 0.752 [0.718, 0.784] | 0.830 [0.799, 0.862] |
| all | named_many | 0.028 [0.012, 0.048] | 0.174 [0.139, 0.212] | 0.078 [0.053, 0.107] |
| all | named_none | 0.065 [0.051, 0.080] | 0.074 [0.051, 0.100] | 0.092 [0.071, 0.112] |

- named_one, normal - rgb: -0.077 [-0.114, -0.041]; video CI [-0.114, -0.040]; median -0.066; p_holm = 0.002
- named_one, normal - depth: +0.078 [+0.040, +0.115]; video CI [0.031, 0.126]; median +0.094; p_holm = 0.002
- named_one, rgb - depth: +0.155 [+0.119, +0.191]; video CI [0.117, 0.194]; median +0.175; p_holm = 6.3e-07
- named_many, normal - rgb: +0.050 [+0.019, +0.083]; video CI [0.016, 0.096]; median +0.046; p_holm = 0.005
- named_many, normal - depth: -0.096 [-0.136, -0.055]; video CI [-0.144, -0.045]; median -0.099; p_holm = 1.9e-04
- named_many, rgb - depth: -0.146 [-0.187, -0.105]; video CI [-0.189, -0.108]; median -0.153; p_holm = 7.0e-05
- named_none, normal - rgb: +0.027 [+0.002, +0.051]; video CI [-0.001, 0.051]; median +0.016; p_holm = 0.258
- named_none, normal - depth: +0.017 [-0.015, +0.050]; video CI [-0.017, 0.049]; median -0.009; p_holm = 1.000
- named_none, rgb - depth: -0.009 [-0.037, +0.015]; video CI [-0.034, 0.017]; median +0.003; p_holm = 1.000

Named events (`06_naming/fig6_named_events.txt`; not Fig. 6's window, whose graph is not on this machine, see `00_inventory.md`):

```
# window VID12_s15_19900_crop, focus node 6, partner node 9
t=4: L-hook electrocautery (node 6) re-enters the view
t=20: L-hook electrocautery (node 6) moves from right of to above the gallbladder (node 9)
t=26: L-hook electrocautery (node 6) moves from right of to above the gallbladder (node 9)
t=28: L-hook electrocautery (node 6) moves from right of to above the gallbladder (node 9)
```
