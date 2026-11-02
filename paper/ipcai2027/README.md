# The IPCAI 2027 paper's results

The scripts here print the numbers of *Toward 4D Surgical Scene
Understanding Beyond Cholecystectomy: An Assessment of 4D Scene Graph
Construction and Validation* from the pipeline's outputs. Each script takes
every path as an argument, prints a short table and writes one JSON with
the numbers it prints. Run a script by its path from the repository root,
after `pip install -e ".[render,tools,stereomis,d4d]"`.

- [Inputs](#inputs): the clips and the pipeline stages each result reads
- [Results](#results): each number, the command that prints it, its JSON key
- [Where a number differs](#where-a-number-differs)

## Inputs

Each dataset comes from its publisher. The clip lists fix which clips a
number reads.

| dataset | clips | stages that make the inputs |
|---|---|---|
| ATLAS-120k | `atlas120k_meta/clips.txt`, 339 clips | `pipeline.extract_atlas120k`; `pipeline.depth_population` (DA3); `pipeline.pi3x`; seeds by `nodes/gt_seeds.py`, tracked by `pipeline.condition_population track` |
| CholecSeg8k, 25 windows | `clips/cholecseg8k_windows.txt` | `pipeline.extract_cholecseg8k`; `pipeline.crop_cholecseg8k --rects clips/cholecseg8k_crop_rects.json`; DA3; seeds and tracking as for ATLAS-120k |
| CholecSeg8k, 85 dense chunks | `clips/cholecseg8k_dense_chunks.txt` | `pipeline.extract_cholecseg8k --count 20`; `pipeline.crop_cholecseg8k --rects clips/cholecseg8k_dense_crop_rects.json`; DA3; seeds by `graph/component_seeds.py`, then tracking |
| StereoMIS | `stereomis_meta/clips.json`, 41 usable clips after the rule of `geometry/excluded_clips.py` | `pipeline.extract_stereomis`; `pipeline.depth_stereomis` (DA3 and Pi3X); `geometry/mapanything_depth.py`; `geometry/stereo_depth.py`; regions tracked from 4 points per side |
| D4D | `d4d_meta/population.json` (532 scans); `d4d_meta/trajectory_clips.json` (9 clips) | `pipeline.d4d_depth`; `trajectory_eval.tools.score_d4d prepare`, then DA3 and Pi3X on its clips |

The tracking runs, as the paper ran them:

```bash
python paper/ipcai2027/nodes/gt_seeds.py --dataset atlas120k --data-root CLIPS --clips atlas120k_meta/clips.txt --out SEEDS
python -m pipeline.condition_population track --input-dir CLIPS --clips atlas120k_meta/clips.txt --tracks-root TRACKS \
    --tag rgb_gt_forward --seed-labels SEEDS --sam-input rgb --track-base rgb --rule forward_from_first \
    --seed-min-area 400 --sam-ckpt sam_vit_h_4b8939.pth --gpus 0 1 2 3
```

`TRACKS/<clip>/track_rgb_rgb_gt_forward/` then holds a clip's labels, the
`--tag` the scripts below read. The D4D scripts need D4D's loader on
`PYTHONPATH` (the main README's "Install" says how).

## Results

`G`, `N` and `R` stand for `paper/ipcai2027/geometry`, `nodes` and `graph`.

### Geometry (Sect. 4.1, Fig. 3, Tables S4-S6)

| result | paper | command | JSON key |
|---|---|---|---|
| StereoMIS clips left out by the kinematics check | P2_6 clip 7, P2_7 clip 1 | `G/excluded_clips.py --root STEREOMIS --clips stereomis_meta/clips.json --stereo-depth SD --out X` | `excluded` |
| StereoMIS AbsRel, 41 clips | Pi3X 0.083, DA3 0.098, MapAnything 0.105, flat 0.203 | `G/stereomis_depth.py --clips stereomis_meta/clips.json --clip-root CLIPS --stereo-depth SD --exclude EX --out X` | `absrel`, `differences` |
| D4D AbsRel, 532 scans | DA3 0.147 [0.132, 0.163], flat 0.172, half scan 0.0045, 15 % of the gap | `G/d4d_depth.py ... --out SIDES`, then `G/d4d_depth_table.py --sides SIDES --out X` | `all`, `moving`, `static` |
| D4D whole clip | Pi3X 0.084 (52 % [46, 58]); moving: DA3 0.122, Pi3X 0.069, MapAnything 0.075 | same | `all.absrel`, `moving.absrel` |
| StereoMIS relative ATE, 41 clips | Pi3X 0.289 [0.197, 0.388], DA3 0.334 [0.244, 0.429], MapAnything 0.430 [0.377, 0.492], line 0.630, still 1.000 | `trajectory_eval.tools.score_stereomis` (controls, methods), `G/mapanything_trajectories.py`, then `G/trajectory_table.py --stereomis C M MA --d4d D --exclude EX --out X` | `stereomis.conditions`, `stereomis.differences` |
| D4D relative ATE, 9 clips | Pi3X 0.156, DA3 0.142, line 0.743 | `trajectory_eval.tools.score_d4d score`, then as above | `d4d` |
| Reprojection IoU, StereoMIS | Pi3X 0.876, DA3 0.864, MapAnything 0.856, still 0.830; lower-ATE model in 23 of 41 clips; rho -0.00 / -0.03 | `G/reprojection.py --clips stereomis_meta/clips.json --clip-root CLIPS --tracks-root TRACKS --out ROWS`, then `G/reprojection_table.py --rows ROWS --trajectories M --exclude EX --out X` | `mean_iou`, `lower_ate_named`, `rho_with_ate` |

### Nodes (Sect. 4.2, Fig. 4, Tables S8-S9)

| result | paper | command | JSON key |
|---|---|---|---|
| Tracking IoU, ATLAS-120k | 0.893 vs 0.621 fixed (geometric view); vessels and ducts 0.716 (all-class view) | `N/tracking_iou.py --dataset atlas120k --data-root CLIPS --tracks-root TRACKS --tag track_rgb_rgb_gt_forward --clips atlas120k_meta/clips.txt --out X` | `rows["geometric/tissue"]`, `rows["all/<type>"]` |
| Tracking IoU, CholecSeg8k | 0.872 vs 0.452 fixed (geometric view) | same, `--dataset cholecseg8k --clips paper/ipcai2027/clips/cholecseg8k_windows.txt` | same |
| IoU 2.7 s later by class group | vessels and ducts: tracked 0.716, reprojected 0.536, fixed 0.360, +0.175 ★ | `N/reprojection_by_group.py --data-root CLIPS --tracks-root TRACKS --tag track_rgb_rgb_gt_forward --clips atlas120k_meta/clips.txt --out X` | `groups` |

### Graph (Sect. 4.3, Fig. 5, Tables S10-S11)

| result | paper | command | JSON key |
|---|---|---|---|
| Statements per clip over 5.4 s, 243 clips | 20.9 2D + 12.8 temporal, x1.61 [1.57, 1.66]; depth order 2.7, approach 1.8, camera 1.8 | `R/count_statements.py --clips atlas120k_meta/clips.txt --data-root CLIPS --tracks-root TRACKS --tag track_rgb_rgb_gt_forward --out X` | `mean_per_clip`, `reference_ratio` |
| Exits on ATLAS-120k | 506 exits; precision 0.51 -> 0.70, recall 0.63 -> 0.34; shuffled 0.60, border rule 0.56 | `python -m evalkit.tools.exit_vs_loss --dataset atlas120k --clips atlas120k_meta/clips.txt --data-root CLIPS --tracks-root TRACKS --tag track_rgb_rgb_gt_forward --geometry da3 --gt-events --out RUN`; `python -m evalkit.tools.exit_vs_loss_rescore --dataset atlas120k --run RUN ...`; `R/exits_atlas.py --run RUN --out X` | `exits`, `entries`, `within_view_exits` |
| Exits on CholecSeg8k, one node per connected component | 149 exits; precision 0.57 -> 0.76, +0.19 [+0.06, +0.31] ★ | `R/component_seeds.py`, tracking, the exit tool on the three runs, then `R/exits_cholecseg8k.py --windows-run W --class-run C --component-run P --data-root CLIPS --gt-tracks GT --out X` | `component_nodes` |

## Where a number differs

- Two means round differently from the paper's print: the straight-line
  ATE is 0.62950 (printed 0.630) and DA3 with the two excluded clips is
  0.35749 (printed 0.358).
- A few interval endpoints differ by one in the last digit, because the
  draws of `evalkit.tools.paired_stats.boot_ci` differ from those of the
  runs.
- The CholecSeg8k numbers match exactly with the class table of the
  paper's run, where "Region line" was ignored. The current table fills it,
  which moves those numbers by at most 0.003 and the dense chunks' seeds on
  18 of 85 chunks.
- The D4D whole-clip depth (112 frames per clip) was made by a driver that
  is not released yet; `G/d4d_depth.py` reads its depth files.
