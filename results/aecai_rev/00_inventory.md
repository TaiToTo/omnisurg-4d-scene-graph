# 00 Inventory: what the AE-CAI numbers came from

Every statement below was checked against the code of the tag
`paper/aecai2026-endolina-final` (workbench commit `e1d60001`; the run that
made the numbers recorded commit `e4297ae`, clean) or against its outputs. A
statement that was inferred, not checked, says so.

Paths: `$OMNISURG_SOURCE` is the workbench. `$AECAI_REV_WORK/aecai_src` holds
the tag's code, extracted with `git archive` (see `REPRODUCE.md`).

## 1. Pipeline entry points

| stage | where (tagged code) | setting in the AE-CAI run |
|---|---|---|
| depth | `scripts/run_cholec_depth.py` → `da3_surgery_wrapper` | `depth-anything/DA3-LARGE`, `process_res` 504, `--no-border-inpaint` (from each clip's `frame_manifest.json`) |
| normal map | `depth_sam_tracking_experiment/geometry.py` `normal_map`, `sam_input_image` | from depth and intrinsics |
| SAM automatic | `track_sam3.py` `seed_label_map` → `SamAutomaticMaskGenerator` | vit_h, `points_per_side` 24, `pred_iou_thresh` 0.8, `stability_score_thresh` 0.8, `min_mask_region_area` 100 |
| seed masks | `track_sam3.py` `masks_from_label_map` | regions under 400 px dropped (`seed_min_area`) |
| tracking | `track_sam3.py` `propagate_from_seed` | `Sam3VideoInstanceSession("facebook/sam3")`, on RGB, seed at sample 15, forward then backward |
| overlay | `surgical_core/viewer/hierarchy_frame.py` `build_hierarchy_frame` (called by `scripts/export_viewer_dataset.py`) | linked when one side covers ≥ 0.8 of the other (`CONTAIN_TAU`), or IoU ≥ 0.3 (`OVERLAP_TAU`) |
| events | `scripts/export_graph_figure.py` `timeline_events`; graph by `scripts/build_temporal_graph.py` | entry, exit, change of the dominant relation to the nearest node |

The driver is `depth_sam_tracking_experiment/run_w1_sam_input_ablation.sh`
(the 27-window run: `run_all17_pipeline.sh`; the pps sweep:
`run_w1_pps_sweep.sh`).

**The tracker is SAM 3's tracker, not SAM 2's.** The manuscript cites SAM 2
for the memory-based propagation on purpose; the tag's `REPRODUCE.md`
records that decision.

## 2. Evaluation code

All in `depth_sam_tracking_experiment/eval_track.py`; the table is
`aggregate_w1.py` (unweighted mean over windows).

- **Instance F1** (`instance_metrics`, `_match_ious`): GT instances are the
  connected components of each non-background class; they are matched to the
  predicted regions by IoU, greedily and one to one; F1@t counts matches with
  IoU ≥ t. A window's F1 averages its frames with at least one GT instance.
- **n_reg** (`n_regions`): the number of distinct region ids ≥ 0 in the whole
  label map of the frame. No area cut, no valid-depth restriction. A window
  averages its annotated frames.
- **oracle mIoU** (`GT_mIoU`, `_region_to_class`): each region gets the GT
  class it overlaps most on valid pixels; the class map is scored by mIoU
  over the non-background classes present in GT or prediction.
- **The "300 px" cut** (`INSTANCE_MIN_CC = 300`): an **area** in pixels
  (a component or region is kept when it has ≥ 300 pixels), applied to
  **both** the GT connected components (`_gt_instances`) and the predicted
  regions (`_pred_regions`), counted inside the valid-depth domain. It
  enters instance F1 only; n_reg and oracle mIoU do not use it.
- **Valid domain**: pixels whose depth is finite and > 1e-6.

**300 px as a fraction of the valid area** (`07_filter/legacy_300px_as_fraction.csv`):
median 0.00125, range 0.00118–0.00177 over the 522 annotated frames (median
valid area 239,904 px). Of {0, 0.001, 0.005} the nearest is **0.001**, the
default for every item after item 7.

## 3. The 27 windows

Listed in `run_all17_pipeline.sh` and in `aecai_rev/config.yaml`
(`windows.legacy27`). Each window is `VID<nn>_s15_<start>_crop`: 30 frames
of video `nn`, CholecSeg8k numbers `start`, `start+15`, …, the seed at
sample 15. The rule, recovered from the tag's
`miccai2026_workshop/docs/window_definition_and_coverage_review.md`: each
window starts at the first frame of a run of consecutive CholecSeg8k clips.
Not every run got a window: VID01 has a five-clip run with none, runs longer
than a window spill over, and one VID52 window starts one clip late. 27
windows touch 90 of the 101 clips. `00_windows/README.md` gives this rule
in English.

## 4. CholecSeg8k and Cholec80

- CholecSeg8k: 17 videos, 101 clips of 80 consecutive annotated frames,
  8,080 frames (counted from `$CHOLECSEG8K_ROOT`).
- Cholec80 videos are readable here (`cholec80/cholec80/videos`, 25 fps).
- **CholecSeg8k numbers 5 videos (01, 09, 12, 20, 35) at 25 fps and 12 at
  about 30 fps**: their annotated frames advance 0.833 native frames per
  CholecSeg8k number (measured from the manifests of all 27 windows). A
  stride of 15 numbers is therefore 0.6 s in 10 of the 27 windows and 0.5 s
  in the other 17, whose windows span 14.5 s, not 17.4 s. The manuscript
  states 0.6 s and 17.4 s for all.
- **Known extractor issue, not fixed here**: in the ~30 fps-numbered videos
  the workbench extractor decoded the unannotated frames at the unconverted
  number, 1–3 s later in the video than their place in the window. 11 of
  the 27 windows have native frame numbers that run out of order
  (`04_identity/frames_in_order.csv`); `docs/porting.md` ("CholecSeg8k clips
  whose frames run out of order") counts 14 affected, 11 visibly. The
  annotated frames themselves are right. The tracker saw the misplaced
  frames, so tracking-dependent numbers can be affected; item 4 is reported
  on all 27 windows and on the 16 whose frames run in order.

## 5. Frozen outputs

- Frames, masks, depth: `$OMNISURG_SOURCE/outputs/cholec_gt/<window>/`
  (`input_images/`, `seg_masks/`, `exports/mini_npz/results.npz`). The mask
  files equal the manifest's annotated frames on every window (522).
- Tracks, `label_<frame>.npy` per frame: `depth_sam_tracking_experiment/out_w1_all/`
  (pps 24, the AE-CAI run, 2026-07-07) and `out_w1_pps{8,16,32,48}/`
  (2026-07-09). Seed maps were not kept by those runs. `run_track --seeds`
  wrote them again here (`$AECAI_REV_WORK/seeds/`); `00_check/seed_maps_check.csv`
  compares them with each frozen run's seed frame: equal in 319 of the 324
  conditions of pps 8/16/24/32. In 4 pps-24 conditions the region set or its
  numbering differs, and in 1 pps-8 condition the frozen seed frame holds one
  region fewer. The scores never read these maps; they read the frozen tracks.
- Per-window evaluation JSONs, frozen in the tag:
  `depth_sam_tracking_experiment/miccai2026_workshop/results/eval/w1/`.
- Overlay and graph JSONs: `outputs/cholec_gt/<window>/pc_vis/`. Those of
  the windows checked are dated after the tag (2026-07-31 and later).
- **Fig. 6's graph is not on this machine.** No `temporal_graph__sam3d.json`
  here has nodes 14 and 15 whose relation changes at t = 16, 25, 27 and 29,
  and the tag's `manuscript/images/FIGURES.md` says the figure was made on
  the local PC only.

## 6. GPUs, checkpoints, versions

| run | machine | software |
|---|---|---|
| AE-CAI pps 24 | RTX A6000, Docker | Python 3.12.13, torch 2.12.1+cu126, transformers 5.13.0 |
| AE-CAI pps 8/16/32/48 | RTX A6000 (as recorded) | Python 3.12.3, torch 2.12.1+cu126, transformers 5.13.0 |
| this revision (pps 4/6/12, seed maps) | gpu29, RTX 6000 Ada | Python 3.12.3, torch 2.10.0+cu128, transformers 5.8.1 |

- SAM: `sam_vit_h_4b8939.pth`, sha256 `a7bf3b02…` (same file in all runs).
- Tracker: `facebook/sam3` from the workbench's HF cache.
- Depth: `depth-anything/DA3-LARGE`.
- Re-running pps 24 / normal on `VID01_s15_80_crop` in this environment:
  the seed frame's labels are identical; each of the other 29 frames differs
  from the frozen labels in 1 to 20 pixels (at most 1.1e-4 of the frame). The
  new pps 4/6/12 points therefore come from an environment that reproduces
  the frozen tracks up to a few pixels per frame.
- Depth is not reproduced bit for bit in this environment. Re-estimating
  `VID01_s15_80_crop` with the tagged `scripts/run_cholec_depth.py` here
  gives depth within 0.0078 of the frozen depth (0.6 % of its maximum),
  confidence within 0.057, extrinsics within 0.0023 and intrinsics within
  1.2 px. The 27 AE-CAI windows keep their frozen depth; the 86 enumerated
  windows' depth was estimated here, so a difference between the two
  populations mixes the windows with this environment difference.

## 7. Table 1 reproduced

`00_check/` (`python3 -m aecai_rev.check_legacy`):

| SAM input | F1@.5 | F1@.75 | n_reg | mIoU^oracle | windows | frames |
|---|---|---|---|---|---|---|
| RGB | 0.562 | 0.371 | 15.8 | 0.757 | 27 | 522 |
| depth | 0.480 | 0.291 | 6.7 | 0.469 | 27 | 522 |
| normal | 0.589 | 0.396 | 9.4 | 0.646 | 27 | 522 |

- The tagged `eval_track.py`, re-run on the frozen pps 24 tracks, writes
  JSONs identical to the frozen ones on all 81 window × modality pairs
  (every key and every per-frame value).
- `aecai_rev.score` under the 300 px cut gives the frozen per-frame F1@.5,
  F1@.75, n_reg and oracle mIoU on all 1,566 frame × modality pairs (to the
  4 decimals the JSON keeps).

## 8. Choices made without the author (to confirm)

1. **Window population.** The numbers in `01`–`07` are on the 27 AE-CAI
   windows. `00_windows/` enumerates one window per CholecSeg8k clip
   (86 kept, 15 excluded because they would start before the video does);
   their tables are under `enumerated/`. Their frames lie on a different
   phase of the stride from the AE-CAI windows, so none of the AE-CAI data
   is reused: each window is extracted by this repository's extractor
   (unannotated frames at their true time, so no window runs out of order),
   then cropped from its own frames and depth-estimated by the tagged
   scripts, as the AE-CAI windows were. The crop found from the frames
   varies between windows of one video as it did for the AE-CAI windows
   (VID01's three AE-CAI windows: 435×622, 435×628, 442×618); two VID28
   windows come out 383 and 409 px high where the AE-CAI window is 478.
2. **"Clip" means a CholecSeg8k clip folder** (80 frames), not a run of
   consecutive folders. With runs there would be 41 windows.
3. **The stride counts CholecSeg8k numbers**, as in the AE-CAI windows, not
   native frames (see section 4).
4. **Extended sweep.** At {8, 12, 16, 24, 32} no modality's n_reg brackets
   the mean GT instance count, so the interpolation item 1 asks for cannot
   be computed. pps 4 and 6 were added and the frozen pps 48 read;
   `01_sweep/` marks which points are in the specified grid.
5. **Statistics unit.** Intervals and tests are window-level, as specified.
   Because the 27 windows come from 17 videos, the video-level bootstrap
   interval (this repository's verdict rule) is given beside each.
