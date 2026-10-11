# Porting plan

**Status: in progress.** This is the plan for moving the code out of the
private research workbench (`$OMNISURG_SOURCE`) into this repository. It is
deleted once the port is done; git history keeps it.

## Progress

A check compares the port with the workbench on the same input; "—" means
no check is recorded.

| area | ported | checked |
|---|---|---|
| evaluator, `evalkit/` | class tables, metrics, entry point, pilot mode | in pilot mode, every key shared with the pilot evaluator equal on 23 conditions × 315 clips, at `eval_code_sha` `a5d7ea46…` |
| score tools, `evalkit/tools/` | every tool in the layout of `AGENTS.md` but `track_metrics` | `paired_table`, `arms_paired`, `claims_table`, `leave_one_video_out`, `print_status_tables`, `kmerge`: the workbench's values on the same inputs |
| camera trajectory scorer, `trajectory_eval/` | `pose_metrics`, the scoring of a clip, `pose_controls`, `scores` | `pose_metrics`: the workbench's values on 1200 random, planar and static trajectories. The scoring of a clip and `pose_controls`: the workbench's 387 StereoMIS rows, every value, scored again from its stored outputs |
| extraction and crop | ATLAS-120k, CholecSeg8k, LapEx, StereoMIS, the D4D population | the CholecSeg8k crop, 13 of the 27 CholecSeg8k clips and the D4D population: the workbench's bytes |
| depth | DA3, Pi3X, point clouds, the population drivers | DA3, point clouds and Pi3X: the workbench's bytes on the GPU machine, on three clips |
| segmentation and tracking | `track`, `per_frame`, `condition_population` | `track` and `per_frame`: the workbench's bytes on the GPU machine, on three clips. `condition_population`: the workbench's commands, for 43 conditions. The 40 segmenter inputs that `tests/test_render_outputs_are_pinned.py` pins: the workbench's, rendered with each case's ring setting |
| viewer | `viewer/`, `pipeline/viewer_bundle` | — (new: the workbench's demo read other files) |

The three depth stages were compared on
`adrenalectomy__16GPCUPkXYQ__gt_0004`,
`adrenalectomy__16GPCUPkXYQ__tile_0007` and `VID01_s15_80_crop`. The Pi3X
stage's manifest differed in `runtime_sec` alone. The workbench ran at the
determinism measurement's commit, and wrote the same bytes twice in each
case before the comparison. The depth stage wrote its point clouds. The
point-cloud stage read the bundle that the workbench's depth stage had
stored on that machine. Both sides ran DA3 at commit `41736238` and `pi3`
at commit `9fa3ddb3`.

The tracking and per-frame stages were compared at the settings of
`op_edge_center` (under both propagation rules), `rgb_center18`,
`op_edge_perframe`, `op_rgb_perframe_pps24` and `op_edge_pf_pps8_pi3x`,
with the edge ring masked. These settings of the paper's conditions were
not compared:

- `--seed-labels`, which the `t5_*` conditions use;
- the segmenter inputs `normal`, `depth` and `rgb_edge`;
- `--keep-edge-ring`, which `ch_edge_center` and `ch_edge_perframe` most
  likely used.

## Not ported yet

- **`track_metrics`**, the measures of identity over time. It waits on "The
  GT track of `track_metrics`". It is ported without its minimum area,
  `MIN_AREA`, since the evaluator has no minimum object size.
- **The export stage**: the regions, scene graphs, hierarchies and graphs
  through time that `pipeline.viewer_bundle` reads, with the `surgical_core`
  modules they need. Its byte check needs a clip whose tracking labels the
  workbench holds.
- **The paper's tables**: the claims grid and the StereoMIS table.
- **The StereoMIS scoring**: the command that scores each method and each
  control on every clip.
- **D4D**: the census, depth, seeds, the front/behind relation and the
  camera trajectory. The reader waits on "How this repository depends on
  D4D's loader".
- **LapEx's comparisons**. They wait on "How LapEx is scored".
- **The conditions seeded from GT masks** (`t12_gtseed`, `t12_paste`). They
  wait on "Conditions seeded from GT".
- **Two comparisons of the granularity result**: the geometry inputs
  against `rgb` at a matched number of regions, and merges that never join
  two regions across a geometric boundary.
- **A metric guide on the evaluator.** `docs/pilot_metric_guide.html` shows
  the pilot evaluator. A page built from `tests/scenes.py` and
  `tests/cartoon.py` would show the evaluator's values, taken from the
  tests' pinned values.
- **Short module headers** for the modules `STILL_LONG` lists in
  `tests/test_module_headers_are_short.py`. The hashed ones go before the
  freeze.

## Decisions still to carry out

1. **Pilot mode is removed before the freeze.** Pilot mode stays in the
   evaluator until just before the freeze, so that the check runs through
   the entry point that produces the paper's numbers. Pilot mode is
   `python -m evalkit.evaluate --pilot`, which scores each clip through
   `evalkit/pilot_clip.py`; it is not a script of its own. Just before the
   freeze, everything in the hashed files that only pilot mode uses is
   removed:
   - `evalkit/pilot.py` and `evalkit/pilot_clip.py`, and their entries in
     `evalkit/code_sha.py`;
   - the entry point's `--pilot`, and the `pilot` argument of
     `score_condition` with the imports it needs;
   - the `pilot` arguments of `evalkit/scored.py` and `evalkit/inst_bf.py`;
   - every sentence of a hashed module that says what pilot mode does.

   The normal-mode tests must still pass, and then the evaluator is frozen.
   Moving pilot mode out of `evalkit/` was the alternative. It could then be
   deleted at any time, but the check would reach a driver of its own and
   never the entry point.
2. **What the paper measures.** Decided on 2026-10-10:
   - **The `forward_from_first` conditions.** The paper reports them. They
     are measured on the GPU machine before submission.
   - **The conditions seeded from GT masks.** They are measured again with
     this repository's code. Open: whether the paper reports them.
   - **`track_metrics`.** It is run on the paper's conditions. Open: how
     the paper reports its measures.
3. **The conditions take their names at the re-score.**
   `evalkit/tools/condition_names.tsv` maps each workbench condition that
   the paper's tables read to its name here, or says why its records show
   none. With the re-score, and not before, the label directories and the
   score JSONs take these names, and `paired_table`, `claims_table` and
   `arms_paired` read their tags from the table. Until then the tools read
   the workbench's names.

## Remaining steps

1. **Re-score.** Score every condition's predictions with the evaluator.
   `condition_inventory` must then report no condition missing and no two
   that are not comparable. After any change to a hashed file, the
   check against the pilot evaluator runs again before anything is
   re-scored.
2. **Port what is not ported yet.**
   - A tool prints the workbench's values on the pilot's score JSONs, apart
     from the differences its port lists. A test plants each fault that
     only the evaluator's JSONs can carry.
   - A stage writes the workbench's files byte for byte, apart from the
     exceptions below. Both sides run on one machine, in one environment,
     so that a difference is the code's. If the GPU machine's packages no
     longer equal those the determinism measurement recorded, the
     workbench's stage first runs twice.
   - The conditions under `forward_from_first` are new measurement, not a
     port. They are made on the GPU machine and scored as in step 1.
3. **Freeze the evaluator.** Pilot mode is removed first ("Pilot mode is
   removed before the freeze"), the hashed modules' headers are shortened,
   and the questions under "Before the freeze" are answered. Then the
   freeze rule of `AGENTS.md` applies.
4. **Prepare the release.**
   - `CITATION.cff`, once the anonymity rules below allow the authors to be
     named. A clip's layout is specified in `docs/pipeline.md`, "A clip".
   - The `recon3d` extra pins Depth Anything 3 and Pi3 to the commits the
     workbench's GPU machine ran, read from pip's `direct_url.json` there.
     A constraints file beside `pyproject.toml` lists that machine's
     packages.
   - Before the release, a check of the anonymity rules of Healthcare
     Technology Letters, since AE-CAI is under double-blind revision.
   - Done when a third party can clone, install and get a green `pytest`,
     and, on a machine with a CUDA GPU, install the pipeline with the
     README's steps and run each ported stage on a clip.

## Byte equality exceptions

A ported stage's output may differ from the workbench's in these places
only:

- **Pi3X's `runtime_sec`.**
- **CholecSeg8k clips whose frames run out of order.** In videos that
  CholecSeg8k numbers at about 30 fps, the workbench's extractor placed the
  frames between annotations 1 to 3 s late, and one image could appear
  twice. This affects 14 of the 27 clips. The ported extractor converts
  those frames, and writes the other 13 clips byte for byte. Every
  CholecSeg8k condition is run again on the 27 clips the port extracts.
- **A frame that the ATLAS-120k extraction shrinks.** OpenCV's area resize
  rounds differently from one build to another: about one pixel in ten
  thousand differs by one level. So frames are compared on one machine.
- **The ATLAS-120k manifest key `pixel_source`.** The workbench's extractor
  gained it after the GPU machine's clips were extracted.

## Open questions

Each question says what it blocks and where its answer is written.

### Before the freeze

- **Boundary dilation.** `evalkit/boundary.py` dilates with `cv2.dilate`.
  A numpy shift-or agrees on every mask tried, borders included. Should a
  hashed file depend on OpenCV's behaviour while OpenCV is unpinned?
  Blocks: the freeze. Answer in: `evalkit/boundary.py`.
- **CholecSeg8k's Region line.** CholecSeg8k draws a 1 px white line
  between regions, which the class table types `ignored`. 87 % of its
  pixels are the image's outer pixel, which the crop removes. The rest
  sits mostly in video43 and video52, where an edge against it is no
  boundary, so those videos lose much of their GT boundary. Does the loader
  fill the line from its neighbours, by a fixed rule with the count
  recorded, or does the line stay ignored with the loss documented?
  Blocks: the freeze. Answer in: `docs/evaluation.md`, the CholecSeg8k table.
- **Seeds made outside the tracking stage.** `rule_of_seed_info` refuses a
  `seed_source` other than `sam` or `per_frame`. So the evaluator refuses
  the six `t5_*` conditions of the granularity result, and every condition
  that `pipeline/track.py --seed-labels` makes. The record names the seeds
  by the path as given, so one directory given by two paths reads as two
  conditions. Open: which values of `seed_source` the evaluator scores, how
  a score records and names its seeds' source, and which comparisons the
  tools refuse across sources. Blocks: the freeze, and the granularity
  result. Answer in: `docs/evaluation.md`, which says no such condition is
  reported.
- **Conditions seeded from GT.** The workbench seeded `t12_gtseed` and
  `t12_paste` from GT masks. On the 151 ATLAS-120k clips with an even
  number of frames, the seed sits one frame before the centre:
  `track_metrics.pick_seed_frame` breaks a tie toward the earlier frame,
  while the tracking stage and `rule_of_seed_info` take `frames[n // 2]`.
  So on those clips the conditions hold neither propagation rule. Open:
  which frame a GT seed is placed on under each rule and when the centre
  frame has no GT, which code places it, and how the evaluator scores it.
  Blocks: measuring the `t12_*` conditions again. Answer in:
  `docs/evaluation.md` and `docs/pipeline.md`.
- **Cuts inside GT clips.** The workbench searched for scene changes inside
  GT clips at frame pairs whose pixel difference was above 40, and judged
  40 of those 125 pairs. `cholecystectomy__1ud3syYKD3A__gt_0001` holds six
  judged cuts, and `cholecystectomy__Bj13QcLRCVc__gt_0001` one undecided
  candidate. After a cut, a tracking condition's scores measure the scene
  change, not the input. Open: judge the 85 remaining pairs, then use a
  clip with a cut whole, exclude it, or split it at the cut. Blocks:
  extracting the population again, and the freeze. Answer in:
  `atlas120k_meta/README.md`.

### Conditions

- **The edge ring of the tracker's input.** The tracking stage builds the
  seed's input and the tracker's input with one ring setting.
  `seed_info.json` records the setting once, as
  `seed_input.edge_ring_masked`, for the seed's input only. When the seed
  came from GT or from outside the stage, or the seed's input burns no
  edges, the record holds no setting, even when the tracker's input burns
  edges. So `check_provenance` and `condition_inventory` cannot compare it.
  Open: where the record holds the tracker's setting. Answer in:
  `docs/pipeline.md`, the tracking stage.
- **Depth made by two versions of the depth stage.** On the development
  machine, 7 of the 9 CholecSeg8k clips carry depth from a branch of the
  workbench's depth stage that never reached its `main`: their bundle holds
  a `ray_map`, and DA3 ran with `ref_view_strategy="middle"`. The port
  follows `main`, which leaves DA3 at its default. So the byte check
  reproduces `main`, not necessarily the depth a scored condition read.
  Open: which version made the depth each scored ATLAS-120k condition
  read, and where it was the branch, whether that depth is made again.
  Answer in: `docs/pipeline.md`, the depth stage.
- **The order of tied presence scores.** The tracking stage paints a
  frame's objects in the order of their presence scores, through
  `np.argsort`. On the prompted frame all of them tie, and above 16 tied
  objects their order depends on numpy's build and the CPU. Where tied
  masks overlap, a label can then differ between machines. Open: keep the
  sort, or sort stably and list the change among the byte equality
  exceptions. Answer in: `docs/pipeline.md`, the tracking stage.

### Tools and tables

- **The GT track of `track_metrics`.** The datasets carry no ids for
  individual things, so the tool defines its own GT track. The workbench
  linked each class's connected components over time, greedily, at an IoU
  of 0.3 or more, keeping a track through three missed observations. Open:
  whether a track is a whole class followed over time or linked
  components, and then the linking rule; whether a gap counts in
  observations or seconds; which frame hold starts from when the seed frame
  has no GT; which `hold_mean` definition keeps the name; and what hold
  means under `forward_from_first`. Blocks: porting `track_metrics`.
  Answer in: `docs/evaluation.md`.
- **The paired table on the evaluator's scores.** `paired_table` prints the
  pilot's table only. The six `t5_*` rows wait on "Seeds made outside the
  tracking stage". A table holds one propagation rule, so it is printed
  once per rule. The evaluator records `objects.predicted` in place of
  `n_regions_mean`. Open: which block answers which of the paper's three
  questions ("Primary metrics" in `docs/evaluation.md`), and which rows,
  keys and views each table holds. Answer in: `paired_table`'s docstring.
- **The claims table on the evaluator's scores.** `claims_table` prints the
  workbench's table from the pilot's scores. Its three rows that read
  `t5_k10` wait on "Seeds made outside the tracking stage". The workbench
  read every claim on `inst_F1_50_labeled`, whose nearest view here is
  `all`, and on the 13-video ATLAS-120k set, where the port makes the
  conditions on 315 clips. Open: which key or keys the claims are read on,
  whether the granularity result answers a question whose primary metrics
  are fixed, and which population the ATLAS-120k column reads. Answer in:
  `claims_table`'s docstring.
- **The pairs `arms_paired` reports.** The workbench compared 21 pairs of
  inputs by default. Two of them compare inputs the paper's conditions
  ran: `rgb_edge` with `rgb`, and `normal_edge` with `normal`. The port
  carries no list, because its tags change at the re-score ("The conditions
  take their names at the re-score"). Open: which
  pairs the paper reports with the tool. Answer in: `arms_paired`'s
  docstring.
- **The benchmark mapping against its source.** The ATLAS-120k mapping to
  the benchmark's 30 classes was checked by hand against ATLAS-bench's
  `datasets/class_mapping.py` at commit e286a584, and all 47 ids agree.
  Open: whether a script that takes that file's path repeats the check.
  Answer in: `docs/evaluation.md`, the benchmark's 30 classes.

### Pipeline

- **Masks that are not GT under the GT's name.** The viewer's step writes
  SAM 3 masks into `seg_masks/` under the annotation's name,
  `<i>_color_mask.png`. Only the frame manifest's `is_anchor` and
  `seg_provenance` tell them apart. The evaluator reads those flags, and
  refuses two VID25 clips that hold such masks on frames the manifest marks
  as having none. Open: whether such a mask moves out of `seg_masks/` or
  takes a name of its own. Blocks: porting the step that writes them.
  Answer in: `docs/pipeline.md`.
- **A clip with no usable depth in any frame.** The point-cloud stage skips
  a frame with no usable depth, as the workbench does. A clip with no
  usable depth in any frame gets no cloud and a count of 0, and the run
  exits 0. No real clip has done this. Open: whether the stage refuses
  such a clip. Answer in: `docs/pipeline.md`, the point-cloud stage.

### StereoMIS

- **The times of a StereoMIS clip.** `surgical_core.stereomis.clips` takes
  a frame's time as its position over the declared frame rate. The P2
  videos declare 59.94 frames a second, but by their timestamps a clip's
  frames are 0.217 s apart at the median, not 0.2 s. `rpe` and the
  constant-motion control read the times; `ate_rel` does not. Open:
  whether the times come from the timestamps. Blocks: the StereoMIS result.
  Answer in: `surgical_core/stereomis.py`.
- **P1's offsets.** The workbench took P1's shipped frame N to be position
  N; measured against the video, it is N − 1. With a file offset of −1, as
  P2's is, 504 of the 672 frames of P1's usable clips get a mask of their
  own number, and the common ground truth offset becomes −3 rather than −4.
  Every clip then moves one frame, its depth is made again, and its
  `ate_rel` changes. Open: both offsets, and whether the grid's start is
  fixed apart from them. Blocks: the StereoMIS result. Answer in:
  `surgical_core/stereomis.py`.
- **What fixes the StereoMIS clips.** `surgical_core.stereomis.clips`
  computes the 86 clips on every run, from the tables of offsets and
  thresholds and from the `stats.npy` of the workbench's depth export,
  which no tracked file makes. A change to an offset moves every
  `ate_rel`, but `trajectory_code_sha` covers the modules of
  `trajectory_eval` only. Open:
  whether the clips are kept as data, as ATLAS-120k's population is;
  whether the hash covers the tables, the clip rule and `pose_controls`;
  and whether the export is made again here. Blocks: the StereoMIS result.
- **The StereoMIS depth outputs the ported stages do not reproduce.**
  `pipeline.depth_stereomis` refuses six of the workbench's clips. Five
  masked clips of `P2_0` record `border_inpaint: true`: the workbench's
  stage took their paint for a border and filled it, and the port fills
  none, so their `da3_masked` will change. `P1__clip_0001` ran Pi3X without
  `--max-points`, which changes no score. Open: whether the masked
  condition is measured with the port as it is, and whether it is reported;
  and whether `P1__clip_0001`'s Pi3X output is made again. Blocks: the StereoMIS result.

### D4D

- **How this repository depends on D4D's loader.** The workbench imports
  `d4d.loader` from a clone of https://github.com/reubendocea/d4d at commit
  `efe12d26`. pip cannot install it: the project's find rule includes only
  `d4d_surgical*`, while the package is `d4d/`. The repository has no
  licence file, though its `pyproject.toml` declares MIT. Open: upstream
  fixes both and an extra pins the commit; a fork carries the fix; or the
  README says to put the clone on `PYTHONPATH`. Blocks: the census and the
  D4D reader. Answer in: `pyproject.toml` and the README.
- **Two inputs of the census whose code no repository holds.** The census
  reads `moved_camera` from `clip_table.json`, and `active` from
  `tissue_motion.json`. Scripts in the dataset's directory wrote both
  files. `detect_tissue_motion.py` passes `cv2.INTER_NEAREST` as
  `cv2.resize`'s third positional argument, `dst`, so in 11 of the 271
  clips it resized the instrument masks bilinearly. Whether that changes
  any `tissue_moving` side is not measured. Open: whether the two
  scripts are ported or their files kept as data, and whether the motion
  record is made again. Answer in: `d4d_meta/README.md`.
- **The tissue's motion when a side was scanned.** The motion record
  covers the clip only, so `active` is taken at the clip's first moment for
  `start` and its last for `end`. The scans lie outside the clip, at the
  median 4.4 s before and 4.3 s after it. The record also stops before a
  clip's last frame, so no `end` side is `active`. Open: keep this
  stand-in, written down, or measure the motion up to each scan. Blocks:
  reporting the D4D numbers. Answer in: `d4d_meta/README.md`.

### LapEx

- **How LapEx is scored.** The evaluator holds no LapEx class table, so
  neither it nor `kmerge` reads a LapEx clip. Open:
  - its class table, where level 0, interstitial space, is a class that the
    extraction writes as 11;
  - its depth: the workbench filled the endoscope's border on 710 of the
    735 clips, leaving zeros the evaluator refuses;
  - its comparison: the workbench resampled 30 case means, which weighs
    the cases unlike `paired_stats`.

  Blocks: scoring LapEx. Answer in: `docs/evaluation.md`.

### The public viewer

- **What the public viewer may publish.** A bundle holds a clip's frames
  as JPEG, its masks as labels and its scene graphs. Open: whether each
  dataset's terms allow its frames and masks on a public page, with which
  attribution; and what the page says about who made it, which waits on
  the anonymity rules of Healthcare Technology Letters. Blocks: putting a
  bundle online. Answer in: `viewer/README.md`.
- **The size of the point clouds.** A bundled clip is 60 to 90 MB, and its
  clouds are 97 % of it. A clip of 15 frames takes about a minute to load
  on a 10 Mbit/s line. Quantised positions or a compressed GLB would
  shrink them several times, if the points keep their order, which the
  labels index. Open: the encoding, decided with where the bundles are
  hosted. Answer in: `docs/pipeline.md`, the viewer bundle.
- **The workbench's other viewer pages.** The plan leaves behind the
  workbench's skill-classification code and its other 18 viewer pages.
  Some of the paper's figures come from those pages. Open: whether any of
  them is ported. Answer in: this plan.
