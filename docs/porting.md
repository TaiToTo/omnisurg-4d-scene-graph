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
| score tools, `evalkit/tools/` | every tool in the layout of `AGENTS.md` | `paired_table`, `arms_paired`, `claims_table`, `leave_one_video_out`, `print_status_tables`, `kmerge`: the workbench's values on the same inputs. `track_metrics` on component tracks: the values of the workbench's functions on random frames at the four sizes, but where two IoUs tie exactly. `d4d_front_behind`: the workbench's eight front/behind outputs, every field of the 487 sides, the totals and the intervals, scored again from its stored depth and seeds. `d4d_front_behind_verdict`, on the workbench's outputs: its four verdicts, every number |
| camera trajectory scorer, `trajectory_eval/` | `pose_metrics`, the scoring of a clip, `pose_controls`, `scores`, `score_stereomis`, `stereomis_table` | `pose_metrics`: the workbench's values on 1200 random, planar and static trajectories. The scoring of a clip, `pose_controls` and `score_stereomis`: the workbench's 387 StereoMIS rows, every value, scored again from its stored outputs, at the workbench's offsets and times. `stereomis_table`, on those rows: the 24 pairs of the workbench's `summary_20.json`, every field |
| extraction and crop | ATLAS-120k, CholecSeg8k, LapEx, StereoMIS, the D4D census and population | the CholecSeg8k crop and 13 of the 27 CholecSeg8k clips: the workbench's bytes. LapEx's 735 clips: the workbench's frames and masks, and its manifests apart from `depth_info`. The D4D census: every field it writes equal to the workbench's census on the 271 clips. The D4D population: the same bytes from the ported census as from the workbench's |
| depth | DA3, Pi3X, point clouds, the population drivers, the D4D depth stage | DA3, point clouds and Pi3X: the workbench's bytes on the GPU machine, on three clips. DA3 also on the 25 LapEx clips whose border the workbench did not fill. The D4D depth stage: the workbench's bytes on the GPU machine, on the six sides of three clips |
| segmentation and tracking | `track`, `per_frame`, `condition_population`, the D4D seed stage | `track` and `per_frame`: the workbench's bytes on the GPU machine, on three clips. The D4D seed stage, with `--keep-edge-ring`: the workbench's bytes on the six sides the D4D depth stage was compared on. `condition_population`: the workbench's commands, for 43 conditions. The 40 segmenter inputs that `tests/test_render_outputs_are_pinned.py` pins: the workbench's, rendered with each case's ring setting |
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

The D4D depth and seed stages were compared on both sides of
`specimen_1/2025_03_06-16_49_40/Clip_1`,
`specimen_1/2025_03_06-17_39_51/Clip_2`, a clip of 21 frames, and
`specimen_3/2025_05_08-14_24_24/Clip_1`. The workbench's files were made
before its seed stage wrote a record, so the seeds' records were not
compared.

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

- **The export stage**: the regions, scene graphs, hierarchies and graphs
  through time that `pipeline.viewer_bundle` reads, with the `surgical_core`
  modules they need. Its byte check needs a clip whose tracking labels the
  workbench holds.
- **The paper's tables**: the claims grid.
- **D4D**: the camera trajectory.
- **The StereoMIS depth export.** `pipeline.select_stereomis_clips` reads
  its `stats.npy` to mark clips `no_surface`, and no tracked file makes it,
  so the clip list is made again only from the workbench's export. The
  rule leaves out no clip of the list.
- **LapEx's comparisons**. They wait on "How LapEx is compared".
- **The conditions seeded from GT masks** (`t12_gtseed`, `t12_paste`). They
  wait on "Conditions seeded from GT".
- **Seeds made outside the tracking stage, and labels `kmerge` merged, in
  `component_scores` and `track_metrics`.** Both tools refuse them until
  their JSONs record `gt_informed` as the evaluator's score does.
- **The identity table from `track_metrics`.** `print_status_tables
  --identity` reads the JSON the workbench's `summarize_track12` wrote;
  nothing builds the table from the JSON `track_metrics` writes.
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
   none. The `t5_*` rows name the rule their records show. With the
   re-score, and not before, the label directories and the
   score JSONs take these names, and `paired_table`, `claims_table` and
   `arms_paired` read their tags from the table. Until then the tools read
   the workbench's names.
4. **How the regions are compared.** Decided on 2026-10-12:
   - Every condition is scored on whole-class objects by the evaluator,
     and on component objects at 0, 100, 300 and 400 px by
     `component_scores`. `track_metrics` measures both kinds of track.
   - The paper describes how each measure moves across these settings,
     not which condition wins. Open: how the questions and primary
     metrics of `docs/evaluation.md` change.
   - Inputs are compared at a matched number of regions by sweeping the
     segmenter's points per side. Merging regions is not used for it,
     since the merge adds a rule of its own.
   - `pipeline/experiment_plan.json` lists the conditions made for it on
     the GPU machine, on both datasets. Each input that reads depth is made
     on DA3 and on Pi3X. Tracking forward from the first frame is the
     default rule, swept over the grids 2, 4, 8, 12, 24 and 48 in three
     stages; the grid 2 comes last. Tracking from the centre and
     segmenting each frame alone are made at the grid 24.

## Remaining steps

1. **Re-score.** Score every condition's predictions with the evaluator.
   `condition_inventory --require-scored` must then report no problem.
   Its header lists the problems it checks. After any change to a hashed
   file, the check against the pilot evaluator runs again before anything
   is re-scored.
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
- **The tracking stage's `edge_ring_masked`.** The workbench's
  `seed_info.json` has no such key, so `pipeline.byte_check` reports the
  file as different. The reviewer compares the two records by hand with
  this key removed; the stage writes it last.
- **CholecSeg8k clips whose frames run out of order.** In videos that
  CholecSeg8k numbers at about 30 fps, the workbench's extractor placed the
  frames between annotations 1 to 3 s late, and one image could appear
  twice. This affects 14 of the 27 clips. The ported extractor converts
  those frames, and writes the other 13 clips byte for byte. Every
  CholecSeg8k condition is run again on the clips the port extracts, in
  the population of `cholecseg8k_meta/clips.txt`: 25 windows, 11 of
  which moved so that each seeds on a GT frame
  (`cholecseg8k_meta/README.md`).
- **A frame that the ATLAS-120k extraction shrinks.** OpenCV's area resize
  rounds differently from one build to another: about one pixel in ten
  thousand differs by one level. So frames are compared on one machine.
- **The ATLAS-120k manifest key `pixel_source`.** The workbench's extractor
  gained it after the GPU machine's clips were extracted.
- **The D4D population.** The workbench's rule also left out 45 sides as
  `tissue_moving`, from the tissue's motion at the clip's end nearest the
  scan. The rule here keeps them: 532 sides where the workbench kept 487.
- **LapEx's depth.** On 710 of the 735 clips, the workbench filled what it
  took for the endoscope's border and set the depth there to 0. The
  evaluator refuses a frame with depth 0. The depth stage takes the frames
  whole, so its output on those clips differs. On the other 25 clips it
  equals the workbench's.
- **A StereoMIS clip's ground truth rows, times and masks.** The clips are
  read from `stereomis_meta/clips.json`, made with the offsets and the
  timestamps measured again. Every manifest's `gt_row`, `gt_row_offset` and
  `span_mm` differ from the workbench's, so do P2's `time_sec`, and so do
  the images of four masked P1 clips. Each plain clip's images are the
  workbench's.
- **Five masked StereoMIS clips of `P2_0`.** The workbench's depth stage
  took their black paint for a border and filled it (`border_inpaint:
  true`); the ported stage fills none, so their depth differs.
- **The Pi3X clouds of `P1__clip_0001`.** The workbench ran Pi3X on it
  without `--max-points`; the ported driver thins every cloud to 60,000
  points, which leaves the poses alone.

## Open questions

Each question says what it blocks and where its answer is written.

### Before the freeze

- **Mask resize.** `evalkit/inputs.py` resizes a GT mask, and a prediction
  of another shape, to the depth map's shape with `cv2.resize`, nearest
  neighbour. It is the one hashed file that still depends on OpenCV's
  behaviour, and OpenCV is unpinned: a score records its version but cannot
  fix it. Does the resize move to numpy, checked equal to `cv2.resize` on
  the frame sizes the datasets have, or does it stay with OpenCV? If OpenCV
  leaves the evaluator, `RECORDED_LIBRARIES` in `evalkit/code_sha.py` drops
  `cv2` with it. Blocks: the freeze. Answer in: `evalkit/inputs.py`.
- **Conditions seeded from GT.** The workbench seeded `t12_gtseed` and
  `t12_paste` from GT masks. On the 151 ATLAS-120k clips with an even
  number of frames, the seed sits one frame before the centre:
  the workbench's `track_metrics.pick_seed_frame` breaks a tie toward the
  earlier frame, while the tracking stage and `rule_of_seed_info` take
  `frames[n // 2]`. So on those clips the conditions hold neither
  propagation rule. The
  workbench's `normal_edge_t12_gtseed` and `rgb_edge_t12_gtseed` records
  hold no edge ring for the tracker's input, so `condition_inventory` and
  `check_provenance` refuse them until the conditions are measured again.
  Decided on 2026-10-12: the seed stays on `frames[n // 2]` under
  `both_ways_from_centre` and on frame 0 under `forward_from_first`, for a
  GT seed and a SAM seed alike, and the population is chosen so that both
  frames carry GT (`docs/pipeline.md`, "A clip"). CholecSeg8k's windows
  moved for it (`cholecseg8k_meta/README.md`); ATLAS-120k's clips carry GT
  on every frame. Open: which code places a GT seed, and how the evaluator
  scores it. Blocks: measuring the `t12_*` conditions again, with the
  evaluator and with `track_metrics`. Answer in:
  `docs/evaluation.md` and `docs/pipeline.md`.
- **Cuts inside GT clips.** Every candidate whose spot differs by more
  than 30 inside the population is judged, and 39 clips hold a cut
  (`atlas120k_meta/cuts.csv`). After a cut, a tracking condition's scores
  measure the scene change, not the input. Decided on 2026-10-12: split
  each clip at its cuts. The extraction rule ends a run of consecutive
  frames at a cut as well, and keeps the runs that still have at least 8
  frames after thinning; replayed in the workbench, the 315 clips become
  339. Open: the
  extraction stage reads `cuts.csv`; the population is extracted again; and
  the clips that replace the 39 get depth, camera poses and every tracking
  condition. Not resolved, and not blocking: most candidates between 20
  and 30 were not judged, and two of those judged are cuts (the
  README says so). Blocks: extracting the population again, and the freeze.
  Answer in: `atlas120k_meta/README.md`.

### Conditions

- **Depth made by two versions of the depth stage.** On the development
  machine, 7 of the 9 CholecSeg8k clips carry depth from a branch of the
  workbench's depth stage that never reached its `main`: their bundle holds
  a `ray_map`, and DA3 ran with `ref_view_strategy="middle"`. The port
  follows `main`, which leaves DA3 at its default. So the byte check
  reproduces `main`, not necessarily the depth a scored condition read.
  Open: which version made the depth each scored ATLAS-120k condition
  read, and where it was the branch, whether that depth is made again.
  Answer in: `docs/pipeline.md`, the depth stage.

### Tools and tables

- **The paired table on the evaluator's scores.** `paired_table` prints the
  pilot's table only. `paired_table` holds one propagation rule per table,
  so it is printed once per rule. The evaluator records `objects.predicted` in place of
  `n_regions_mean`. Open: which block answers which of the paper's three
  questions ("Primary metrics" in `docs/evaluation.md`), and which rows,
  keys and views each table holds. Answer in: `paired_table`'s docstring.
- **The claims table on the evaluator's scores.** `claims_table` prints the
  workbench's table from the pilot's scores. The workbench
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

### Pipeline

- **Masks that are not GT under the GT's name.** The viewer's step writes
  SAM 3 masks into `seg_masks/` under the annotation's name,
  `<i>_color_mask.png`. Only the frame manifest's `is_anchor` and
  `seg_provenance` tell them apart. The evaluator reads those flags, and
  refuses two VID25 clips that hold such masks on frames the manifest marks
  as having none. Open: whether such a mask moves out of `seg_masks/` or
  takes a name of its own. Blocks: porting the step that writes them.
  Answer in: `docs/pipeline.md`.

### D4D

- **The edge ring of the D4D seeds.** The workbench's front/behind scores
  read seeds cut with the edge ring kept. `pipeline.d4d_seed` writes their
  bytes with `--keep-edge-ring` only; by default it masks the ring, as the
  tracking stage does. On the six sides compared, every seed changes.
  Open: which setting the D4D result is measured with. Blocks: the D4D
  result. Answer in: `docs/pipeline.md`, the D4D depth and seed stages.
- **The D4D verdict and the mark rule.** `d4d_front_behind_verdict`
  gives `A` only when the intervals over sessions, clips and specimens all
  exclude zero, from 2,000 resamples of the pooled share. `paired_stats`
  marks a pair when the interval over videos excludes zero on six videos
  or more, from 10,000 resamples of the mean difference. D4D has five
  specimens, so an interval that the mark rule reads for its sign only
  can turn the verdict to `B`. Open: whether the D4D claim is marked by
  `paired_stats.mark_of` with sessions as the videos, and with which
  number of resamples. Blocks: the D4D result. Answer in:
  `d4d_front_behind_verdict`'s docstring.
- **The size of a D4D region.** `d4d_front_behind` scores a region of
  `MIN_NODE_PIX`, 400 pixels, at the endoscope's resolution, 640 × 512.
  Such a region holds 250 pixels at the depth's resolution, 504 × 406,
  where the tracking stage drops a seed region under 400 pixels. So the
  relation scores regions that the tracking stage would not carry. Open:
  whether the threshold is taken at the depth's resolution. Blocks: the
  D4D result. Answer in: `d4d_front_behind`'s docstring.
- **A pair the floor ties on.** `d4d_front_behind` leaves out a pair
  whose answers tie, and `area` ties when two regions hold the same number
  of pixels. So `area` can leave out a pair that `mono` keeps, and
  `d4d_front_behind_verdict` then refuses the two outputs. On the 532
  sides, with the edge ring masked, this happens on one side: 17,955 pairs
  against 17,956. Open: whether the verdict compares the pairs both
  outputs hold, or `area` decides a tie. Blocks: the D4D result with the
  edge ring masked. Answer in: `d4d_front_behind_verdict`'s docstring.

### LapEx

- **How LapEx is compared.** The workbench resampled 30 case means, which
  weighs the cases unlike `paired_stats`. Open: whether LapEx's comparisons
  resample as `paired_stats` does. Blocks: LapEx's comparisons. Answer in:
  `docs/evaluation.md`.

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
