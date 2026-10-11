# Porting plan

**Status: in progress.** This is the plan for moving the code out of the
private research workbench (`$OMNISURG_SOURCE`) into this repository. It is
deleted once the port is done; git history keeps it.

## Progress

A check compares the port with the workbench on the same input; "—" means
no check is recorded.

| area | ported | checked |
|---|---|---|
| evaluator, `evalkit/` | class tables, metrics, entry point, pilot mode | in pilot mode, every key shared with the pilot evaluator equal on 23 conditions × 315 clips (step 2) |
| score tools, `evalkit/tools/` | every tool in the layout of `AGENTS.md` but `track_metrics` | `paired_table`, `arms_paired`, `claims_table`, `leave_one_video_out`, `print_status_tables`, `kmerge`, `pose_metrics`, `pose_controls`: the workbench's values on the same inputs |
| extraction and crop | ATLAS-120k, CholecSeg8k, LapEx, StereoMIS, the D4D population | the CholecSeg8k crop, 13 of the 27 CholecSeg8k clips and the D4D population: the workbench's bytes |
| depth | DA3, Pi3X, point clouds, the population drivers | — |
| segmentation and tracking | `track`, `per_frame`, `condition_population` | `track` and `per_frame`: the workbench's bytes on the GPU machine, on three clips. `condition_population`: the workbench's commands, for 43 conditions. The 40 segmenter inputs that `tests/test_render_outputs_are_pinned.py` pins: the workbench's, rendered with each case's ring setting |
| viewer | `viewer/`, `pipeline/viewer_bundle` | — (new: the workbench's demo read other files) |

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
  GT track of `track_metrics`".
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

## Decisions that change the original plan

1. **The paper is measured here.** Every IPCAI number comes from the evaluator
   specified in `docs/evaluation.md`, run from this repository. The original
   fallback to the workbench's numbers, and the date it was tied to, are
   withdrawn. The workbench's schedule does not bind this plan.
2. **Nothing is frozen during the port.** The evaluator is built, checked and
   used without a freeze. Every score still carries `eval_code_sha`, and
   `compare_eval` still refuses to mix shas, so a change to the evaluator
   is visible in every number it touches; it is not forbidden. The freeze
   in `AGENTS.md` comes once, later, before the numbers the paper reports
   are measured.
3. **The pilot evaluator is not copied here.** It stays frozen in the
   workbench, `eval_code_sha = 1f8a813a…`. The check against it runs there,
   taking paths as arguments. `docs/evaluation.md` describes the check, under
   "Checked against the pilot evaluator".
   - So the constraints the frozen files imposed no longer bind. Those were: the
     import paths `surgical_core.cholec` and `surgical_core.atlas`, `evalkit/`
     as a script directory at the repository root, `sam3_wrapper/` on
     `PYTHONPATH` by name, and a copy of the older `eval_track.py` as a test
     fixture.
4. **`evalkit/` is a package.** Tools import the evaluator as
   `evalkit.<module>` and each other as `evalkit.tools.<module>`, with no
   `sys.path` edits. `check_env97` existed because frozen scripts could not
   check their own import path; a package does not need it. `pyproject.toml`
   includes `evalkit*` next to `surgical_core*`.
5. **No minimum object size, anywhere.** The evaluator has none
   (`docs/evaluation.md`, "Thresholds"). `track_metrics.MIN_AREA` (400 px)
   is removed with it, and nothing filters GT objects or predicted regions by
   pixel count. The one place a cut survives is pilot mode, which needs the
   pilot evaluator's 300 px cut to reproduce its numbers, and nothing else
   reads it.
6. **AE-CAI stays in the workbench.** Its numbers are reproduced by the tag
   `paper/aecai2026-endolina-final` and the code at that tag. Nothing here
   reproduces them, and no tag of that name is created here.
7. **No version labels.** The earlier evaluator is *the pilot evaluator*, and
   the new one is *the evaluator*.
8. **Names.** The project is `omnisurg-4d-scene-graph`, and the ATLAS-120k
   module is `atlas120k`.
9. **English only.** Workbench code comes with Japanese comments. Each file is
   rewritten in English when it is ported, not afterwards. This holds for the
   reviewed branches too: their `surgical_core` modules were translated in
   part, the tools not at all.
10. **No personal email address.** `tests/test_no_personal_email.py` checks the
    history and the files. In every new clone, set the repository-local
    `user.email` to the GitHub no-reply address before the first commit. The
    global config on a development machine holds a personal one.
11. **Pilot mode is removed before the freeze.** Pilot mode stays in the
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
12. **Two propagation rules, and no seed chosen from GT.** Every tracked
    condition the paper reports is measured under the two propagation rules
    `docs/evaluation.md` defines: `forward_from_first`, the causal setting,
    and `both_ways_from_centre`, the offline setting every pilot condition
    used. The ported tracking stage carries no choice of seed frame from GT.
    So `op_normal` and `op_edge` on ATLAS-120k and `ch_normal` on
    CholecSeg8k, the three scored conditions the workbench's audit found
    seeding by `--seed_auto` at its default, which reads the GT masks, are
    dropped: their centred versions exist. The workbench's ATLAS-120k
    pipeline script runs that default too, and its port does not. The
    tracking stage records its rule with its output, the evaluator writes it
    into every score, and the tools refuse a comparison or a table that
    mixes two rules, as they refuse two shas. A condition with no tracker
    has no rule and may sit beside either, since propagation against
    per-frame segmentation is itself a comparison the paper makes.
13. **The camera trajectory measures are hashed on their own.** `ate_rel`
    decides the StereoMIS result, so the code that computes it is under the
    freeze rule like the evaluator, and `eval_code_sha` does not cover it:
    the two results are measured and frozen at their own times, and a
    change to one must not make the other's scores incomparable. When the
    StereoMIS scorer is ported, `pose_metrics` moves out of `tools/` into a
    package of its own, hashed as `code_sha` hashes the evaluator (each
    file's path and content, each preceded by its length), and every
    StereoMIS score records that hash and is compared only with scores
    whose hash matches. Until then `pose_metrics` is a module under
    `tools/`, tested, and nothing scores with it.
14. **No measure over time carries a star.** `time_IoU` stays a reference
    value, and the evaluator computes no measure of identity from the
    workbench's `track_metrics` (hold, IDF1, ID switches, fragmentation,
    re-entry). `track_metrics` is ported as a tool under `evalkit/tools/`,
    outside `eval_code_sha`, and what it reports is a reference value: it
    is measured and reported in a table that carries no star. So
    `docs/evaluation.md` defines no GT track. The tool defines its own when
    it is ported, without `MIN_AREA` ("No minimum object size, anywhere");
    that track is an open question ("The GT track of `track_metrics`"). The
    workbench pre-registered two definitions under the name `hold_mean`: one
    per condition, in `track_metrics`, and one over the GT tracks that every
    compared condition picked a region for on the seed frame, in
    `summarize_track16`. The tool carries at most one of them under that
    name, and the other under a name of its own, because a different
    denominator is a different metric: in the workbench, moving the
    denominator once flipped a result's sign. Which one keeps the name is
    settled with the tool's GT track.
15. **`kmerge` is a tool.** `kmerge` is ported under `evalkit/tools/`,
    outside `eval_code_sha`. It changes predictions, not metrics: it merges
    a condition's regions down to K and writes them as a condition of their
    own, which the evaluator scores and `compare_eval` compares. So it
    computes no metric, and a comparison of a merged condition can carry a
    star like any other. The paper's granularity result merges to a fixed K
    of 10. The `matched` setting, which took K from the pilot's GT
    components of at least 300 px, is not carried for now, nor are the
    `kgt` seeds, which took the clip's median of that count. Under the
    evaluator a GT object is a whole class, and that K would need a
    definition of its own.
16. **CholecSeg8k clips are cropped as AE-CAI's were.** Every CholecSeg8k
    clip is cropped to its rectangle in `cholecseg8k_meta/crop_rects.json`.
    The table holds the rectangles of the workbench's clips, which the pilot
    scored and AE-CAI reports. The original plan took the rectangle of the
    circle fitted to the cholec80 video instead. `run_all17_pipeline.sh`
    ran `crop_cholec_frames.py --method circle` on the GPU machine's host
    without `--source-root`. The script found no video there, and took each
    rectangle from the clip's own frames, as `paper_clean_rerun_spec.md`
    asks of every clip. The circle's rectangles differ from these on all 27
    clips, by up to 59 px. On VID26 and VID43 they hold black corners of 2
    to 3.6 % of the crop. Run on the workbench's 27 clips with the table,
    the crop stage writes the workbench's images, masks and
    `crop_info.json` byte for byte, and every manifest key the stage writes
    holds the workbench's value. On the clips the extraction stage writes,
    the crops of 13 clips equal the workbench's byte for byte. The crops of
    the other 14 differ in the gap frames alone, which the extraction stage
    converts ("CholecSeg8k clips whose frames run out of order"). A rerun
    of the tag `paper/aecai2026-endolina-final` reproduces these crops only
    where the script finds no video. Where
    `/work/SourceDatasets/cholec80/cholec80/videos` holds the videos, the
    script fits the circle instead. AE-CAI keeps its clips. They stay in
    the workbench's `outputs/cholec_gt`, which nothing here writes to. Each
    of them holds depth, so the crop stage refuses to replace one even with
    `--overwrite`.
17. **What the paper measures.** Decided on 2026-10-10:
    - **The `forward_from_first` conditions.** The paper reports them. They
      are measured on the GPU machine before submission.
    - **The conditions seeded from GT masks.** They are measured again with
      this repository's code. Open: whether the paper reports them.
    - **`track_metrics`.** It is run on the paper's conditions. Open: how
      the paper reports its measures.
18. **The public viewer reads a bundle.** The workbench's viewer read the
    export stage's files from its development server, which listed the
    clips through `/api/clips`, and drew a 10 MB colour raster per frame and
    track. A page on the public internet has no such server. The stage
    `pipeline.viewer_bundle` writes what the viewer reads instead: a catalog
    of the clips, and per clip the frames, the clouds of one geometry
    source, and per track the labels of the cloud's points, run-length
    coded, with the scene graphs. A clip's overlays go from about 600 MB to
    under 1 MB. The workbench's viewer resampled the overlays onto the Pi3X
    cloud and placed the nodes on it in the browser, on every frame load;
    the stage does both once, so the labels the viewer reads always index
    the cloud it draws. Track names and instrument classes come from the
    bundle; the instrument classes are the class tables' `tool` type. The
    viewer names no track and no dataset.

## Order, and what must hold before the next step

1. **Decide what the evaluator measures over time.** Decided: no measure
   over time carries a star, and `track_metrics` and `kmerge` are tools
   ("No measure over time carries a star", "`kmerge` is a tool").
2. **Build the evaluator.** CPU only, as the `evalkit` package. Class tables,
   then metrics, then the entry point. It is done when:
   - the hand-derived tests pass, in both modes;
   - in pilot mode, the evaluator reproduces every key it shares with the
     pilot evaluator, at zero tolerance, on the 38 scored conditions, apart
     from those that hold neither propagation rule.
   Record its sha with every score; do not freeze it (decision 2). Pilot
   mode stays in it until just before the freeze (decision 11).
   The check against the pilot evaluator passed on 2026-10-11, for
   `eval_code_sha` `a5d7ea46…`. It compared the 23 conditions that hold a
   propagation rule, 315 clips each: per clip 18 values and 17 numbers of
   frames (the pilot evaluator writes none for `time_IoU`), 253,575 in
   all, every one equal. It left out `op_edge`, `op_normal`, the seven
   `t12_*` and the six `t5_*`. The check runs again after any change to a
   hashed file.
3. **Re-score.** CPU only. Score every condition's existing predictions with
   the evaluator. `condition_inventory` must report no mixed ruler and no
   missing condition.
4. **Port the toolkit onto it.** Each tool is done when its tests pass, and:
   - on the pilot's score JSONs, it writes the same bytes as the workbench
     version (the bootstrap is seeded), apart from the departures its row
     names. `paired_table` translates its labels and is compared value by
     value. The pilot's JSONs lack the fields the evaluator now writes
     (class set, view, mode, input hashes, versions, propagation rule); a
     tool reads them all the same, taking the missing fields as the pilot
     evaluator's, and raises when a JSON has some of the fields but not all.
     Dropping this check would let a tool lose behaviour the workbench
     version had without anyone noticing;
   - what only the evaluator's JSONs carry is checked by a self-test that
     plants the fault: `compare_eval` refuses a mix of shas, and a mix of
     modes or class sets under one sha; `condition_inventory` reports a
     planted mixed ruler and a planted missing condition.
   The paper's numbers are the step-3 scores read through these tools.
5. **Port the pipeline, one stage at a time.** Each stage's output must match
   the workbench byte for byte. The exceptions:
   - Pi3X's `runtime_sec`, per `repo_migration_determinism.md`.
   - The 14 CholecSeg8k clips whose gap frames the workbench's extractor
     placed 1 to 3 s late ("CholecSeg8k clips whose frames run out of
     order"), which the ported extractor converts, and which are then
     re-extracted and re-run.
   - In the ATLAS-120k extraction, a frame that the stage shrinks. OpenCV's
     area resize rounds differently from one build to another: between the
     development machine and the GPU machine's clips, about one pixel in ten
     thousand differs by one level, and a frame written at its own size
     does not differ at all. So the frames are compared on one machine, as
     every stage is.
   - In the ATLAS-120k extraction, the manifest key `pixel_source`. The
     workbench's extractor gained it after the GPU machine's clips were
     extracted, so those manifests lack it and the ported stage writes it.
   The 315 clips also get DA3 and `glb_centroid`: the demo's reference grid
   stays DA3.
   This step compares files, not scores, so it does not wait on steps 1 to
   4; it may run beside them. The port is reviewed anywhere. The DA3 stage
   is checked first on the development machine's CPU with the real model;
   every stage is checked on the machine the workbench's stages run on, on
   its GPU. Each time both sides run in one environment, so that a
   difference is the code's. If the GPU machine's packages no longer equal
   `environment.packages` in the determinism measurement's JSON, the
   workbench's stage runs twice first, to measure determinism again.
   Predictions made again from output that is not the workbench's, the
   exceptions above, are scored as in step 3, after the evaluator's check
   against the pilot evaluator. The conditions under `forward_from_first`
   ("Two propagation rules, and no seed chosen from GT") are new
   measurement, not a port: every condition the paper reports ran through
   `run_track_conditions.py`, whose modes all pass `--bidir`. (The
   workshop's forward runs, `run_w2_main.sh` and `run_multi_tracking.sh`,
   are not the paper's conditions.) They are made on the GPU machine once
   the tracking stage is ported, and scored as in step 3.
6. **Prepare the release.**
   - An English README, `docs/data_contract.md`, the `atlas120k_meta/` README
     and `CITATION.cff`.
   - Before anything is public, a check of the anonymity rules of Healthcare
     Technology Letters, since AE-CAI is under double-blind revision.
   - Done when a third party can clone, install and get a green `pytest`,
     and, on a machine with a CUDA GPU, install the pipeline with the
     README's steps and run each ported stage on a clip.

## Open questions

1. **The GT track of `track_metrics`.** The tool measures identity against
   GT tracks, and the datasets carry no ids for individual things, so the
   tool defines the track itself ("No measure over time carries a star").
   The workbench linked each class's connected components over time: a
   component joins a track of the same class when its IoU with the track's
   last mask is at least 0.3, a track stays a candidate for three missed
   observations, and the links are made greedily, largest IoU first. Open
   before the tool is ported:
   - whether a track is one class's whole-class object followed over time,
     which needs no linking rule, or linked components, which need one;
   - if components, the linking rule: the IoU threshold, how many missed
     observations a track survives, and which pixels make an observation
     now that no size cut applies;
   - whether a gap in a track is counted in observations, as the
     workbench's linking rule and re-entry counted it, or in seconds, since
     the GT frames are not evenly spaced; hold's time offsets were in
     seconds already;
   - which frame hold starts from when the seed frame has no GT. The
     workbench's `track_metrics` refused such a clip, and
     `summarize_track16` started from the GT frame nearest in time, up to
     5.6 s away; that happened on 14 of the 27 CholecSeg8k clips. A
     condition segmented frame by frame has no seed frame, and
     `summarize_track16` gave it the seed frame of a tracked condition;
   - which of the two pre-registered `hold_mean` definitions keeps the name
     ("No measure over time carries a star");
   - what hold means under forward propagation (`forward_from_first`),
     where its time offsets from the seed frame run one way only; the
     workbench claimed this changes the measure and did not measure it.

   The tool is run on the paper's conditions ("What the paper measures").
2. **The skill-classification code and the other 18 viewer pages.** The plan
   leaves both behind. The paper's figures come from some of those pages.
3. **Which numbers step 5 changes before submission.** Step 3 scores every
   condition the workbench ran. Step 5 measures the `forward_from_first`
   conditions for the first time. The paper reports them ("What the paper
   measures"), so step 5 runs before submission.

   A stage that passes step 5's byte check leaves the step-3 numbers as
   they are. Two things that step 5 makes again can change them:
   - the 14 CholecSeg8k clips extracted again ("CholecSeg8k clips whose
     frames run out of order");
   - depth made again ("Depth made by two versions of the depth stage").

   Open: does the paper report those clips and that depth as step 5 makes
   them again, or as step 3 scores them?
4. **Boundary dilation before the freeze.** `evalkit/boundary.py` dilates
   with `cv2.dilate`; a numpy shift-or over the (2·tol + 1)² offsets agrees
   on every mask tried, borders included. The question is whether a hashed
   file should depend on a library's behaviour at all while OpenCV is
   unpinned. Decide before the evaluator is frozen. Pilot mode depends on
   OpenCV more deeply: `evalkit/pilot.py` numbers a class's components in
   the order `cv2.connectedComponents` labels them, and the pairing's tie
   rule reads those numbers. That never reaches the frozen files, since
   pilot mode is removed before the freeze (decision 11).
5. **The benchmark mapping against its source.** The ATLAS-120k mapping to
   the benchmark's 30 classes was typed from the document and checked by
   hand against ATLAS-bench's `datasets/class_mapping.py` at commit
   e286a584, all 47 ids agreeing. A script that takes that file's path and
   repeats the check would make it reproducible.
6. **CholecSeg8k's Region line.** The white line between regions is 1 px
   wide; in the masks 87 % of its pixels are the image's outer 1 px (gone
   with the crop) and the rest sits mostly in video43 and video52. Left
   `ignored`, an edge against it is no boundary, so those videos lose much of
   their GT boundary, and unevenly: after the nearest-neighbour resize the
   line survives only in places. Decide whether the loader fills the line
   from its neighbours, at full resolution, by a deterministic rule with the
   filled count recorded, or whether it stays ignored with the loss
   documented. Either way the pilot evaluator read it as background and
   counted an edge against it as a boundary, a normal-mode difference to list.
7. **CholecSeg8k clips whose frames run out of order.** In videos where
   CholecSeg8k numbers frames at about 30 fps, the workbench's extractor
   (`scripts/extract_cholec_track.py`) resolves the true 25 fps frame for the
   frames that carry a mask and leaves the gap frames it decodes from the
   video at the unconverted number, 1 to 3 s later in the video than their
   place in the clip. The manifest's times follow the clip; the images do
   not, and one image can appear twice. 14 of the 27 pilot clips are
   affected, 11 visibly. Removing them changes the mark of several F1 and
   IDF1 comparisons (power, not sign) and none of the `boundary_F` or
   `hold_mean` ones. The ported extractor (`pipeline/extract_cholecseg8k.py`)
   converts the gap frames: it interpolates their native frames between the
   clip's annotated frames, and carries the rate past them. Decided: the 14
   clips are re-extracted with it and re-run on the workbench's GPU machine
   for the paper's numbers, a deliberate exception to step 5's byte-for-byte
   rule. The ported extractor writes the other 13 clips as the workbench's
   did, byte for byte.
   The evaluator refuses two of the nine clips on this machine. In
   `VID25_s15_162_crop` native frame 387 appears twice, at one timestamp.
   In it and in `VID25_s15_402_crop`, the gap frames hold masks the
   viewer's step wrote before the clips were extracted again, 504 × 504
   where the GT is 457 × 456, on frames the manifest says have none: 8 and
   19 frames. The pilot's scores of these clips count the annotated frames
   only, so its data root did not hold those masks. The question closes
   when the re-run is scored.
8. **The evaluator map against `evalkit/frame.py`.** Two things to carry
   into the next redraw of `docs/figures/evaluator_map.png`, neither wrong
   today. The map gives step 2, one frame in one view, no module, and
   names `frame` at step 3 only; in the code both are in `frame.py`, as
   `score_view` and `score_frame`, so a reader looking for where one view
   is composed finds no box. And the map's step 3 shows a frame scored or
   its keys undefined, never skipped: the excluded marker takes a frame out
   whole, counted for the clip driver, and a depth map with an invalid
   pixel stops the run with an error, counted nowhere. A phrase in the
   step 3 box, "or skipped whole, and counted", would close that in the
   figure. `evalkit/README.md` is the
   short version and need not say either.
9. **The fewest videos for an interval.** `paired_stats.boot_ci` refuses a
    population of one video, where every resample is the same video and
    the interval is a point. Two is the floor that removes that failure,
    not a statistical one: with n videos the chance that a resample draws
    one video n times is n^-n, above 2.5 % up to three videos, so on two
    or three the 95 % interval is the range of the video means, and two
    videos that agree in sign give a mark. `wilcoxon_video` draws its own
    line at six for the same reason. Whether the interval gets a floor
    above two, and where, is a decision about the paper's populations, not
    the code's; until it is made, a subset's interval is read for its sign
    only, as `--drop-video` says. `paired_table`'s floor of five clips per
    cell is decided with this one.
    `leave_one_video_out` keeps the workbench's two lines, which are
    decided with this one. It leaves a video out only while three videos
    remain, so on four videos each interval with a video left out is the
    range of three video means; the printout says so. It prints an
    interval on four videos as thin.
10. **The edge ring of the tracker's input.** The tracking stage builds two
    inputs with `sam_input_image`: the seed frame's, in the `--sam_input`
    mode, and the tracker's, in the `--track_base` mode, which may be
    `normal_edge` or `rgb_edge`. Both inputs are made with one ring setting.
    The stage's `seed_info.json` records the setting once, as
    `seed_input.edge_ring_masked`, and that field describes the seed's input
    only. When the seed came from GT or from outside the stage, `seed_input`
    or the field is `None`. When the `--sam_input` mode burns no edges, the
    field is `None`. In these cases the record holds no setting, even when
    the tracker's input burns edges. In the workbench's
    `ipcai2027_experiment/out/track12/` and
    `ipcai2027_experiment/atlas97/out/track12/`, 750 label directories of
    `t12_gtseed`, one per clip and input, were tracked on `normal_edge` or
    `rgb_edge`, and none of their records holds the setting of the tracker's
    input. They were made on 2026-09-17, after the workbench set
    `EDGE_MASK_RING` to True, and no script of the workbench sets it
    otherwise, so the ring was most likely zeroed. The ported stage takes
    the setting as an argument, `--keep-edge-ring`, and writes the
    workbench's record, so its record holds the setting of the tracker's
    input in the same cases only, and neither `check_provenance` nor
    `condition_inventory` can compare it. Decide where the record holds that
    setting, and, if the `t12_gtseed` conditions enter the paper
    ("Conditions seeded from GT"), what their records are taken to say.
11. **Depth made by two versions of the depth stage.** On the development
    machine's copy of the workbench, 7 of the 9 CholecSeg8k clips (VID01 and
    VID12) carry depth written by a branch of the depth stage that never
    reached the workbench's `main`: their `results.npz` holds a `ray_map` and
    their manifests `backproject_mode: "ray"`. That branch passed DA3
    `ref_view_strategy="middle"`; the stage on `main`, which step 5 ports,
    passes nothing, and DA3's default is `saddle_balanced`. The reference view
    sets the frame the poses are given in, so the two need not give the same
    depth or poses. The 2 VID25 clips, extracted again later, carry neither,
    as `main`'s stage writes them. So a stage that passes the byte check
    reproduces `main`'s stage, not necessarily the depth a prediction read,
    and a clip run again (the 14 under "CholecSeg8k clips whose frames run
    out of order" among them) may get depth unlike its neighbours'. To settle
    on G: which version made the depth each scored condition read, on
    ATLAS-120k too. A `ray_map` in `results.npz` marks the branch; for a clip
    without one, a run of `main`'s stage compared with the stored files says
    whether `main`'s stage, in G's environment, makes them again. Then
    whether the ported stage follows `main`, and the depth so made is made
    again with every condition on it, or the branch's setting becomes the
    stage's.
12. **Two orders of the world transform.** `cam_to_world` computes
    `(p - t) @ R`; the workbench's back-projection computes
    `(R.T @ (p.T - t)).T`. On the development machine the two give the same
    bits only where the BLAS runs the same kernel: under numpy 2.5.3 on
    Accelerate they differ in the last bit below about 1024 points (at the
    tests' 9×11 frame, 20 of 288 elements) and agree above; under numpy
    1.26.4 on OpenBLAS they agree at every size tried. The depth stages'
    clouds are far above the line, so step 5's byte check on the
    workbench's GPU machine does not answer it. `project.backproject`
    computes in the same order, through `cam_to_world`, but no stage calls
    `project.backproject`. In the workbench, the view-consistency analysis
    (`view_consistency.py`, `camera_motion.py`, `export_03b_figs.py`),
    `track_sam3_3d.py` and `legacy/pipeline.py` call it, none of them on
    the paper's path, and each passes it a whole frame, far above the line
    too. So no measured number shows the difference; only a frame as small
    as the tests' does. If one of those callers is ported and its check
    finds a difference, the choice is between restoring the workbench's
    order and accepting a documented non-bit-equality — made then, not
    found later.
13. **Whether the geometry path has to be fast.** The depth stage
    back-projects each frame once when a clip is exported, with
    `backproject_depth`, `cam_to_world` and `world_to_gltf`, and writes the
    points to a GLB file. Nothing waits on it there, so its cost is a batch
    cost today. The viewer reads the GLB files the export wrote and
    recomputes nothing. A viewer that recomputed geometry while the user
    moves would need the path faster than it is. The three functions
    together take 40 to 65 ms on a 1080p frame across runs on one machine,
    numpy only. The shape checks are not the cost. Each compares one tuple
    per array, and the four on the path through `backproject` take 0.4 µs
    together. The per-pixel work is the cost, and the world transform
    carries most of it. The points come out float64 whatever the dtype of
    the depth map and the intrinsics. The pixel grid is built with an
    integer `arange`, and numpy promotes int64 with float32 to float64. One
    such array of points holds about 50 MB at 1080p. On float32 points the
    world transform measured 2.5 times faster. `backproject_depth` and
    `backproject` rebuild the pixel grid on every call, although a clip's
    resolution is fixed. That rebuild is a few per cent of the time. None
    of this is worth changing while the port lasts. float32 changes the
    output, and the pipeline port asks each stage to match the workbench
    byte for byte. The byte check reads the GLB files too. Decide once the
    stages match, and decide with it whether a viewer ever recomputes
    geometry or only reads what the export wrote. The functions are in
    `surgical_core/geometry/camera.py` and
    `surgical_core/geometry/project.py`.
14. **Conditions seeded from GT.** A seed frame is scored like any other
    frame, because the paper's conditions are seeded from the pipeline's own
    masks. Which of the 38 conditions were seeded from GT instead, and
    whether such a condition is scored on its seed frame or enters a table
    at all, is settled before step 3, on the machine that holds the
    predictions. The check against the pilot evaluator in step 2 left out
    by name every condition that holds neither propagation rule
    (`pilot_check --leave-out`). The `seed_source` that each
    condition's `seed_info.json` records says where its seed came from;
    where it does not tell, the command that made the condition does.

    The conditions that took something from GT:
    - `t12_gtseed`, on five inputs, and `t12_paste`: the workbench seeded
      them from GT masks (`--seed_source gt`);
    - the workshop's oracle row: GT instrument masks painted onto a
      condition's labels;
    - the viewer's `gt_tracked` track (a candidate): one GT frame carried
      by SAM 3.

    The ported tracking stage seeds from no GT.

    Where the `t12_*` seed sits depends on the clip's number of frames.
    Every frame of the 315 ATLAS-120k clips has GT.
    - On the 164 clips with an odd number, it sits on the centre frame.
    - On the 151 clips with an even number, it sits one frame before the
      centre.

    Two pieces of code take the centre differently:
    - `track_metrics.pick_seed_frame`, which placed the seed, takes the GT
      frame nearest (n − 1)/2 and breaks a tie toward the earlier frame;
    - the tracking stage and `rule_of_seed_info` take `frames[n // 2]`.

    So on those 151 clips the `t12_*` conditions hold neither propagation
    rule, whatever seed source the evaluator accepts.

    They are measured again with this repository's code ("What the paper
    measures"). Open before that:
    - which frame a GT seed is placed on under each propagation rule;
    - which frame it is placed on when a clip's centre frame has no GT.
      This happens on CholecSeg8k, whose GT covers only some of a clip's
      frames;
    - which code places a GT seed;
    - how the evaluator records and scores a condition seeded from GT
      ("Seeds made outside the tracking stage").
15. **Masks that are not GT under the GT's name.** The viewer's step writes
    SAM 3 masks into `seg_masks/` as `<i>_color_mask.png`, told apart from
    the annotation only by the frame manifest's `is_anchor` and
    `seg_provenance`, and the two VID25 clips still hold such masks from
    before their re-extraction, on frames the frame manifest marks as having
    none. The evaluator reads the flags and refuses the VID25 clips until
    those files are removed. When the pipeline is ported (step 5), decide
    whether a mask that is not annotation moves out of `seg_masks/` or takes
    a name of its own, so that the distinction is in the file and not only in
    the frame manifest.
16. **Which commit of Depth Anything 3 the `recon3d` extra pins.** The extra
    names the repository at its head, so two installs can get two versions.
    The commit to pin is the one the workbench ran on its GPU machine. pip
    recorded it there, in the `direct_url.json` of that install. The extra
    is pinned once that record has been read.
17. **A constraints file from the GPU machine.** The pipeline was measured
    with the package versions on the workbench's GPU machine. A constraints
    file lists them, so that `pip install -e ".[recon3d]" -c <file>` gives
    another machine the same versions. Once that machine's environment has
    been read, the file is written from it and added beside `pyproject.toml`.
    Until then the extra alone says what a machine needs.
18. **A clip with no usable depth in any frame.** The point-cloud stage
    skips a frame with no usable depth, as the workbench does, because the
    data can hold such a frame. A clip with no usable depth in any frame
    gets no cloud, no manifest entry and a count of 0, and the run exits 0.
    No real clip has done this; a bundle with no depth at all is more likely
    a broken bundle than data. Decide whether the stage refuses such a clip.
19. **Cuts inside GT clips.** The workbench searched for scene changes only
    inside GT clips, at frame pairs whose pixel difference was above 40,
    and judged 40 of those 125 pairs; the other 85 were not looked at, and
    pairs below 40 never were. Of the 315 clips,
    `cholecystectomy__1ud3syYKD3A__gt_0001` contains six judged cuts, all
    inside its kept run (between its frames 22/23, 28/29, 51/52, 56/57,
    69/70 and 76/77; the centre frame, where tracking from the centre is
    seeded, sits at one), and `cholecystectomy__Bj13QcLRCVc__gt_0001`
    contains one candidate left undecided. The workbench decided on
    2026-09-13 to use such a clip whole; the question then was the crop
    rectangle, not tracking. Every frame but the seed is scored on a mask
    that tracking propagated, so after a cut the per-frame scores of a
    tracking condition measure the scene change, not the input, and every
    measure over time is meaningless there. The comparison is paired by
    video, so every condition loses alike on that one video. Before the
    population is extracted again and the evaluator is frozen: judge the
    85 remaining pairs on the workbench's `cuts` review page
    (`experiment/crop_necessity/HANDOFF.md`, "5.2"), then decide whether a
    clip with a cut is used whole, excluded, or split at the cut.
    Excluding or splitting changes `clips.txt` and `depth_manifest.json`.
    Splitting also makes the cut marks
    (`experiment/crop_necessity/marks/marks_20260913_174731.jsonl`) a file
    a measurement reads, so they would return to `atlas120k_meta/`.
20. **The order of tied presence scores.** The tracking stage paints each
    frame's objects in the order of their presence scores, through
    `np.argsort`. On the prompted frame no object has a score, so all of
    them tie. `np.argsort` keeps the tracker's order among ties only up to
    16 objects, and a seed cut at 24 points per side often holds more.
    Above 16, the order depends on numpy's build and the CPU: on the
    development machine, under numpy 2.4.6 on arm64, 17 tied objects come
    out as 0, 14, 13, 12 and so on. The workbench sorted the same way, so
    the two stages give the same labels on one machine. Where the masks of
    tied objects overlap, a label of the prompted frame can differ between
    machines. The tracking stage's byte check on the workbench's GPU
    machine therefore runs both stages under one numpy on one CPU. Decide
    whether the stage keeps this sort, or sorts stably and lists the change
    among the differences from the workbench that the byte check allows.
21. **Seeds made outside the tracking stage.** The evaluator reads a
    condition's propagation rule from each clip's `seed_info.json`.
    `rule_of_seed_info` refuses a record whose `seed_source` is neither
    `sam` nor `per_frame`. It was run on the records of the 38 conditions
    the pilot evaluator scored on ATLAS-120k:

    | conditions | count | refused | why |
    |---|---|---|---|
    | tracked both ways from the centre frame | 5 | no | |
    | segmented frame by frame | 18 | no | |
    | `op_edge`, `op_normal` | 2 | yes | seed frame chosen from GT, off the centre on 298 of the 315 clips |
    | `t12_*` | 7 | yes | seeds from GT masks (`seed_source` `gt`) |
    | `t5_*`, the granularity result | 6 | yes | seeds made outside the tracking stage (`seed_source` `external`) |

    - The check against the pilot evaluator covers the 23 conditions not
      refused.
    - The seven `t12_*` scores read six prediction directories:
      `t12_gtseed` and `t12_rgb` both read `track_rgb_t12_gtseed`.
    - CholecSeg8k's conditions are not counted here.

    `make_t5_seeds.py` made the `t5_*` seeds:
    - from the per-frame `rgb` segmentation at 24 points per side, on the
      seed frame;
    - merged to K regions, in every arm but `t5_floor`, which it left
      unmerged. `t5_kgt` takes its K from the clip's GT, so its seed is not
      the pipeline's own;
    - on the centre frame of every clip, carried both ways.

    The evaluator still refuses all six. A seed made outside the tracking
    stage holds no propagation rule, whatever frame it sits on.

    The same refusal reaches this repository's own conditions.
    `pipeline/track.py --seed-labels` records `seed_source` `external`. So
    the evaluator refuses every condition the ported tracking stage makes
    from seeds made outside it.

    `docs/evaluation.md` says the paper reports none of the conditions that
    hold neither rule. The granularity result makes that sentence false.

    The record names the seeds by the path the tracking stage was given, as
    given. The workbench's records hold
    `../ipcai2027_experiment/atlas97/out/seeds/k10`, relative to where the
    stage ran. `check_provenance` and `condition_inventory` compare that
    path as written. So one directory given by two paths reads as two
    conditions, and one path given from two places can name two directories.

    Decide before the evaluator is frozen, and before the granularity result
    is measured again:
    - which values of `seed_source` the evaluator scores;
    - how a score records where its seed came from;
    - how the record names the seeds: by the path as given, by the resolved
      path, or by a hash of the seed files, as `kmerge.json` holds one of its
      source;
    - which comparisons the tools refuse between conditions whose seeds
      came from different places, as they refuse a comparison across two
      propagation rules.
22. **Condition names.** A workbench condition's name says which task made
    it, not what the condition is. `t12_*` comes from the workbench's task
    12 (`run_12_trackbase.sh`), and `t5_*` from step T5 of its task 05
    (`make_t5_seeds.py`). A condition also has two names: its score's tag
    (`t12_rgb`) and its prediction directory (`track_rgb_t12_gtseed`).
    A score records its `track_dir_name`, so a condition renamed after
    step 5 no longer matches its scores. Decide before step 5 writes new
    conditions:
    - how a name says what varies between conditions: the input, where the
      seed came from, and the propagation rule;
    - where the table lives that maps each workbench name to its name here.
      That table is the only place a workbench name appears;
    - whether `paired_table.PILOT_BLOCKS` and `claims_table.CLAIMS`, which
      read the pilot's score JSONs by their tags, keep those tags or take
      them from that table;
    - whether the comparisons the paper reports are listed in one place,
      read by every tool that prints them. An entry would hold the base,
      then the compared condition, and whether each condition is
      `per_frame`, as `claims_table.CLAIMS` does. `paired_table.PILOT_BLOCKS`
      puts the compared condition first, and `arms_paired` takes its pairs
      from the command line ("The pairs `arms_paired` reports").
23. **How LapEx is scored.** The evaluator holds class tables for
    ATLAS-120k and CholecSeg8k only, so it cannot read a LapEx clip.
    `kmerge` reads a condition through the evaluator, so it cannot merge
    one either. Three things are open before LapEx is scored again:
    - **Its class table.** Each of LapEx's 11 classes needs a type. LapEx
      labels every pixel, and level 0 is interstitial space, a class. The
      extraction writes level 0 as 11, because the workbench's evaluators
      read id 0 as background. Either the table names 11, or the move is
      dropped and the clips are extracted again. The workbench scored all
      11 classes. It planned two more views, one without the instruments
      and one without the instruments and the gauze (`compress`), and
      measured neither.
    - **Its depth.** LapEx's frames keep the black surround of the
      endoscope's view. On 710 of the 735 clips, the workbench's depth
      stage found that border and filled it (`border_inpaint` in
      `depth_info`). Its depth maps of those clips hold zeros, on 1.8 to
      8.9 % of the pixels, and the evaluator and `kmerge` refuse a frame
      with such a pixel. The ported depth stage neither finds nor fills a
      border, and it refuses a LapEx clip, which has no `crop_info.json`.
      So LapEx needs depth made again, on frames cropped or filled, and its
      predictions made again on that depth, by the ported per-frame stage,
      which records their propagation rule.
    - **Its comparison.** The workbench averaged each case's clips first,
      then resampled the 30 case means, 2,000 times. `paired_stats`
      averages over clips and resamples the cases, 10,000 times. A case
      holds 14 to 55 clips, so the two weigh the cases differently. The
      workbench's reading, fixed before the numbers, gave the merge a star
      only if it had one at both K = 8 and K = 10. The paper merges to 10
      ("`kmerge` is a tool").
24. **The paired table on the evaluator's scores.** `paired_table` prints
    the pilot's table. It does not yet print one from the evaluator's
    scores:
    - the six `t5_*` rows wait on "Seeds made outside the tracking stage";
    - the paper reports two propagation rules ("What the paper measures"),
      and a table holds one, so the table is printed once per rule;
    - the evaluator records `objects.predicted` in place of
      `n_regions_mean`, so the table would have no column of regions.

    Decide which block answers which of the paper's three questions
    ("Primary metrics"), which rows each rule's table holds, which keys and
    views it prints, and whether it prints `objects.predicted`. Until then
    the keys come from `--keys`.
25. **How this repository depends on D4D's loader.** The workbench imports
    `d4d.loader` from a clone of https://github.com/reubendocea/d4d at
    commit `efe12d26`, put on `PYTHONPATH`. pip cannot install the loader
    from that commit. The project is named `d4d-surgical`, and its package
    find rule includes only `d4d_surgical*`, while the package is `d4d/`, so
    the wheel pip builds holds no module. PyPI holds no `d4d-surgical`, and
    its `d4d` is an unrelated project. The repository has no licence file,
    though its `pyproject.toml` declares MIT. The ways open, none chosen:
    - upstream fixes the find rule and adds a licence file, and an extra
      pins that commit;
    - a fork carries the fix, and an extra pins the fork's commit;
    - no extra: the README says to put the clone on `PYTHONPATH`, and the
      D4D reader refuses to run without it.

    The census and the D4D reader wait on this. The population step needs
    no loader.
26. **Two inputs of the census whose code no repository holds.** The census
    reads `moved_camera` from `clip_table.json`, and `active` from the
    record of the tissue's motion, `tissue_motion.json`. Scripts in the
    dataset's directory wrote both files, and neither the workbench nor
    upstream tracks them. `detect_tissue_motion.py` passes
    `cv2.INTER_NEAREST` to `cv2.resize` as its third positional argument,
    which is `dst`, so it resizes bilinearly an instrument mask whose size is
    not the depth map's. In 11 of the 271 clips the masks are 894 × 714 and
    the frames 640 × 512. Whether the resizing changes any of the 45
    `tissue_moving` sides is not measured. `gt_blk_frac` is not affected.
    Decide whether the two scripts are ported or their files kept as data,
    and whether the motion record is made again with the masks resized by
    nearest neighbour.
27. **The tissue's motion when a side was scanned.** The record of the
    tissue's motion covers the clip only, so `active` is taken at the clip's
    first moment for `start` and at its last for `end`. The scans lie
    outside the clip: at the median, the `start` scan 4.4 s before the
    clip's first frame, and the `end` scan 4.3 s after its last. The record also stops
    sampling 0.5 to 1.7 s before a clip's last frame, so no `end` side is
    `active`, and all 45 `tissue_moving` sides are `start` sides. Whether the
    tissue moved when a side was scanned is measured for no side. Decide
    before the D4D numbers are reported whether the rule keeps this
    stand-in, written down beside it, or the motion is measured up to the
    time of each scan.
28. **The pairs `arms_paired` reports.** The workbench's script compared
    21 pairs of inputs by default, on the 13-video ATLAS-120k set and on
    CholecSeg8k. Two of the pairs compare inputs the paper's conditions
    ran: `rgb_edge` with `rgb`, and `normal_edge` with `normal`.
    `paired_table.PILOT_BLOCKS` compares `rgb_edge` with `rgb`, and never
    `normal_edge` with `normal`. The port carries no list of pairs, because
    the tags of the conditions scored again wait on "Condition names".
    Decide:
    - which pairs the paper reports with the tool;
    - whether the rows of `claims_grid.py` whose granularity is not
      matched read the evaluator's scores through this tool, or
      `pps_f1.py` is ported to write the F1 they read, and
      `export_prop_pps.py` with it.
29. **The times of a StereoMIS clip.** `surgical_core.stereomis.clips`
    takes a frame's time as its position over the declared frame rate. The
    P2 videos declare 59.94 frames a second and do not hold it: by their
    timestamps a clip's frames are 0.217 s apart at the median, not 0.2 s.
    `ate_rel` does not read the times. `rpe` and the constant-motion
    control read them. Decide before the StereoMIS result is measured
    whether the times come from the timestamps.
30. **P1's offsets.** The workbench measured P1's file offset (0) against
    the frames P1 ships, and took the shipped frame N to be position N. It
    is position N − 1. Measured against the video, P1's file offset is most
    likely −1, as P2's is. With −1, 504 of the 672 frames of P1's usable
    clips get a mask of their own number, where none does now. With −1,
    the workbench's rule for the common ground truth offset also picks −3
    rather than −4. Every clip then takes the frames one position earlier,
    is run through the depth stages again, and gets a new `ate_rel`. Decide
    both offsets before the StereoMIS result is measured, and whether the
    grid's start is fixed apart from them.
31. **What fixes the StereoMIS clips.** `surgical_core.stereomis.clips`
    computes the 86 clips each time it runs. It reads the tables of offsets
    and thresholds, and the `stats.npy` of the workbench's depth export
    `StereoMIS_depth`, which no tracked file makes. A change to an offset
    moves every `ate_rel`, but the hash of the camera trajectory measures
    covers `pose_metrics` only. Decide before the StereoMIS result is
    measured:
    - whether the clips are kept as data, as ATLAS-120k's population is;
    - whether the hash covers the tables and the clip rule;
    - whether the hash covers `evalkit/tools/pose_controls.py`, which makes
      the floors and the ceiling the result is compared with;
    - whether the export is made again here, or kept as an input named by
      its hashes.
32. **The claims table on the evaluator's scores.** `claims_table` prints
    the workbench's table from the pilot's scores. It does not yet print one
    from the evaluator's:
    - the three rows that read `t5_k10` cannot be scored until "Seeds made
      outside the tracking stage" is decided;
    - `--key` has no default. The workbench read every claim on
      `inst_F1_50_labeled`, the pilot's `labeled` domain. The evaluator's
      view closest to that domain is `all`;
    - the workbench's table read the 13-video ATLAS-120k set, 93 clips. The
      ported pipeline makes the granularity conditions on the 315 clips of
      `atlas120k_meta/clips.txt`. The manuscript's draft gives the merge on
      ATLAS-120k as +0.088 over 14 procedures, so the draft reads the 315
      clips.

    Decide which key or keys the claims are read on, and whether the
    granularity result answers a question whose primary metrics are already
    fixed ("Primary metrics" in `docs/evaluation.md`). Decide with it which
    population the ATLAS-120k column reads.
33. **What the public viewer may publish.** A bundle holds a clip's frames
    as JPEG, its masks as labels, and the scene graphs built on them. Two
    questions are open before a bundle is put online:
    - Whether each dataset's terms allow its frames and masks on a public
      page, and with which attribution. CholecSeg8k's frames come from
      Cholec80, and ATLAS-120k's from public videos.
    - What the page says about who made it. The workbench's demo named the
      authors, their university and their funders, with the funders' marks.
      AE-CAI is under double-blind revision, so this waits on the anonymity
      rules of Healthcare Technology Letters ("Prepare the release").
34. **The size of the point clouds.** A bundled clip is 60 to 90 MB, and its
    clouds are 97 % of it: a GLB holds one float32 position and one RGBA
    colour per pixel, 2.8 MB on DA3's grid and 4 MB on Pi3X's. A clip of 15
    frames takes about a minute to load on a 10 Mbit/s line. Quantised
    positions or a compressed GLB would shrink them several times, if the
    encoding keeps the points in their order, since the labels index them
    by order. Decide with the place the bundles are hosted.
35. **The StereoMIS depth outputs the ported stages do not reproduce.**
    `pipeline.depth_stereomis` refuses six clips of the workbench's trees:
    - 5 of the 6 masked clips of `P2_0` record `border_inpaint: true`. The
      workbench's stage looked for a border on every clip outside
      ATLAS-120k, and where it covered 1 % or more of the frame, filled it
      and zeroed the depth outside it; the five clips hold its
      `endoscope_content_mask.npy` and `prep_images/`, and the sixth,
      `clip_0008`, neither (checked). Their paint covers 11 to 26 % of the
      pixels and the edge of every frame, so the stage took it for a border.
      The ported stage fills none, so their `da3_masked` will change. All
      five are `slow` clips: they enter `da3_masked` over every clip, over
      `slow` and in the `P2_0` cell of the table by sequence, and no plain
      condition or `pi3x_masked` (checked).
    - `P1__clip_0001` (plain) holds Pi3X clouds of 250,880 points: it ran
      without `--max-points`. The flag thins the GLB and the clouds' records
      (`n_vertices`, `glb_centroid`, `median_vertices`) only; the workbench's
      `run_pi3_depth.py` writes the bundle before it (checked), so no score
      changes.

    Decide whether the masked condition is measured with the ported stage
    as it is, with new numbers on the five clips, or another way, and
    whether the manuscript reports the masked result. Decide whether
    `P1__clip_0001`'s Pi3X output is made again with the flag.
