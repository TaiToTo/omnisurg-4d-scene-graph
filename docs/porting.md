# Porting plan

**Status: in progress.** This is the plan for moving the code out of the
private research workbench (`$OMNISURG_SOURCE`) into this repository. It is
deleted once the port is done; git history keeps it.

It is committed on purpose. The first extraction kept its step-by-step plan
in the git-ignored `docs/local/`, on one machine, and that plan is now lost.

## History

The workbench is where every measurement so far was made. A first extraction
went to an earlier repository, `TaiToTo/omnisurg-4dsg` (private). Part of the
evaluation toolkit was reviewed there, on pull requests #2 to #6. None of them
merged, because two things changed:

- the paper is now measured with a new evaluator (below);
- the earlier repository's history keeps a personal email address in a pull
  request ref that cannot be removed.

This repository replaced it. The earlier repository stays, private, until the
port is done: the reviewed code on its branches is the starting point for most
of the toolkit here. `docs/evaluation.md` grew out of `docs/eval_v3.md` on its
`archive/metric-guide` branch.

The workbench's own planning documents are in Japanese, under
`$OMNISURG_SOURCE/docs/`:

- `repo_migration.md` gives the reasons and the rules.
- `repo_migration_plan.md` has the audit and the list of what moves.
- `repo_migration_determinism.md` shows that the three pipeline stages are
  byte-deterministic.

Where they and this document differ, this document holds.

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
    evaluator until the check against the pilot evaluator has passed, so
    that the check runs through the entry point that produces the paper's
    numbers. Pilot mode is `python -m evalkit.evaluate --pilot`, which
    scores each clip through `evalkit/pilot_clip.py`; it is not a script of
    its own. Once the check has passed, everything only pilot mode uses is
    removed: `evalkit/pilot.py`, `evalkit/pilot_clip.py`, the entry point's
    `--pilot`, and the `pilot` arguments of `evalkit/scored.py` and
    `evalkit/inst_bf.py`. The normal-mode tests must still pass, and then
    the evaluator is frozen. Moving pilot mode out of `evalkit/` was the
    alternative. It could then be deleted at any time, but the check would
    reach a driver of its own and never the entry point.
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
13. **No measure over time carries a star.** `time_IoU` stays a reference
    value, and no measure of identity from the workbench's `track_metrics`
    (hold, IDF1, ID switches, fragmentation, re-entry) joins the evaluator.
    `track_metrics` is ported as a tool under `evalkit/tools/`, outside
    `eval_code_sha`, and what it reports is a reference value: measured, and
    written about when it shows something, never marked with a star. So
    `docs/evaluation.md` defines no GT track. The tool defines its own when it
    is ported, without `MIN_AREA` (decision 5). The evaluator's whole-class
    object, followed over time, needs no rule to link pieces and no size;
    the workbench linked each class's components instead. If `hold_mean` is
    carried under that name, it keeps the workbench's definition, which was
    pre-registered there. A different denominator is a different metric,
    with a different name: in the workbench, moving the denominator once
    flipped a result's sign. `kmerge` is ported as a tool too. It changes
    predictions, not metrics: it merges a condition's regions down to K,
    and the evaluator's instance metrics score the merged regions, so a
    comparison of them is read like any other. The paper's granularity
    result merges to a fixed K of 10. The `matched` setting, which took K
    from the pilot's GT components of at least 300 px, is not carried for
    now: under the evaluator a GT object is a whole class, and that K would
    need a definition of its own.

## What moves

"Reviewed" names the earlier repository's branch that holds the version
reviewed there. Every one of them needs its comments rewritten in English, in
addition to the work listed.

### The evaluator

| part | from the workbench | reviewed | work |
|---|---|---|---|
| class tables, one per dataset | `surgical_core/cholec/__init__.py`, `surgical_core/atlas/labels.py` | — | New, as `docs/evaluation.md` defines them. The 30-class protocol comes from ATLAS-bench's `datasets/class_mapping.py`. |
| metrics, entry point, `eval_code_sha` | `depth_sam_tracking_experiment/eval_track.py`, `eval_gt_clips.py` | — | New, from `docs/evaluation.md`. |
| hand-derived test scenes | — | `archive/metric-guide` (`evalkit/tests/scenes.py`, `cartoon.py`) | Reuse the scenes. Expected values for the normal mode are derived again under the new rules. |
| pilot-mode tests | — | `archive/metric-guide` (`evalkit/tests/test_v2_metric_guide.py`) | Its expected values are the pilot evaluator's, worked out by hand from the scenes. Pilot mode must reproduce them, so they are kept as pilot-mode tests rather than derived again. |
| metric guide | — | `archive/metric-guide` (`docs/metrics/make_guide.py`, `index.html`) | The page as the pilot evaluator built it is `docs/pilot_metric_guide.html`, with notes where the evaluator differs. Whether a guide is built on the evaluator is the `evalkit/metric-guide` workstream in `docs/workstreams.md`; the script calls the pilot evaluator's functions, and would call the evaluator's instead. |

The workbench's other copies of the CholecSeg8k colour table still carry
the pilot's Hepatic Vein colour (`sam3_wrapper/scripts/cholec_utils.py`,
`track_cholec_gt_comparison.py`), and `scripts/extract_cholec_frames.py` has
a table of its own that is wrong and unused. When they are ported they read
`evalkit.classes` instead, or the table is deleted.

The same holds for ATLAS-120k: `surgical_core/atlas/labels.py` copies the
label table, and `scripts/extract_atlas_frames.py` turns the RGB masks into
ids with a colour table of its own (`rgb_mask_to_index`). The evaluator reads
both kinds of mask through `ClassTable.mask_ids`, so it needs no extracted
copy to read the RGB clips, and the ported copies read `evalkit.classes`.

### The toolkit around it

| file | from the workbench | reviewed | work |
|---|---|---|---|
| `paired_stats.py` | `ipcai2027_experiment/scripts/paired_stats.py` | `extract/03-metrics` (#3) | `VERDICT_RULE` does not change. The comparability check is `scores.check_comparable`, which applies the rule of `docs/evaluation.md`, in place of the pilot's, and the statistics are taken on the clips it compared. `video_of` is defined here rather than delegated. Each row records the metric's `sign` and `verdict` reads the interval in it, so a metric where smaller is better, or a reference value that is never marked, is not oriented by the caller. A bootstrap over fewer than two units refuses rather than return a point. |
| `compare_eval.py` | `depth_sam_tracking_experiment/compare_eval.py` | `extract/03-metrics` (#3) | It refuses to mix shas through `scores.check_comparable`, which also compares dataset, class set, view and mode. The table prints each metric's direction and no mark: the workbench's circle and cross followed the sign of the mean difference, a second verdict beside `paired_stats.verdict`. The per-clip list, the chart and `wins` follow the question's primary metric, named with `--key`; the directions of the pilot keys come from `scores.PILOT_SIGNS`, the one table `paired_stats` reads too. |
| `track_metrics.py` | `depth_sam_tracking_experiment/track_metrics.py` | `extract/03-metrics` (#3) | It imports `BACKGROUND`, `_gt_idmap` and `_load_depth` from the pilot's `eval_track`; they come from the evaluator instead. `MIN_AREA` is removed (decision 5). It is a tool, outside the evaluator, and its values are reference values (decision 13). |
| `surgical_core/clip_time.py` | `surgical_core/clip_time.py` | `extract/03-metrics` (#3) | English only. |
| `surgical_core/viewer/labels.py`, `palette.py` | `surgical_core/viewer/` | `extract/03-metrics` (#3) | English only. `label_table_of` and `cholec_gt_table` move to a new `gt_tables.py`, the one viewer module that imports `evalkit`; `labels.py` builds its table from plain data, so a video with no class table gets one too. The rest of `surgical_core/viewer` is below, under "Not yet extracted anywhere". |
| `kmerge.py` | `ipcai2027_experiment/scripts/kmerge.py` | `extract/04-kmerge` (#5) | It imports the pilot's `eval_track`, and borrows `verdict` from `paired_stats`. It keeps its own copy of `video_of`, which the port takes from `paired_stats` instead. It is a tool (decision 13), and takes the evaluator's instance metrics in place of the pilot's. |
| `surgical_core/geometry/` | `depth_sam_tracking_experiment/geometry.py` | `extract/04-kmerge` (#5) | English only. Shared by the pipeline and the toolkit. |
| `condition_inventory.py` | `ipcai2027_experiment/scripts/condition_inventory.py` | `extract/05-inventory` (#4) | It reads `eval_code_sha`, `eval_code_tag` and `eval_version` from score JSONs. "One ruler" is now what `docs/evaluation.md` calls comparable: sha, dataset, class set, view and mode all equal, and one propagation rule. `bidir` joins the provenance that tells one condition from another. |
| `check_env.py` | `ipcai2027_experiment/atlas97/scripts/check_env97.py` | `extract/05-inventory` (#4) | Not carried (decision 4). |
| `reeval_diff.py` | — (written in the earlier repository) | `extract/06-rescore` (#6) | Ported as `evalkit/tools/pilot_check.py`, the check against the pilot evaluator, run in the workbench. The earlier repository's version insists the sha equals the pilot's and diffs `eval_code_sha` with everything else; `pilot_check` works the other way round: the shas differ by construction, only the keys the two evaluators share are compared, and those must be equal. |

Tests come with the file they test:

- `test_paired_stats_bootstrap.py` comes with `paired_stats`.
- `test_eval_identity.py` comes with `track_metrics`. Its part that pins the
  sha over the GT loaders changes with the evaluator.
- `test_geom_edge_ring.py` comes with `geometry`.
- `test_kmerge_per_clip.py` and `test_kmerge_root_resolution.py` come with
  `kmerge`.
- `test_condition_inventory_roots.py` comes with `condition_inventory`.

All of these are in `$OMNISURG_SOURCE/depth_sam_tracking_experiment/tests/`.

Not carried: `test_frozen_sha.py`, `test_eval_track_v2_regression.py`, its
fixture `eval_track_v1.py`, and `test_eval_track_metrics.py`. All of them test
the pilot evaluator. The properties the last one pins are pinned again by the
evaluator's hand-derived tests.

The workbench tests that touch the ported modules are counted in
`repo_migration_plan.md` as 17 in `tests/` and 6 in
`depth_sam_tracking_experiment/tests/`. List them when their module is ported.

### Not yet extracted anywhere

From the take list in `repo_migration_plan.md`:

- **Pipeline.**
  - `scripts/`: `run_cholec_depth.py`, `regen_cholec_glb.py`,
    `extract_atlas_frames.py`, `extract_cholec_frames.py`,
    `crop_cholec_frames.py`, `export_viewer_dataset.py`,
    `build_temporal_graph.py`, `export_instrument_mask.py`,
    `run_atlas_pipeline.sh`, `run_gt_tracked_export.sh`,
    `measure_determinism.py`.
  - `depth_sam_tracking_experiment/`: `track_sam3.py`, `sam3d_core.py`,
    `loaders.py`, `depth_source.py`, `viz_common.py`, `sam_env.py`.
  - `ipcai2027_experiment/scripts/`: `run_per_frame_seg.py`, `geom_blend.py`.
  - `pi3_wrapper/scripts/run_pi3_depth.py`.
  - `crop_cholec_frames.py` cuts each CholecSeg8k video to a rectangle
    inside the endoscope's view. Ported, it reads the rectangle as data, the
    one each clip of that video records in `crop_info.json`, as the
    ATLAS-120k crop rectangles below are data; the circle fit in
    `preprocess` that found it comes along only to find a new video's. The
    depth stage then neither detects nor fills a border
    (`docs/workstreams.md`, "No endoscope border in the depth stage").
- **Wrappers.** `da3_surgery_wrapper/` and `pi3_wrapper/` become
  `recon3d_wrapper/`, and `sam3_wrapper/` moves as it is.
- **Rest of `surgical_core`.** `pointcloud`, `preprocess`, the ten modules of
  `viewer` not listed above, and the non-evaluation parts of `cholec`
  (`cholect50.py`, `seg8k_align.py`) and `atlas`.
- **ATLAS-120k metadata, into `atlas120k_meta/`.** No video goes in.
  - Crop rectangles: `experiment/crop_necessity/verdicts/verdicts_latest.json`.
  - Cut marks: `experiment/crop_necessity/marks/*.jsonl`.
  - The 315-clip population: `ipcai2027_experiment/frozen/atlas97_clips.txt`.
    Clips of the release overlap in two videos. `tests/test_atlas120k_meta.py`
    pins the overlaps and checks that no two clips of the population share a
    frame. The population's reader, when it is ported, refuses two clips of
    one video that share a frame.
  - The depth manifest: `ipcai2027_experiment/frozen/atlas97_depth_manifest.json`.
  - The 100-video manifest and audit:
    `ipcai2027_experiment/task22_atlas100/out/{manifest,audit}/`.
  - The readers: `surgical_core/atlas/clip_rects.py` and `frame_ratio.py`,
    into `surgical_core/atlas120k/`.
- **Viewer.**
  - The `demo`, `workbench` and `depthcmp` pages.
  - Their `src/` directories.
  - `viewer/scripts/merge_geometry_sources.py` and
    `verify_geometry_alignment.mjs`.
  - `vite.config.js`, without the experiment routes.

What stays behind is listed in `repo_migration_plan.md`, in the section on
what stays.

## Order, and what must hold before the next step

Which of these can move side by side, in separate branches and sessions,
while the evaluator is built and reviewed is worked out in `docs/workstreams.md`.

1. **Decide what the evaluator measures over time.** Decided: no measure
   over time carries a star, and `track_metrics` and `kmerge` are tools
   (decision 13).
2. **Build the evaluator.** CPU only, as the `evalkit` package. Class tables,
   then metrics, then the entry point. It is done when:
   - the hand-derived tests pass, in both modes;
   - in pilot mode, the evaluator reproduces every key it shares with the
     pilot evaluator, at zero tolerance, on the 38 scored conditions.
   Record its sha with every score; do not freeze it (decision 2). Pilot
   mode stays in it until just before the freeze (decision 11).
3. **Re-score.** CPU only. Score every condition's existing predictions with
   the evaluator. `condition_inventory` must report no mixed ruler and no
   missing condition.
4. **Port the toolkit onto it.** Each tool is done when its tests pass, and:
   - on the pilot's score JSONs, it writes the same bytes as the workbench
     version (the bootstrap is seeded). This holds for every tool. The
     pilot's JSONs lack the fields the evaluator now writes (class set, view,
     mode, input shas, versions, propagation rule); a tool reads them all the
     same, taking the missing fields as the pilot evaluator's, and raises
     when a JSON has some of the fields but not all. Dropping this check
     would let a tool lose behaviour the workbench version had without
     anyone noticing;
   - what only the evaluator's JSONs carry is checked by a self-test that
     plants the fault: `compare_eval` refuses a mix of shas, and a mix of
     modes or class sets under one sha; `condition_inventory` reports a
     planted mixed ruler and a planted missing condition.
   The paper's numbers are the step-3 scores read through these tools.
5. **Port the pipeline, one stage at a time.** Each stage's output must match
   the workbench byte for byte. Two exceptions: Pi3X's `runtime_sec`, per
   `repo_migration_determinism.md`; and the 14 CholecSeg8k clips whose gap
   frames the workbench's extractor placed 1 to 3 s late ("CholecSeg8k
   clips whose frames run out of order"), which the ported extractor
   converts or refuses, and which are then re-extracted and re-run. The 315
   clips also get DA3 and `glb_centroid`: the demo's reference grid stays
   DA3.
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
   against the pilot evaluator.
6. **Prepare the release.**
   - An English README, `docs/data_contract.md`, the `atlas120k_meta/` README
     and `CITATION.cff`.
   - Before anything is public, a check of HTL's anonymity rules, since AE-CAI
     is under double-blind revision.
   - Done when a third party can clone, install and get a green `pytest`,
     and, on a machine with a CUDA GPU, install the pipeline with the
     README's steps and run each ported stage on a clip.

The data these steps read stays in the workbench:

- `outputs/atlas97` and `outputs/cholec_gt`, which the paper reads;
- `outputs/atlas`, the 13-video tree the demo reads, which holds the clips
  the determinism measurement used (`adrenalectomy__16GPCUPkXYQ__gt_0004` and
  `adrenalectomy__16GPCUPkXYQ__tile_0007`) and is where step 5 compares
  stages, with one CholecSeg8k clip from `outputs/cholec_gt`;
- the determinism measurement's JSON (`measure_determinism.py --json-out`);
- the predictions under `ipcai2027_experiment/atlas97/out/`;
- the pilot's score JSONs.

Every command takes those paths as arguments.

## Open questions

1. **The skill-classification code and the other 18 viewer pages.** The plan
   leaves both behind. The paper's figures come from some of those pages.
2. **Whether step 5 has to finish before submission.** Step 3 already gives
   the numbers. A stage that passes step 5's byte check cannot change them;
   the 14 clips re-extracted under "CholecSeg8k clips whose frames run out
   of order" can, and so can depth made again under "Depth made by two
   versions of the depth stage".
3. **Boundary dilation before the freeze.** `evalkit/boundary.py` dilates
   with `cv2.dilate`; a numpy shift-or over the (2·tol + 1)² offsets agrees
   on every mask tried, borders included. The question is whether a hashed
   file should depend on a library's behaviour at all while OpenCV is
   unpinned. Decide before the evaluator is frozen. Pilot mode depends on
   OpenCV more deeply: `evalkit/pilot.py` numbers a class's components in
   the order `cv2.connectedComponents` labels them, and the pairing's tie
   rule reads those numbers. That never reaches the frozen files, since
   pilot mode is removed before the freeze (decision 11).
4. **The benchmark mapping against its source.** The ATLAS-120k mapping to
   the benchmark's 30 classes was typed from the document and checked by
   hand against ATLAS-bench's `datasets/class_mapping.py` at commit
   e286a584, all 47 ids agreeing. A script that takes that file's path and
   repeats the check would make it reproducible.
5. **CholecSeg8k's Region line.** The white line between regions is 1 px
   wide; in the masks 87 % of its pixels are the image's outer 1 px (gone
   with the crop) and the rest sits mostly in video43 and video52. Left
   `ignored`, an edge against it is no boundary, so those videos lose much of
   their GT boundary, and unevenly: after the nearest-neighbour resize the
   line survives only in places. Decide whether the loader fills the line
   from its neighbours, at full resolution, by a deterministic rule with the
   filled count recorded, or whether it stays ignored with the loss
   documented. Either way the pilot evaluator read it as background and
   counted an edge against it as a boundary, a normal-mode difference to list.
6. **CholecSeg8k clips whose frames run out of order.** In videos where
   CholecSeg8k numbers frames at about 30 fps, the workbench's extractor
   (`scripts/extract_cholec_track.py`) resolves the true 25 fps frame for the
   frames that carry a mask and leaves the gap frames it decodes from the
   video at the unconverted number, 1 to 3 s later in the video than their
   place in the clip; the recorded times are right, the order of the images
   is wrong, and one image can appear twice. 14 of the 27 pilot clips are
   affected, 11 visibly. Removing them changes the verdict of several F1 and
   IDF1 comparisons (power, not sign) and none of the `boundary_F` or
   `hold_mean` ones. When the extractor is ported in step 5 it converts the
   gap frames or refuses the video, the 14 clips are re-extracted and re-run,
   and that is a deliberate exception to step 5's byte-for-byte rule.
   The evaluator refuses two of the nine clips on this machine. In
   `VID25_s15_162_crop` native frame 387 appears twice, at one timestamp.
   In it and in `VID25_s15_402_crop`, the gap frames hold masks the
   viewer's step wrote before the clips were extracted again, 504 × 504
   where the GT is 457 × 456, on frames the manifest says have none: 8 and
   19 frames. The pilot's scores of these clips count the annotated frames
   only, so its data root did not hold those masks.
7. **The evaluator map against `evalkit/frame.py`.** Two things to carry
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
8. **The fewest videos for an interval.** `paired_stats.boot_ci` refuses a
    population of one video, where every resample is the same video and
    the interval is a point. Two is the floor that removes that failure,
    not a statistical one: with n videos the chance that a resample draws
    one video n times is n^-n, above 2.5 % up to three videos, so on two
    or three the 95 % interval is the range of the video means, and two
    videos that agree in sign give a mark. `wilcoxon_video` draws its own
    line at six for the same reason. Whether the interval gets a floor
    above two, and where, is a decision about the paper's populations, not
    the code's; until it is made, a subset's interval is read for its sign
    only, as `--drop-video` says.
9. **Depth made by two versions of the depth stage.** On the development
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
10. **Two orders of the world transform.** `cam_to_world` computes
    `(p - t) @ R`; the workbench's back-projection computes
    `(R.T @ (p.T - t)).T`. On the development machine the two give the same
    bits only where the BLAS runs the same kernel: under numpy 2.5.3 on
    Accelerate they differ in the last bit below about 1024 points (at the
    tests' 9×11 frame, 20 of 288 elements) and agree above; under numpy
    1.26.4 on OpenBLAS they agree at every size tried. The depth stages'
    clouds are far above the line, so step 5's byte check on G does not
    answer it. `project.backproject` carries the order further, but no
    stage calls it: in the workbench, only two analyses of view consistency
    do (`view_consistency.py`, `camera_motion.py`), and their point sets can
    be small. If they are ported and a difference surfaces in their check,
    the choice is between restoring the workbench's order and accepting a
    documented non-bit-equality — made then, not found later.
11. **Whether the geometry path has to be fast.** The depth stage
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
    resolution is fixed. That rebuild is a few per cent of the time.
    `backproject` also back-projects every frame it is given, on the path
    of the view-consistency analyses, not on the export. None of this is
    worth changing while the port lasts. float32 changes the
    output, and the pipeline port asks each stage to match the workbench
    byte for byte. The byte check reads the GLB files too. Decide once the
    stages match, and decide with it whether a viewer ever recomputes
    geometry or only reads what the export wrote. The functions are in
    `surgical_core/geometry/camera.py` and
    `surgical_core/geometry/project.py`.
12. **Conditions seeded from GT.** A seed frame is scored like any other
    frame, because the paper's conditions are seeded from the pipeline's own
    masks. Which of the 38 conditions were seeded from GT instead, and
    whether such a condition is scored on its seed frame or enters a table
    at all, is settled before step 3, on the machine that holds the
    predictions. The `seed_source` that each
    condition's `seed_info.json` records says where its seed came from;
    where it does not tell, the command that made the condition does.
    The workshop's oracle row, GT instrument masks painted onto a
    condition's labels, is one; the viewer's `gt_tracked` track, one GT
    frame carried by SAM 3, is another candidate. What the tracking stage
    does with such a seed is "The seed frame chosen from GT".
13. **Masks that are not GT under the GT's name.** The viewer's step writes
    SAM 3 masks into `seg_masks/` as `<i>_color_mask.png`, told apart from
    the annotation only by the frame manifest's `is_anchor` and
    `seg_provenance`, and the two VID25 clips still hold such masks from
    before their re-extraction, on frames the frame manifest marks as having
    none. The evaluator reads the flags and refuses the VID25 clips until
    those files are removed. When the pipeline is ported (step 5), decide
    whether a mask that is not annotation moves out of `seg_masks/` or takes
    a name of its own, so that the distinction is in the file and not only in
    the frame manifest.
