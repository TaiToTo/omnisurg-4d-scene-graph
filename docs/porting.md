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
4. **`evalkit/` is a package.** Tools import the evaluator and each other
   as `evalkit.<module>`, with no `sys.path` edits. `check_env97` existed
   because frozen scripts could not check their own import path; a package
   does not need it. `pyproject.toml` includes `evalkit*` next to
   `surgical_core*`.
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
| `paired_stats.py` | `ipcai2027_experiment/scripts/paired_stats.py` | `extract/03-metrics` (#3) | `VERDICT_RULE` does not change. The comparability check now comes from the evaluator, not the pilot's `check_comparable`. `video_of` is defined here rather than delegated. |
| `compare_eval.py` | `depth_sam_tracking_experiment/compare_eval.py` | `extract/03-metrics` (#3) | It refuses to mix shas through the evaluator's check, which also compares dataset, class set, view and mode. |
| `track_metrics.py` | `depth_sam_tracking_experiment/track_metrics.py` | `extract/03-metrics` (#3) | It imports `BACKGROUND`, `_gt_idmap` and `_load_depth` from the pilot's `eval_track`; they come from the evaluator instead. `MIN_AREA` is removed (decision 5). Whether it is part of the evaluator waits on open question 1; if it is, it moves to the table above. |
| `surgical_core/clip_time.py` | `surgical_core/clip_time.py` | `extract/03-metrics` (#3) | English only. |
| `surgical_core/viewer/labels.py`, `palette.py` | `surgical_core/viewer/` | `extract/03-metrics` (#3) | English only. The rest of `surgical_core/viewer` is below, under "Not yet extracted anywhere". |
| `kmerge.py` | `ipcai2027_experiment/scripts/kmerge.py` | `extract/04-kmerge` (#5) | It imports the pilot's `eval_track`, and borrows `verdict` and `video_of` from `paired_stats`. Waits on open question 1. |
| `surgical_core/geometry/` | `depth_sam_tracking_experiment/geometry.py` | `extract/04-kmerge` (#5) | English only. Shared by the pipeline and the toolkit. |
| `condition_inventory.py` | `ipcai2027_experiment/scripts/condition_inventory.py` | `extract/05-inventory` (#4) | It reads `eval_code_sha`, `eval_code_tag` and `eval_version` from score JSONs. "One ruler" is now what `docs/evaluation.md` calls comparable: sha, dataset, class set, view and mode all equal. |
| `check_env.py` | `ipcai2027_experiment/atlas97/scripts/check_env97.py` | `extract/05-inventory` (#4) | Not carried (decision 4). |
| `reeval_diff.py` | — (written in the earlier repository) | `extract/06-rescore` (#6) | Becomes the check against the pilot evaluator, run in the workbench. Today it insists the sha equals the pilot's and diffs `eval_code_sha` with everything else; the check is the other way round: the shas differ by construction, only the keys the two evaluators share are compared, and those must be equal. |

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
- **Wrappers.** `da3_surgery_wrapper/` and `pi3_wrapper/` become
  `recon3d_wrapper/`, and `sam3_wrapper/` moves as it is.
- **Rest of `surgical_core`.** `pointcloud`, `preprocess`, the ten modules of
  `viewer` not listed above, and the non-evaluation parts of `cholec`
  (`cholect50.py`, `seg8k_align.py`) and `atlas`.
- **ATLAS-120k metadata, into `atlas120k_meta/`.** No video goes in.
  - Crop rectangles: `experiment/crop_necessity/verdicts/verdicts_latest.json`.
  - Cut marks: `experiment/crop_necessity/marks/*.jsonl`.
  - The 315-clip population: `ipcai2027_experiment/frozen/atlas97_clips.txt`.
    TODO: in one video, cholecystectomy `_-aytJndMV4`, adjacent clips of the
    release share 441 frames. The population holds two of its clips, which
    share none, but nothing checks it: the population's reader should refuse
    two clips that share a frame.
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

1. **Settle open question 1.** It decides what the evaluator computes and
   what `track_metrics` and `kmerge` become.
2. **Build the evaluator.** CPU only, as the `evalkit` package. Class tables,
   then metrics, then the entry point. It is done when:
   - the hand-derived tests pass, in both modes;
   - in pilot mode, the evaluator reproduces every key it shares with the
     pilot evaluator, at zero tolerance, on the 38 scored conditions.
   Record its sha with every score; do not freeze it (decision 2).
   The first `pip install -e .` of this repository happens here. Compare the
   packages it resolves with `environment.packages` in the determinism
   measurement's JSON, so that a mismatch is known before step 5 rather than
   found there.
3. **Re-score.** CPU only. Score every condition's existing predictions with
   the evaluator. `condition_inventory` must report no mixed ruler and no
   missing condition.
4. **Port the toolkit onto it.** Each tool is done when its tests pass, and:
   - on the pilot's score JSONs, it writes the same bytes as the workbench
     version (the bootstrap is seeded). This holds for every tool. The
     pilot's JSONs lack the fields the evaluator now writes (class set, view,
     mode, input shas, versions); a tool reads them all the same, taking the
     missing fields as the pilot evaluator's, and raises when a JSON has
     some of the fields but not all. Dropping this check would let a tool
     lose behaviour the workbench version had without anyone noticing;
   - what only the evaluator's JSONs carry is checked by a self-test that
     plants the fault: `compare_eval` refuses a mix of shas, and a mix of
     modes or class sets under one sha; `condition_inventory` reports a
     planted mixed ruler and a planted missing condition.
   The paper's numbers are the step-3 scores read through these tools.
5. **Port the pipeline, one stage at a time.** Each stage's output must match
   the workbench byte for byte. The one exception is Pi3X's `runtime_sec`, per
   `repo_migration_determinism.md`. If the package comparison in step 2 found
   a difference, measure determinism again first. The 315 clips also get DA3
   and `glb_centroid`: the demo's reference grid stays DA3.
6. **Prepare the release.**
   - An English README, `docs/data_contract.md`, the `atlas120k_meta/` README
     and `CITATION.cff`.
   - Before anything is public, a check of HTL's anonymity rules, since AE-CAI
     is under double-blind revision.
   - Done when a third party can clone, install and get a green `pytest`.

The data these steps read stays in the workbench:

- `outputs/atlas97` and `outputs/cholec_gt`, which the paper reads;
- `outputs/atlas`, the 13-video tree the demo reads, which holds the clips
  the determinism measurement used (`adrenalectomy__16GPCUPkXYQ__gt_0004` and
  `tile_0007`) and is where step 5 compares stages;
- the determinism measurement's JSON (`measure_determinism.py --json-out`);
- the predictions under `ipcai2027_experiment/atlas97/out/`;
- the pilot's score JSONs.

Every command takes those paths as arguments.

## Open questions

1. **Identity metrics and merge cost.** `docs/evaluation.md` leaves
   consistency over time undecided: `time_IoU` is a reference value, and the
   candidates (hold, IDF1, ID switches, fragmentation, re-entry) live in the
   workbench's `track_metrics` and are not reported. Choosing one means
   defining a GT track, a second object definition finer than the whole-class
   object, and saying how the two coexist (`docs/evaluation.md`, "Consistency
   over time"). Decide whether they and `kmerge` join the evaluator. If
   `hold_mean` is carried under that name it keeps the workbench's definition,
   which was pre-registered there; a different denominator is a different
   metric, with a different name. In the workbench, moving the denominator
   once flipped a result's sign.
2. **The skill-classification code and the other 18 viewer pages.** The plan
   leaves both behind. The paper's figures come from some of those pages.
3. **Whether step 5 has to finish before submission.** Step 3 already gives
   the numbers, and the determinism result says step 5 cannot change them.
4. **Pilot mode's own rules.** Five places where pilot mode must not read
   the evaluator's tables or helpers, each noted where it was found and
   collected here so the pilot-mode driver settles them in one go:
   - ATLAS-120k typing: the pilot evaluator knew one type, Tools/camera;
     the `original` set types every id. Pilot mode needs a typing of its own
     (`evalkit/classes.py`).
   - `mIoU`: `ClassScores.miou` sums IoUs in class id order; the pilot took
     `np.mean` over a dict in set order. The last bit differs on about a
     quarter of frames, and zero tolerance is the promise, so the driver
     averages `ious` the pilot's way (`evalkit/classmap.py`).
   - `time_IoU`: the pilot ordered frames by `sorted()` of the `label_*.npy`
     names, lexicographic, not numeric unless zero-padded; the driver orders
     them that way in pilot mode, and writes the one per-clip value under
     every view (`evalkit/time_iou.py`).
   - `EXTRA_IGNORE`: the pilot removed those ids too, from the command line,
     recorded in each score as `extra_ignore`. Every score in the workbench
     that records it has it empty, but those are the workshop's; confirm on
     the 38 conditions' JSONs before pilot mode assumes an empty set
     (`evalkit/vi.py`).
   - The clip means: `summarize_clip` averages with `sum() / len()`, takes a
     mean only over the frames a key is defined on and leaves None where
     there is none; the pilot took `np.mean`, wrote 0 where no frame defined
     a key, and rounded the summary to four decimals. The last bit of the
     two means differs on about 40 % of random clips, so the pilot-mode
     driver aggregates the pilot's way and does not call `summarize_clip`
     (`evalkit/clip.py`).
5. **Boundary dilation before the freeze.** `evalkit/boundary.py` dilates
   with `cv2.dilate`; a numpy shift-or over the (2·tol + 1)² offsets agrees
   on every mask tried, borders included. The question is whether a hashed
   file should depend on a library's behaviour at all while OpenCV is
   unpinned. Decide before the evaluator is frozen.
6. **The benchmark mapping against its source.** The ATLAS-120k mapping to
   the benchmark's 30 classes was typed from the document and checked by
   hand against ATLAS-bench's `datasets/class_mapping.py` at commit
   e286a584, all 47 ids agreeing. A script that takes that file's path and
   repeats the check would make it reproducible.
7. **CholecSeg8k's Region line.** The white line between regions is 1 px
   wide; in the masks 87 % of its pixels are the image's outer 1 px (gone
   with the crop) and the rest sits mostly in video43 and video52. Left
   `ignored`, an edge against it is no boundary, so those videos lose much of
   their GT boundary, and unevenly: after the nearest-neighbour resize the
   line survives only in places. Decide whether the loader fills the line
   from its neighbours, at full resolution, by a deterministic rule with the
   filled count recorded, or whether it stays ignored with the loss
   documented. Either way the pilot evaluator read it as background and
   counted an edge against it as a boundary, a normal-mode difference to list.
8. **CholecSeg8k clips whose frames run out of order.** In videos where
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
9. **What the geometric view cannot see.** The geometric view removes the
   `appearance` pixels from the GT and the regions alike, so a region lying
   on them only, such as a blood spot a colour-driven pipeline cuts out of
   the liver, is no object and costs nothing. Count, in both datasets' GT
   masks, the `appearance` pixels in connected components whose whole outer
   border is one `tissue` class: the spots where the organ plainly runs on
   underneath. If they are rare, nothing changes. If not, decide before the
   freeze whether the evaluator counts the regions lying wholly on removed
   pixels, reported like `unlabelled_share` and never starred. Filling the
   spots from their neighbours was considered and set aside; the reasons are
   in `docs/evaluation.md`, "Views".
10. **The evaluator map against `evalkit/frame.py`.** Two things to carry
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
