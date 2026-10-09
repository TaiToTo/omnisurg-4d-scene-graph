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
    its own. Once the check has passed, everything in the hashed files that
    only pilot mode uses is removed:
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
    compared condition registered, in `summarize_track16`. The tool carries
    at most one of them under that name, and the other under a name of its
    own, because a different denominator is a different metric: in the
    workbench, moving the denominator once flipped a result's sign. Which
    one keeps the name is settled with the tool's GT track.
15. **`kmerge` is a tool.** `kmerge` is ported under `evalkit/tools/`,
    outside `eval_code_sha`. It changes predictions, not metrics: it merges
    a condition's regions down to K and writes them as a condition of their
    own, which the evaluator scores and `compare_eval` compares. So it
    computes no metric, and a comparison of a merged condition can carry a
    star like any other. The paper's granularity result merges to a fixed K
    of 10. The `matched` setting, which took K from the pilot's GT
    components of at least 300 px, is not carried for now: under the
    evaluator a GT object is a whole class, and that K would need a
    definition of its own.

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
a table of its own that is wrong and unused. None of the three is carried for
now ("Not carried for now"), so the tables stay in the workbench. A later
port reads `evalkit.classes` instead, or deletes the table.

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
| `track_metrics.py` | `depth_sam_tracking_experiment/track_metrics.py` | `extract/03-metrics` (#3) | It imports `BACKGROUND`, `_gt_idmap` and `_load_depth` from the pilot's `eval_track`; they come from the evaluator instead. `MIN_AREA` is removed ("No minimum object size, anywhere"). It is a tool, outside the evaluator, and its values are reference values ("No measure over time carries a star"). Its GT track is open ("The GT track of `track_metrics`"). |
| `surgical_core/clip_time.py` | `surgical_core/clip_time.py` | `extract/03-metrics` (#3) | English only. |
| `surgical_core/viewer/labels.py`, `palette.py` | `surgical_core/viewer/` | `extract/03-metrics` (#3) | English only. `label_table_of` and `cholec_gt_table` move to a new `gt_tables.py`, the one viewer module that imports `evalkit`; `labels.py` builds its table from plain data, so a video with no class table gets one too. The rest of `surgical_core/viewer` is below, under "Not yet extracted anywhere". |
| `kmerge.py` | `ipcai2027_experiment/scripts/kmerge.py` | `extract/04-kmerge` (#5) | Ported as `evalkit/tools/kmerge.py`, outside `eval_code_sha` ("`kmerge` is a tool"). It merges a condition's predictions down to K and writes them as a condition of their own, which the evaluator scores and `compare_eval` compares, so it computes no metric; `kmerge.json` records the source and the sha256 of its predictions, so a source re-tracked after the merge is told apart. Its merge equals the workbench's `kmerge_sequence` on 590 maps. It differs from the workbench in what it reads: it merges every frame that has a prediction, where the workbench merged every fifth; and it refuses a frame with a pixel without valid depth, or with a region id below -1, as the evaluator does, where the workbench dropped the labels of such pixels and merged on. Not carried for now: its scoring with the pilot's instance metrics, the curve over K, the `matched` K, `--pairs`, and the merges by threshold and by geometry (`tmerge_sequence`, `kgeo_sequence`). |
| `surgical_core/geometry/` | `depth_sam_tracking_experiment/geometry.py` | `extract/04-kmerge` (#5) | English only. Shared by the pipeline and the toolkit. |
| `pose_metrics.py` | `ipcai2027_experiment/scripts/pose_metrics.py` | — | Ported as `evalkit/tools/pose_metrics.py`, to be hashed on its own ("The camera trajectory measures are hashed on their own"). It measures an estimated camera trajectory against StereoMIS's: `ate` after one similarity fit, `rpe` and `scale_consistency`; `ate_rel` decides the result. Its values equal the workbench's on 1200 random, planar and static trajectories. It refuses what the workbench let through: trajectories of unequal length, times out of order and values that are not finite. The rest of the StereoMIS result is listed under "Not yet extracted anywhere": `stereomis_io.py` (calibration, rectified frames, the measured offsets between video, ground truth and depth), `pose_controls.py` (the static camera, constant motion and stereo visual odometry the result is read against), and `summarize_20.py` and `run_20.sh` (the run and its table). `d4d_pose.py`, the D4D check of the same result, is listed there with D4D. None of StereoMIS's data is on the development machine, so they are checked on the workbench's GPU machine when they move. |
| `condition_inventory.py` | `ipcai2027_experiment/scripts/condition_inventory.py` | `extract/05-inventory` (#4) | It reads `eval_code_sha`, `eval_code_tag` and `eval_version` from score JSONs. "One ruler" is now what `docs/evaluation.md` calls comparable: `eval_code_sha`, dataset, class set, view and mode all equal, and one propagation rule. The provenance fields now include `bidir`, whether the tracker ran both ways. |
| `check_env.py` | `ipcai2027_experiment/atlas97/scripts/check_env97.py` | `extract/05-inventory` (#4) | Not carried (decision 4). |
| `reeval_diff.py` | — (written in the earlier repository) | `extract/06-rescore` (#6) | Ported as `evalkit/tools/pilot_check.py`, the check against the pilot evaluator, run in the workbench. The earlier repository's version insists the sha equals the pilot's and diffs `eval_code_sha` with everything else; `pilot_check` works the other way round: the shas differ by construction, only the keys the two evaluators share are compared, and those must be equal. |

Tests come with the file they test:

- `test_paired_stats_bootstrap.py` comes with `paired_stats`.
- `test_eval_identity.py` comes with `track_metrics`. Its part that pins the
  sha over the GT loaders changes with the evaluator.
- `test_geom_edge_ring.py` comes with `geometry`.
- `test_kmerge_per_clip.py` and `test_kmerge_root_resolution.py` tested what
  the port of `kmerge` leaves out: its averaging by clip, and the root it
  read clips from. `tests/test_kmerge.py` tests the merge.
- `test_condition_inventory_roots.py` comes with `condition_inventory`.
- `pose_metrics` has no test in the workbench; its `selftest` is
  `tests/test_pose_metrics.py`, with the values worked out by hand.

All of these are in `$OMNISURG_SOURCE/depth_sam_tracking_experiment/tests/`,
apart from the two tests of `kmerge` and `test_condition_inventory_roots.py`,
which are in `$OMNISURG_SOURCE/tests/`.

Not carried: `test_frozen_sha.py`, `test_eval_track_v2_regression.py`, its
fixture `eval_track_v1.py`, and `test_eval_track_metrics.py`. All of them test
the pilot evaluator. The properties the last one pins are pinned again by the
evaluator's hand-derived tests.

The workbench tests that touch the ported modules are counted in
`repo_migration_plan.md` as 17 in `tests/` and 6 in
`depth_sam_tracking_experiment/tests/`. List them when their module is ported.

### Pipeline stages already ported

Each stage is described in `docs/pipeline.md`.

- `scripts/run_cholec_depth.py` is `pipeline/depth.py`. The ported stage
  neither detects nor fills a border: the CholecSeg8k runs passed
  `--no-border-inpaint`, and the ATLAS-120k runs left `--border-inpaint` at
  `auto`.
- `scripts/regen_cholec_glb.py` is `pipeline/point_clouds.py`.
- `pi3_wrapper/scripts/run_pi3_depth.py` is `pipeline/pi3x.py`.
- `scripts/measure_determinism.py` is `pipeline/byte_check.py`, which checks
  every stage, not the depth stage alone.
- `scripts/extract_atlas_frames.py`, as
  `experiment/crop_necessity/run_atlas97_extract.sh` runs it
  (`--skip-tiles --gt-step-sec 0.52 --gt-min-frames 8 --clip-rects`), is
  `pipeline/extract_atlas120k.py`. The stage runs every video itself and
  checks the clips it writes against `atlas120k_meta/clips.txt`, in place
  of `run_atlas97_extract.sh`, `plan_atlas97_population.py` and
  `verify_atlas97_population.py` from `experiment/crop_necessity/`.

### Not yet extracted anywhere

This list comes from the take list in `repo_migration_plan.md`, checked
against what the paper's conditions ran: the workbench's run scripts and the
documents its claims are written in. Within a file, only the code paths
those runs exercise move; the rest is listed under "Not carried for now"
below.

- **Pipeline.**
  - Beside the ATLAS-120k extraction: `run_atlas97_depth.sh` from
    `experiment/crop_necessity/`, and
    `ipcai2027_experiment/task22_atlas100/scripts/prepare_video_root.py`,
    which converts the one AV1 video of the population
    (`rarp/NitKIjCcS7U`) to H.264.
  - CholecSeg8k extraction: `scripts/extract_cholec_track.py`, which
    `depth_sam_tracking_experiment/run_all17_pipeline.sh` runs before the
    crop and the depth stage. `extract_cholec_frames.py`, which the take
    list named, ran in no condition.
  - `crop_cholec_frames.py` cuts each CholecSeg8k video to a rectangle
    inside the endoscope's view, with `--method circle`. Ported, it reads
    the rectangle as data, one per clip in `cholecseg8k_meta/crop_rects.json`
    beside the 27 clips' list, as the ATLAS-120k crop rectangles below are
    data. The depth stage then neither detects nor fills a border
    (`docs/workstreams.md`, "No endoscope border in the depth stage").
  - Segmentation and tracking, from `depth_sam_tracking_experiment/`:
    `track_sam3.py`, `sam3d_core.py` (`num_to_natural` and `get_sam`),
    `loaders.py` (`CholecGtLoader`), `depth_source.py` (DA3 and the `pi3`
    swap), `viz_common.py` and `sam_env.py`. From
    `ipcai2027_experiment/scripts/`: `run_per_frame_seg.py`, whose five
    modes `geom_blend.sam_input_image` passes to `geometry.sam_input_image`
    unchanged, and `run_conditions.py`, which spreads one tracked condition
    over the GPUs and imports the command from `run_track_conditions.py`.
    From `seg_quality_experiment/scripts/`: `run_track_conditions.py`, whose
    `MODES` hold the operating point's flags (`--seed_auto --bidir
    --max_frames 0`). The operating point's `--seed_inst_thresh 1.0` is not
    in `MODES`: the drivers `ipcai2027_experiment/atlas97/scripts/run_batch97.sh`
    and `ipcai2027_experiment/task11_atlas13/scripts/run_batch.sh` pass it
    after `--`, and neither driver moves. The ported stage needs no such
    flag, because it seeds on the centre frame and never chooses a seed
    frame from GT ("Two propagation rules, and no seed chosen from GT").
    `eval_atlas_gt_clips.py`, which `run_track_conditions.py` imports for
    its list of clips, only re-exports `gt_clips` from the pilot evaluator's
    `eval_gt_clips.py`; the ported stage reads the population file instead.
    `get_sam`, five lines that paint the mask generator's masks into one
    map, is written again in the stage from what it does; the mask generator
    comes from the `segment-anything` package. `loaders.py` returns a black
    image for a missing frame; the port raises.
  - The input of one condition: `ipcai2027_experiment/scripts/make_t5_seeds.py`,
    which makes the seeds of the granularity result.
  - The paper's tables: `ipcai2027_experiment/atlas97/scripts/summary97.py`
    and `print_status_tables.py`, which the manuscript names as the source
    of every table; from `ipcai2027_experiment/scripts/`, `claims_grid.py`,
    `arms_paired.py`, `lovo_verdict.py`, which names the videos a verdict
    rests on, and `summarize_20.py`, the StereoMIS table; and
    `ipcai2027_experiment/task15_granularity/scripts/claims_table.py` and
    `settle_inputs.py`. `claims_table.py` and `claims_grid.py` read
    `outputs/atlas`, the 13-video set, and `outputs/cholec_gt`.
    `arms_paired.py` pairs arms the port leaves out; ported, the pairs it
    reads are the paper's. With them,
    `ipcai2027_experiment/task11_atlas13/scripts/check_provenance.py`, which
    makes no table: it checks each condition's `seed_info.json` against the
    settings the condition was meant to run with.
  - The camera trajectory on StereoMIS, from `ipcai2027_experiment/scripts/`:
    `stereomis_io.py`, `run_20.sh`, `pose_metrics.py` and `pose_controls.py`.
    `run_20.sh` runs the DA3 stage and, through
    `pi3_wrapper/scripts/queue_pi3_clips.sh`, the Pi3X stage with
    `--max-points 60000`. That flag thins the point cloud written to the
    GLB, and with it `glb_centroid`, `n_vertices` and `median_vertices`; the
    depth and the poses in the npz do not change.
  - D4D, from `ipcai2027_experiment/scripts/`: the front/behind relation,
    scored against D4D's structured-light surfaces, and the camera
    trajectory, scored against D4D's optical tracker.
    - `d4d_io.py` reads D4D through the upstream loader `d4d.loader`,
      which the port depends on and does not copy.
    - `d4d_census.py` and `d4d_population.py` fix the population from the
      inputs alone, before any score is read.
    - `d4d_depth.py` makes depth on the frames the GT was taken at, and
      `d4d_seed.py` makes the seed regions on the same frames.
      `d4d_seed.py` copies the tracking stage's seed step instead of
      importing it, and records the sha256 of `track_sam3.py`'s source,
      which no longer matches once the tracking stage is ported.
    - `d4d_predicate.py` scores the front/behind relation, and
      `d4d_verdict.py` compares it with the area floor.
    - `d4d_pose.py` scores the trajectory with `pose_metrics` and
      `pose_controls`.

    D4D's data is on the workbench's GPU machine only, so the D4D scripts
    are checked there.
  - What the viewer reads, and no score does: `export_viewer_dataset.py`
    and `build_temporal_graph.py`, which build the graphs;
    `export_instrument_mask.py`; `run_atlas_pipeline.sh`, which made the
    13-video tree the demo opens through a `?clip=` link (its catalogue
    reads `atlas97/`); `run_gt_tracked_export.sh`.
- **Wrappers.** `da3_surgery_wrapper/` and `pi3_wrapper/` become
  `recon3d_wrapper/`. Of `sam3_wrapper/`, the package moves: `get_device`,
  `load_sam3_video_tracker_model` and the part of `Sam3VideoInstanceSession`
  the tracker calls.
- **Rest of `surgical_core`.** `cholec/seg8k_align.py`'s search, which
  moves into the CholecSeg8k extraction; and, for the viewer, `pointcloud`,
  the six modules of `viewer` not yet ported (`graph_frame`,
  `hierarchy_frame`, `provenance`, `seg_frame`, `temporal_graph`,
  `text_frame`) and `cholec/cholect50.py`.
- **ATLAS-120k metadata, into `atlas120k_meta/`.** No video goes in.
  - Crop rectangles: `experiment/crop_necessity/verdicts/verdicts_latest.json`.
  - The 315-clip population: `ipcai2027_experiment/frozen/atlas97_clips.txt`.
    Clips of the release overlap in two videos. `tests/test_atlas120k_meta.py`
    pins the overlaps and checks that no two clips of the population share a
    frame. The population's reader, when it is ported, refuses two clips of
    one video that share a frame.
  - The depth manifest: `ipcai2027_experiment/frozen/atlas97_depth_manifest.json`.
  - The release's inventory, which the overlap check reads:
    `ipcai2027_experiment/task22_atlas100/out/manifest/videos.json`.
  - The readers: `surgical_core/atlas/clip_rects.py` and `frame_ratio.py`,
    into `surgical_core/atlas120k/`.
- **Viewer.**
  - The `demo`, `workbench` and `depthcmp` pages.
  - Their `src/` directories.
  - `viewer/scripts/merge_geometry_sources.py` and
    `verify_geometry_alignment.mjs`.
  - `vite.config.js`, without the experiment routes.

### Not carried for now

This section lists what the port leaves out for now. It is left out because
the port is short of time, not thrown away. Each entry says where its code is
in the workbench, so that a later pull request can port it. Git history keeps
what had been merged here before it was left out.

The first list holds experiments the manuscript reports. Their numbers come
from this repository once they are ported ("The paper is measured here"),
and not before.

- **LapEx** (`lapex_extract.py`, `lapex_kcurve.py` and `lapex_02b02c.py`,
  in `ipcai2027_experiment/scripts/`), the third population of the
  granularity result, which the manuscript reports and does not release.
  Its scores carry the earlier frozen evaluator's `eval_code_sha`
  (`9cf136c7…`), and `reeval_v2.py` lists `outputs/lapex` among the roots
  it scores again with the pilot evaluator. Whichever its scores carry,
  LapEx is scored again with the evaluator once ported.
- **Conditions seeded from GT masks**, which the ported tracking stage does
  not carry ("Two propagation rules, and no seed chosen from GT"):
  - `track_sam3.py --seed_source gt`, the `gt_seed` mode of
    `run_track_conditions.py`, which seeds on
    `track_metrics.pick_seed_frame` with the regions of
    `track_metrics.gt_instances`;
  - `t12_gtseed` on five inputs on ATLAS-120k, and on four inputs on
    CholecSeg8k, made by `ipcai2027_experiment/scripts/run_12_trackbase.sh`;
  - `t12_paste`, made by
    `ipcai2027_experiment/atlas97/scripts/make_paste_floor.py`, which copies
    `t12_gtseed`'s seed frame onto every frame of the clip and runs no
    tracker: the floor the propagated seeds are measured against.

  The manuscript's draft reports them: propagation over the pasted floor,
  in the abstract, and the loss of a geometry rendering under GT seeds.
  Whether they enter the paper, and from which code, is "Conditions seeded
  from GT" under "Open questions".

The second list holds what the paper's numbers do not use.

- **View consistency** (`view_consistency.py`, `camera_motion.py` and
  `export_03b_figs.py`, in `ipcai2027_experiment/scripts/`): whether a
  boundary stays on the same place of the tissue when the camera moves.
  Its one paired test, `normal_edge` against `rgb`, did not favour the
  geometry input; `depth` and `normal` were not tested against `rgb`. Every
  input's score was close to the score measured on the GT boundaries, its
  ceiling. These three are among the callers of the workbench's
  `geometry.backproject` ("Two orders of the world transform").
- **Blended inputs** of `ipcai2027_experiment/scripts/geom_blend.py`
  (`Terms`, `blend`, `SPECS` and the rest of the blend machinery), and the
  flattened RGB bases of `ipcai2027_experiment/scripts/rgb_flatten.py`
  (`rgb_flat_*`), measured on CholecSeg8k and the 13-video ATLAS-120k set.
  Both lost to the inputs they modified, and the manuscript does not report
  them.
- **Fifteen segmenter inputs.** Of the twenty modes of `sam_input_image`,
  the paper's conditions ran `rgb`, `depth`, `normal`, `normal_edge` and
  `rgb_edge`. The other fifteen (`rgb_refl`, `rgb_nedge`, `rgb_dedge`,
  `rgb_shade`, `rgb_edge_shade`, `refl_edge`, `refl_shade`,
  `refl_edge_shade`, `normal_shade`, `depth_edge`, `depth_shade`,
  `edge_only`, `shade_only`, `rgb_shading` and `rgb_normal`) ran only in
  sweeps: `ipcai2027_experiment/scripts/chain_arm_catalog.sh`,
  `chain_resume_arms.sh` and `chain_atlas_fusion.sh`, and `rgb_edge_shade`
  in `chain_kmatch.sh` and `chain_prop.sh` too. `chain_shortlist.sh` draws
  six of them and runs none. `arms_paired.py` pairs `rgb_refl` with `rgb`
  and with the `refl_*` arms, and the workbench's
  `ipcai2027_experiment/README.md` reports those pairs.

  With them go `pseudo_normal_from_rgb`, `relight_rgb`, `color_retinex` and
  the composition table. They are in the workbench's
  `depth_sam_tracking_experiment/geometry.py`.
- **Label transfer and warping**, in the workbench's
  `depth_sam_tracking_experiment/geometry.py`: `project_labels` and
  `project_labels_region`, which only `legacy/pipeline.py` calls, and
  `warp_labels`, which nothing calls.
- **The consensus and 3D variants of the tracker**,
  `depth_sam_tracking_experiment/track_sam3_consensus.py` and
  `track_sam3_3d.py`. `consensus` is a mode of `run_track_conditions.py`,
  and the manuscript reports no consensus condition.
- **Graphs built again**: `scripts/export_graph_cleanup_compare.py`, which
  builds each graph twice for a comparison page, and
  `scripts/backfill_depth_rel.py`, which adds `depth_rel` to graphs written
  before the builder wrote it.
- **ATLAS-120k files the paper does not read.**
  - The cut marks (`experiment/crop_necessity/marks/*.jsonl`), read by a
    review page. A clip with a cut is used whole.
  - The audits and clip lists of the earlier 96-video, 438-clip extraction
    (`ipcai2027_experiment/task22_atlas100/out/{manifest,audit}/`, all but
    `videos.json`).
- **Code paths inside the files that move.** Each was never run for a
  reported number, or never set off its default:
  - `track_sam3.py`: seeding from instrument masks (`--instrument_*`), the
    seed cache, `--seed_frame`, `--stride`, `--max_frames` above 0 (the
    operating point passes 0, every frame), `--seed_topk`, the choice of
    the seed frame from GT (`--seed_inst_thresh` below 1), and
    `--track_edge_gain` and `--track_no_smooth`, which every run left at
    their defaults;
  - `sam3d_core.py`: everything but `num_to_natural` and `get_sam`;
  - `sam3_wrapper/`: `scripts/`, `image_instance`, the concept session and
    `add_boxes`;
  - `depth_source.py`: the `da2` source and the bilateral `_bil` variants;
  - `preprocess` (`surgical_core/preprocess/endoscope_mask.py`, the
    package's one module): the border inpainting, the content-mask
    detection, the circle fit (`fit_endoscope_circle`,
    `circle_frame_inscribed_rect`) that found the rectangles
    `cholecseg8k_meta/` holds, and the p90-mask path
    (`video_endoscope_crop`, with `largest_inscribed_rect`). Every clip of
    the paper has its rectangle; a new video would need the fit. With it go
    `crop_cholec_frames.py`'s `--method p90mask`, `--reduce` and its
    fallback to a clip's own frames;
  - `extract_atlas_frames.py`, the paths `pipeline/extract_atlas120k.py`
    does not carry: the tile clips; the per-video rectangle it estimates
    from the frames (`detect_content_rect` and the three `refine_rect_*`
    steps), which a confirmed rectangle replaces on every clip that has
    one, and every clip of the population has one; reading the frames from
    the mp4 by number, which the script does only for a clip without the
    release's JPEGs or for one padded past its annotation
    (`--gt-pad-factor`); and `--gt-stride`, `--keep-duplicates`,
    `--dry-run` and `--no-clean`;
  - `run_per_frame_seg.py`: `--point_grids_dir`, `--frames gt`, `--smooth`.
- **`scripts/extract_cholec_frames.py`**, which no condition ran.

What stays behind is listed in `repo_migration_plan.md`, in the section on
what stays.

## Order, and what must hold before the next step

Which of these can move side by side, in separate branches and sessions,
while the evaluator is built and reviewed is worked out in `docs/workstreams.md`.

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
3. **Re-score.** CPU only. Score every condition's existing predictions with
   the evaluator. `condition_inventory` must report no mixed ruler and no
   missing condition.
4. **Port the toolkit onto it.** Each tool is done when its tests pass, and:
   - on the pilot's score JSONs, it writes the same bytes as the workbench
     version (the bootstrap is seeded). This holds for every tool. The
     pilot's JSONs lack the fields the evaluator now writes (class set, view,
     mode, input hashes, versions, propagation rule); a tool reads them all the
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
   the workbench byte for byte. The exceptions:
   - Pi3X's `runtime_sec`, per `repo_migration_determinism.md`.
   - The 14 CholecSeg8k clips whose gap frames the workbench's extractor
     placed 1 to 3 s late ("CholecSeg8k clips whose frames run out of
     order"), which the ported extractor converts or refuses, and which are
     then re-extracted and re-run.
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
   - Before anything is public, a check of HTL's anonymity rules, since AE-CAI
     is under double-blind revision.
   - Done when a third party can clone, install and get a green `pytest`,
     and, on a machine with a CUDA GPU, install the pipeline with the
     README's steps and run each ported stage on a clip.

The data these steps read stays in the workbench:

- `outputs/atlas97` and `outputs/cholec_gt`, which the paper reads;
- `outputs/atlas`, the 13-video set, which `claims_table.py` and
  `claims_grid.py` read, which the demo opens through a `?clip=` link, and
  which holds the clips the determinism measurement used
  (`adrenalectomy__16GPCUPkXYQ__gt_0004` and
  `adrenalectomy__16GPCUPkXYQ__tile_0007`) and is where step 5 compares
  stages, with one CholecSeg8k clip from `outputs/cholec_gt`;
- `outputs/stereomis` and `outputs/stereomis_masked`, which `run_20.sh`
  reads and writes;
- `outputs/d4d_pose` and `ipcai2027_experiment/out/09`, which the D4D
  scripts read and write;
- `outputs/lapex`, which is not released;
- the determinism measurement's JSON (`measure_determinism.py --json-out`);
- the predictions under `ipcai2027_experiment/atlas97/out/`;
- the pilot's score JSONs.

Every command takes those paths as arguments.

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
   - how time is counted, in observations as the workbench did or in
     seconds, since the GT frames are not evenly spaced;
   - which of the two pre-registered `hold_mean` definitions keeps the name
     ("No measure over time carries a star");
   - what hold means under forward propagation (`forward_from_first`),
     where its time offsets from the seed frame run one way only; the
     workbench claimed this changes the measure and did not measure it.
2. **The skill-classification code and the other 18 viewer pages.** The plan
   leaves both behind. The paper's figures come from some of those pages.
3. **Whether step 5 has to finish before submission.** Step 3 gives the
   numbers of every condition the workbench ran. The `forward_from_first`
   conditions are new measurement in step 5, so a row that reports them
   waits on it. A stage that passes step 5's byte check cannot change the
   step-3 numbers;
   the 14 clips re-extracted under "CholecSeg8k clips whose frames run out
   of order" can, and so can depth made again under "Depth made by two
   versions of the depth stage".
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
    only, as `--drop-video` says.
10. **The edge ring as a process-wide flag.**
    `surgical_core.geometry.normals.EDGE_MASK_RING` decides whether the
    contour around the image border and around invalid depth is zeroed in the
    edge map the segmenter is prompted with. It is a module global, set for a
    whole process: the tracking stage, `geom_blend.py` (and through it the
    per-frame segmentation stage) and `d4d_seed.py` read it, and each writes
    it into its own provenance record. Whether a setting passed per run, and
    recorded with the output in one form, replaces the flag is decided before
    either stage is ported.
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
12. **The seed frame chosen from GT.** With `--seed_auto`, the tracking stage
    seeds on the frame nearest the window's centre among those whose GT masks
    call at most `--seed_inst_thresh` of it instrument (0.005 by default), or,
    when there is none, on the frame with the least. It reads each frame's
    mask file through the workbench's class tables, by the file's presence
    rather than the GT flag, and counts a frame with no mask as free of
    instruments. It was added to keep an instrument in the seed frame from
    splitting one surface into two tracks. With `--seed_inst_thresh 1.0`, the
    operating point, every frame counts as free and the seed is the centre of
    the strided frame list, frame N // 2 at stride 1, whatever the GT says;
    the ported stage keeps that centre and nothing else of the choice ("Two
    propagation rules, and no seed chosen from GT"). `seed_info.json` records
    the seed frame but not the rule. What stays open is the conditions
    seeded from GT masks (`--seed_source gt`). They seed on
    `track_metrics.pick_seed_frame`, the frame nearest (N − 1)/2 among those
    with a mask file, and take the seed's regions from
    `track_metrics.gt_instances`, which drops a component under `MIN_AREA`.
    That frame can lie several frames from the centre: on the nine 30-frame
    CholecSeg8k clips of the development machine, counting the annotated
    frames only, it is frame 10, 14 or 16 where the operating point seeds at
    15, and it moves with the mask files present. Such a condition cannot
    seed on the centre when the centre has no GT, so it holds neither rule.
    Decide, before the tracking stage is ported, whether these conditions
    stay, under which rule a score records them, and whether their seed keeps
    the `MIN_AREA` cut that "No minimum object size, anywhere" removes
    everywhere else.
13. **Two orders of the world transform.** `cam_to_world` computes
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
14. **Whether the geometry path has to be fast.** The depth stage
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
15. **Conditions seeded from GT.** A seed frame is scored like any other
    frame, because the paper's conditions are seeded from the pipeline's own
    masks. Which of the 38 conditions were seeded from GT instead, and
    whether such a condition is scored on its seed frame or enters a table
    at all, is settled before step 3, on the machine that holds the
    predictions. The check against the pilot evaluator in step 2 needs the
    list sooner: it leaves out by name every condition that holds neither
    propagation rule (`pilot_check --leave-out`). The `seed_source` that each
    condition's `seed_info.json` records says where its seed came from;
    where it does not tell, the command that made the condition does.
    The workshop's oracle row, GT instrument masks painted onto a
    condition's labels, is one; the viewer's `gt_tracked` track, one GT
    frame carried by SAM 3, is another candidate. What the tracking stage
    does with such a seed is "The seed frame chosen from GT".
16. **Masks that are not GT under the GT's name.** The viewer's step writes
    SAM 3 masks into `seg_masks/` as `<i>_color_mask.png`, told apart from
    the annotation only by the frame manifest's `is_anchor` and
    `seg_provenance`, and the two VID25 clips still hold such masks from
    before their re-extraction, on frames the frame manifest marks as having
    none. The evaluator reads the flags and refuses the VID25 clips until
    those files are removed. When the pipeline is ported (step 5), decide
    whether a mask that is not annotation moves out of `seg_masks/` or takes
    a name of its own, so that the distinction is in the file and not only in
    the frame manifest.
17. **Which commit of Depth Anything 3 the `recon3d` extra pins.** The extra
    names the repository at its head, so two installs can get two versions.
    The commit to pin is the one the workbench ran on its GPU machine. pip
    recorded it there, in the `direct_url.json` of that install. The extra
    is pinned once that record has been read.
18. **A constraints file from the GPU machine.** The pipeline was measured
    with the package versions on the workbench's GPU machine. A constraints
    file lists them, so that `pip install -e ".[recon3d]" -c <file>` gives
    another machine the same versions. Once that machine's environment has
    been read, the file is written from it and added beside `pyproject.toml`.
    Until then the extra alone says what a machine needs.
19. **A clip with no usable depth in any frame.** The point-cloud stage
    skips a frame with no usable depth, as the workbench does, because the
    data can hold such a frame. A clip with no usable depth in any frame
    gets no cloud, no manifest entry and a count of 0, and the run exits 0.
    No real clip has done this; a bundle with no depth at all is more likely
    a broken bundle than data. Decide whether the stage refuses such a clip.
20. **Cuts inside GT clips.** The workbench searched for scene changes only
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
