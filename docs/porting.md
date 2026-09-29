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
of the toolkit here.

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
   fallback is withdrawn. It said that if re-scoring and re-running were not
   done by 2026-09-29, the paper would use the workbench's numbers.
2. **The pilot evaluator is not copied here.** It stays frozen in the
   workbench, `eval_code_sha = 1f8a813a…`. The check against it runs there,
   taking paths as arguments. `docs/evaluation.md` describes the check, under
   "Checked against the pilot evaluator".
   - So the constraints the frozen files imposed no longer bind. Those were: the
     import paths `surgical_core.cholec` and `surgical_core.atlas`, `evalkit/`
     as a script directory at the repository root, `sam3_wrapper/` on
     `PYTHONPATH` by name, and a copy of the older `eval_track.py` as a test
     fixture.
3. **AE-CAI stays in the workbench.** Its numbers are reproduced by the tag
   `paper/aecai2026-endolina-final` and the code at that tag. Nothing here
   reproduces them, and no tag of that name is created here.
4. **No version labels.** The earlier evaluator is *the pilot evaluator*, and
   the new one is *the evaluator*.
5. **Names.** The project is `omnisurg-4d-scene-graph`, and the ATLAS-120k
   module is `atlas120k`.
6. **English only.** Workbench code comes with Japanese comments. Each file is
   rewritten in English when it is ported, not afterwards.
7. **No personal email address.** `tests/test_no_personal_email.py` checks the
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
| hand-derived test scenes | — | `archive/metric-guide` (`evalkit/tests/scenes.py`, `cartoon.py`) | Reuse the scenes. The branch's tests check the pilot evaluator, so their expected values are derived again under the new rules. |

### The toolkit around it

| file | from the workbench | reviewed | work |
|---|---|---|---|
| `paired_stats.py` | `ipcai2027_experiment/scripts/paired_stats.py` | `extract/03-metrics` (#3) | `VERDICT_RULE` does not change. The comparability check now comes from the evaluator, not the pilot's `check_comparable`. `video_of` is defined here rather than delegated. |
| `compare_eval.py` | `depth_sam_tracking_experiment/compare_eval.py` | `extract/03-metrics` (#3) | It refuses to mix shas through the evaluator's check. |
| `track_metrics.py` | `depth_sam_tracking_experiment/track_metrics.py` | `extract/03-metrics` (#3) | It imports `BACKGROUND`, `_gt_idmap` and `_load_depth` from the pilot's `eval_track`; they come from the evaluator instead. Waits on open question 1. |
| `surgical_core/clip_time.py` | `surgical_core/clip_time.py` | `extract/03-metrics` (#3) | English only. |
| `kmerge.py` | `ipcai2027_experiment/scripts/kmerge.py` | `extract/04-kmerge` (#5) | It imports the pilot's `eval_track`, and borrows `verdict` and `video_of` from `paired_stats`. Waits on open question 1. |
| `surgical_core/geometry/` | `depth_sam_tracking_experiment/geometry.py` | `extract/04-kmerge` (#5) | English only. Shared by the pipeline and the toolkit. |
| `condition_inventory.py` | `ipcai2027_experiment/scripts/condition_inventory.py` | `extract/05-inventory` (#4) | It reads `eval_code_sha` and `eval_version` from score JSONs; it reads the evaluator's fields instead. |
| `check_env.py` | `ipcai2027_experiment/atlas97/scripts/check_env97.py` | `extract/05-inventory` (#4) | It computes the sha through the pilot's `eval_gt_clips`. It exists because frozen files could not check their own import path. The evaluator can, so decide whether this tool is still needed. |
| `reeval_diff.py` | — (written in the earlier repository) | `extract/06-rescore` (#6) | Becomes the check against the pilot evaluator, run in the workbench. |

Tests come with the file they test:

- `test_paired_stats_bootstrap.py` comes with `paired_stats`.
- `test_eval_identity.py` comes with `track_metrics`. Its part that pins the
  sha over the GT loaders changes with the evaluator.
- `test_geom_edge_ring.py` comes with `geometry`.

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
  - `scripts/`: `run_cholec_depth.py`, `regen_cholec_glb.py`, the frame
    extraction and crop scripts, `export_viewer_dataset.py`,
    `build_temporal_graph.py`, `export_instrument_mask.py`,
    `run_atlas_pipeline.sh`, `run_gt_tracked_export.sh`,
    `measure_determinism.py`.
  - `depth_sam_tracking_experiment/`: `track_sam3.py`, `sam3d_core.py`,
    `loaders.py`, `depth_source.py`, `viz_common.py`, `sam_env.py`.
  - `ipcai2027_experiment/scripts/`: `run_per_frame_seg.py`, `geom_blend.py`.
  - `pi3_wrapper/scripts/run_pi3_depth.py`.
- **Wrappers.** `da3_surgery_wrapper/` and `pi3_wrapper/` become
  `recon3d_wrapper/`, and `sam3_wrapper/` moves as it is.
- **Rest of `surgical_core`.** `pointcloud`, `preprocess` and `viewer`, plus
  the non-evaluation parts of `cholec` and `atlas`.
- **ATLAS-120k metadata, into `atlas120k_meta/`.** No video goes in.
  - Crop rectangles: `experiment/crop_necessity/verdicts/verdicts_latest.json`.
  - Cut marks: `experiment/crop_necessity/marks/*.jsonl`.
  - The 315-clip population: `ipcai2027_experiment/frozen/atlas97_clips.txt`.
  - The depth manifest.
  - The 100-video manifest and audit.
  - The readers: `clip_rects.py` and `frame_ratio.py`.
- **Viewer.**
  - The `demo`, `workbench` and `depthcmp` pages.
  - Their `src/` directories.
  - `merge_geometry_sources.py` and `verify_geometry_alignment.mjs`.
  - `vite.config.js`, without the experiment routes.

What stays behind is listed in `repo_migration_plan.md`, in the section on
what stays.

## Order, and what must hold before the next step

1. **Settle the open questions that touch the evaluator** (1 and 2 below).
   Whatever goes into `eval_code_sha` has to be decided before it is frozen.
2. **Build the evaluator.** CPU only. Class tables, then metrics, then the entry
   point. It is done when:
   - the hand-derived tests pass;
   - in the pilot-equivalent configuration, the evaluator reproduces every key
     it shares with the pilot evaluator, at zero tolerance, on the 38 scored
     conditions.
   Then freeze it and record its sha.
3. **Port the toolkit onto it.** Each tool is done when:
   - its tests pass;
   - on the pilot's score JSONs, it writes the same bytes as the workbench
     version (the bootstrap is seeded).
   `compare_eval` also has to refuse a mix of shas, shown by a self-test.
4. **Re-score.** CPU only. Score every condition's existing predictions with the
   frozen evaluator. `condition_inventory` must report no mixed ruler and no
   missing condition. These are the paper's numbers.
5. **Port the pipeline, one stage at a time.** Each stage's output must match
   the workbench byte for byte. The one exception is Pi3X's `runtime_sec`, per
   `repo_migration_determinism.md`. If the packages this repository resolves
   differ from the ones recorded there, measure determinism again first. The
   315 clips also get DA3 and `glb_centroid`: the demo's reference grid stays
   DA3.
6. **Prepare the release.**
   - An English README, `docs/data_contract.md`, the `atlas120k_meta/` README
     and `CITATION.cff`.
   - Before anything is public, a check of HTL's anonymity rules, since AE-CAI
     is under double-blind revision.
   - Done when a third party can clone, install and get a green `pytest`.

The data these steps read stays in the workbench:

- `outputs/atlas97` and `outputs/cholec_gt`;
- the predictions under `ipcai2027_experiment/atlas97/out/`;
- the pilot's score JSONs.

Every command takes those paths as arguments.

## Open questions

1. **Identity metrics and merge cost.** `docs/evaluation.md` does not cover
   `track_metrics` (hold, IDF1, IDsw, Frag) or `kmerge`. Decide whether they are
   part of the evaluator and its sha, and define `hold_mean`'s denominator. In
   the workbench, a change to that denominator once flipped a result's sign.
2. **300 or 400 px.** The evaluator's `MIN_OBJECT_PX` is 300.
   `track_metrics.py` uses a `MIN_AREA` of 400, the same value as the
   pipeline's seed cutoff. The workbench has left that mismatch open for a long
   time. Settle it together with question 1.
3. **The skill-classification code and the other 18 viewer pages.** The plan
   leaves both behind. The paper's figures come from some of those pages.
4. **Whether step 5 has to finish before submission.** Step 4 already gives
   the numbers, and the determinism result says step 5 cannot change them.
5. **Whether `evalkit/` becomes a package.** Nothing frozen forces it to be a
   script directory any more.
