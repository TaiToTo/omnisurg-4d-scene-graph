# Workstreams: what can be ported side by side

**Status: in progress.** `docs/porting.md` gives the order of the port. This
document says which pieces of it can move at the same time, in separate
branches and separate sessions, while the evaluator is being built and
reviewed. It is deleted with `docs/porting.md` once the port is done.

The order in `docs/porting.md` binds three things: nothing is re-scored
before the evaluator is checked against the pilot evaluator; the evaluator
itself is built in sequence, class tables, then metrics, then the entry
point; and a pipeline stage is done only when its output equals the
workbench's, byte for byte, which only the GPU machine can show. Everything
else that neither reads a score nor imports the evaluator can move now. This
document lists what that is.

## Where a workstream can run

Each piece has a source, and the source decides where the work can be done:

- **W, the workbench.** `$OMNISURG_SOURCE`, on one machine, never committed.
  Only a session on that machine can read it.
- **E, the earlier repository.** `TaiToTo/omnisurg-4dsg` (private, on
  GitHub). A session anywhere with access to that repository can read it. Its
  branches hold the versions that were already reviewed there; `docs/porting.md`
  names the branch per file.
- **G, the GPU machine.** The machine the workbench's pipeline runs on. It
  holds the pipeline's outputs, and the determinism measurement was made on
  it. A stage's byte check runs only there.

Where a file exists in both W and E, start from E: that is the reviewed
version. Then diff it against W before opening the pull request, so that a
fix made in the workbench after the review is not lost. A workstream whose
source is W only has to run on the machine that has the workbench.

## Rules every workstream follows

These repeat what `AGENTS.md`, `docs/review.md` and `docs/porting.md` already
say, in the form a session starting cold needs.

- **Branch from `main`**, never from a branch under review. One concern per
  branch, about five changed files, one pull request. Branch names are
  `<area>/<thing>`, as the history shows (`evalkit/class-table-cholecseg8k`,
  `docs/porting-plan`).
- **Rewrite in English as you port**, code, comments, docstrings and test
  names alike. The workbench's test functions carry Japanese identifiers
  (`test_cholec_は_timestamp_sec_を使う`); they are renamed, not kept.
- **A comment gives the reason, not a reference.** The workbench's docstrings
  cite task ids and audit sections (`task22 の T4`, `§3.7 R7`, viewer design
  documents). Each of those is replaced with the reason itself, or dropped.
- **Tests come with the file.** Every workstream below names its tests. A
  ported test that passes before the module is ported is not testing the
  port; run it against the new module and read the failures.
- **No machine paths, no personal addresses.** Set the repository-local
  `user.email` to the GitHub no-reply address before the first commit;
  `tests/test_no_personal_email.py` checks the history. Data files from the
  workbench are grepped for `/home/`, `/var/autofs/` and `@` before they are
  committed.
- **Dependencies stay what `pyproject.toml` declares** (`numpy`,
  `opencv-python`, `Pillow`). A module that needs more is either split so that
  the part the toolkit uses does not, or the dependency is added as an
  optional extra in the same pull request, with the reason in the file.
- **Run the tests from a venv**, not the system interpreter, with both
  extras. Without them the tests of `surgical_core.geometry.render` and of
  the tools' statistics and chart are skipped, and the run still looks green.

  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install -e ".[render,tools]" pytest
  pytest
  ```

## Not now: blocked on a decision or on another stage

`track_metrics` and `kmerge` wait on a decision not yet made; started before
it, they would be ported twice. The pipeline's later stages wait on what
they borrow and on the stages before them.

| piece | waits for |
|---|---|
| `track_metrics`, `kmerge` | open question 1 in `docs/porting.md` |
| the tracking stage: `track_sam3.py`, `sam3_wrapper/` and their helpers | the depth stages below, and what it borrows (below the table) |
| the export stages and the viewer | the tracking stage, whose output they read |

The tracking stage takes its seed frame and its GT instances from the pilot
evaluator's code (`track_metrics.pick_seed_frame`, `gt_instances`,
`eval_track._gt_idmap`) and its class tables from the workbench's
`surgical_core.cholec` and `atlas`. Ported, it reads `evalkit.classes`, and
which frame seeds a clip is decided by `docs/evaluation.md`, not borrowed
from a metric. It also reads the edge-ring setting from a process-wide flag
in `surgical_core.geometry.normals`. Whether a setting passed per run, and
recorded with the output, replaces the flag is decided before the stage is
ported: it changes what the output records, and every later stage would read
the flag too.

Open question 1 is not a branch. It is a decision, and `docs/evaluation.md`
("Consistency over time") lists what deciding it means. A session can prepare
it by writing down, for each candidate, the GT-track definition it needs and
how that coexists with the whole-class object; the decision itself is made by
the author, in `docs/evaluation.md`.

## Now: workstreams that can start today

| branch | brings | source | runs where |
|---|---|---|---|
| `evalkit/metric-guide` | a page that shows each metric on the test scenes | E `archive/metric-guide` | anywhere |
| `docs/short-headers` | the module headers of `main` brought under the cap of `docs/review.md` | this repository | anywhere |

### `evalkit/metric-guide`

**Source.** E `archive/metric-guide`: `docs/metrics/make_guide.py`, which
drew the scenes and the cartoon and printed what the pilot evaluator computed
on them. Its output is `docs/pilot_metric_guide.html` here. The script itself
calls the pilot evaluator's functions through `v2_import.py`, so it cannot run
here as it is.

**What it is.** A page, built from `tests/scenes.py` and `tests/cartoon.py`,
that shows every metric next to the picture it is computed on: for each
scene, the GT, the prediction, and the value each metric gives it, so that
"what does a 3 px shift cost `SQ`" or "what does a region on background do to
`F1_50`" is answered by looking rather than by reading the formula. It is a
debugging aid as much as documentation: when a score on real data looks off,
the page says which controlled fault produces that behaviour.

**Work.** Rewrite `make_guide.py` against `evalkit`'s evaluator, in English,
under `docs/` or `evalkit/`. Each scene's numbers come from the evaluator at
build time, never typed in; a test asserts that the numbers on the page equal
the values the hand-derived tests pin, so the page cannot drift from the
tests. Where normal mode and pilot mode differ on a scene, show both values
and name the rule in `docs/evaluation.md` that separates them: that makes the
"Why the pilot evaluator was replaced" list visible on pictures. Keep the
cartoon for the overview and the small scenes for the per-metric appendix, as
the pilot page does. Decide, in the pull request, whether the page is
committed or built in CI.

**Done when.** The page regenerates from the two helpers with one command,
its numbers equal the tests' pinned values, both modes appear where they
differ, and nothing in the guide is a third copy of a scene.

### `docs/short-headers`

**Source.** This repository. `tests/test_module_headers_are_short.py` lists
in `STILL_LONG` the modules whose header was over the cap when the rule was
made; `docs/review.md` says what a header holds and where the rest goes.

**What it is.** Each listed module's header is cut to what the module does,
in at most twelve lines, plus its Usage. History goes to the commit that
removes it, with the reason it was there, so that git keeps it; a definition
of a term its area's specification has is dropped, and one it lacks is
added there in one line (`docs/evaluation.md` for the evaluator and its
tools, the README of `atlas120k_meta/` for its readers); a reason for a
choice moves to a one- or two-line comment at the choice, or is dropped when
no reader would "fix" the choice without it. A module whose paragraph needs
"and" is split. The module is then removed from `STILL_LONG`, which the test
refuses to keep it in once it is short.

**Order.** The modules under `evalkit/` that `code_sha.HASHED_MODULES`
names go first, before the evaluator is frozen: after the freeze not a byte
of them changes, docstrings included. The function docstrings of a hashed
module are read in the same commit as its header: history, a second
definition and a long reason leave them too, and go where `docs/review.md`
sends them. `evalkit/pilot.py` is not shortened: it is removed before the
freeze ("Pilot mode is removed before the freeze" in `docs/porting.md`), and
leaves `STILL_LONG` then. A few modules per pull request, each pull request one
area (`evalkit/`, `evalkit/tools/`, `surgical_core/`, `tests/`).

**Done when.** `STILL_LONG` is empty and the suite passes; the list and the
tests that keep it are then deleted, and the cap alone remains.

## Ported anywhere, checked on G: the depth stages

The pipeline moves one stage at a time (step 5 of `docs/porting.md`). A
stage is checked by running it and the workbench's stage on the same clip
and comparing every file the two write. The check needs G and no evaluator,
so these workstreams come in two parts:

- **Anywhere:** the port, in English, with CPU tests that drive the stage
  through a stand-in model. It is reviewed and merged like any other branch.
- **On G:** the byte check of the merged stage, on the two clips the
  determinism measurement used, `adrenalectomy__16GPCUPkXYQ__gt_0004` (15
  frames) and `tile_0007` (120 frames). A difference it finds is fixed in a
  pull request of its own. The stage is done when the check passes.

| branch | brings | source | checked on G |
|---|---|---|---|
| `pipeline/byte-check` | the tool that runs a stage on a fresh copy of a clip's inputs and compares two runs file by file | W `scripts/measure_determinism.py` | is the check |
| `surgical-core/camera` | the camera transforms, the depth colormap, the GLB writer and the endoscope mask the two stages share | W `surgical_core/` | through the stages |
| `recon3d/da3` | `recon3d_wrapper/` with DA3 behind it, and the depth stage | W `da3_surgery_wrapper/`, `scripts/run_cholec_depth.py` | not yet |
| `recon3d/pi3x` | Pi3X behind the same interface, and its stage | W `pi3_wrapper/` | not yet |

The order is the table's: the stages use the tool and the helpers, and Pi3X
fits the interface DA3 gave. `scripts/regen_cholec_glb.py`, which records
`glb_centroid` after the depth stage, follows `recon3d/da3` as a branch of its
own.

Rules for these four, besides the ones above:

- **The output stays the workbench's, file for file**: names, formats, the
  manifest's keys and their order. A change to what a stage writes is a
  change to the data contract, made after the stage's check has passed,
  never with the port.
- **Depend on the models, never copy them.** `depth-anything-3` and `pi3`
  come from their git repositories at the commits G ran; the rest at the
  versions the determinism measurement recorded (`torch` 2.10.0, `numpy`
  1.26.4, `transformers` 5.8.1, `opencv-python` 4.11.0.86, `trimesh`
  4.11.3). All of it is an optional extra, `recon3d`; the base install does
  not change. The commits are read on G; until then the extra names the
  repositories, and the pull request says so. DA3 needs `numpy` below 2.
- **CI needs no torch and no weights.** The model sits behind the
  interface, and the CPU tests drive the stage through a stand-in that
  returns fixed depth. A test that needs torch skips without the extra, as
  the render tests do; one that needs the weights runs on G.
- **This repository's helpers replace the workbench's copies**, and the byte
  check says whether that changed anything. One difference is known:
  `surgical_core.geometry.valid_depth_mask` keeps depth above `DEPTH_MIN`
  (1e-6), the workbench's `pointcloud.valid_depth_mask` above 0.

### `pipeline/byte-check`

**Source.** W `scripts/measure_determinism.py` (606 lines) and
`tests/test_measure_determinism.py` (366 lines). Standard library and
`numpy`.

**What it is.** It gives a stage a fresh working copy of a clip's inputs
(images and masks linked in, a manifest stripped of what the stage writes),
collects everything else the stage writes under that copy, and compares two
runs byte for byte, reporting how large a difference is. Pi3X's
`runtime_sec` is the one field allowed to differ, and a file that differs
only there is counted apart, never as equal. In the workbench it runs one
stage twice; here it runs the workbench's stage and this repository's on the
same inputs.

**Work.** Port the working copy, the collection and the comparison as they
are, with their tests, in English. The workbench's table of stages becomes
two commands passed in, one per side, so that this repository names no
workbench path. Each run records its environment, as now.

**Done when.** The ported tests pass; a planted one-ulp difference, a
missing file and a changed manifest field are each reported; nothing in the
tool names a workbench path.

### `surgical-core/camera`

**Source.** What the two stages import from W's `surgical_core`:
`geometry/__init__.py` (89 lines: `backproject_depth`, `cam_to_world`,
`world_to_gltf`, `project_world_to_pixel`) with `tests/test_cholec_geometry.py`
(257 lines); `viewer/depth_vis.py`, `glb.py` and `camera_axes.py`;
`preprocess/endoscope_mask.py` (275 lines).

**What it is.** The transforms from pixels to the viewer's glTF space, the
depth colormap, the point-cloud writer, and the mask of the endoscope's field
of view, outside which the depth stage fills the frame in before DA3 sees it
(off for ATLAS-120k).

**Work.** W's `surgical_core.geometry` is not the package of that name here,
which came from the tracking experiment; its transforms join that package as
one module, and where `surgical_core.geometry.project` already back-projects,
one of the two definitions goes. The viewer modules join
`surgical_core.viewer`, and `preprocess` comes as it is. The colormap's
`matplotlib` is the `render` extra's.

**Done when.** The ported tests pass from the package, and each function has
one definition.

### `recon3d/da3`

**Source.** W `da3_surgery_wrapper/da3_surgery_wrapper/inference.py` (83
lines) and `scripts/run_cholec_depth.py` (340 lines). The wrapper's other
scripts and its documents, in Japanese, stay behind; what a reader needs from
them goes into `docs/pipeline.md`.

**What it is.** The first stage: depth for every frame of a clip from DA3,
an image of it, and a point cloud per frame. It writes `depth_raw/*.npy`,
`depth_vis/*.jpg`, `exports/mini_npz/results.npz` and `pc_vis/frame_*.glb`.

**Work.** `recon3d_wrapper/` holds the interface a reconstruction model
answers to, with DA3 as its first backend; the stage that drives it goes
under `pipeline/`. The `recon3d` extra comes in this branch.

**Done when.** Anywhere: the CPU tests pass, through the stand-in, on what
the stage writes and on the manifest's keys. On G: every file of both clips
equals the workbench's.

### `recon3d/pi3x`

**Source.** W `pi3_wrapper/pi3_wrapper/inference.py` (381 lines),
`pi3_wrapper/scripts/run_pi3_depth.py` (468 lines), and the tests in
`pi3_wrapper/tests/` (`test_inference.py`, `test_manifest_merge.py`,
`test_roundtrip.py`), which need torch but no weights.

**What it is.** Pi3X reconstructs depth and the camera's poses together. The
stage writes `exports/mini_npz/results__pi3x.npz`, `pc_vis/frame_*__pi3x.glb`
and `depth_vis/*__pi3x.jpg`, and adds `geometry_sources.pi3x` to the
manifest, with its `runtime_sec`.

**Work.** The second backend behind the same interface. If the poses do not
fit the interface DA3 gave, the interface changes in this branch, and DA3's
tests show that the change costs DA3 nothing.

**Done when.** Anywhere: the ported tests pass. On G: every file of both
clips equals the workbench's, `runtime_sec` aside.
