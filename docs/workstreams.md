# Workstreams: what can be ported side by side

**Status: in progress.** `docs/porting.md` gives the order of the port. This
document says which pieces of it can move at the same time, in separate
branches and separate sessions, while the evaluator is being built and
reviewed. It is deleted with `docs/porting.md` once the port is done.

The order in `docs/porting.md` binds two things: nothing is re-scored before
the evaluator is checked against the pilot evaluator, and the evaluator
itself is built in sequence, class tables, then metrics, then the entry
point. It also says when a pipeline stage is done: when its output equals
the workbench's, byte for byte, apart from the exceptions it names, which in
full only the GPU machine can show. Everything else that neither reads a
score nor imports the evaluator can move now. This document lists what that
is.

## Where a workstream can run

Each piece has a source, and the source decides where the work can be done:

- **W, the workbench.** `$OMNISURG_SOURCE`, never committed. Only a session
  on a machine that holds a copy can read it: the development machine, and G.
- **E, the earlier repository.** `TaiToTo/omnisurg-4dsg` (private, on
  GitHub). A session anywhere with access to that repository can read it. Its
  branches hold the versions that were already reviewed there; `docs/porting.md`
  names the branch per file.
- **G, the GPU machine.** The machine the workbench's pipeline runs on. It
  holds a copy of the workbench, the pipeline's outputs and the model
  weights, and the determinism measurement was made on it. A stage's check
  on the GPU, the one that says the stage is done, runs only there.

Where a file exists in both W and E, start from E: that is the reviewed
version. Then diff it against W before opening the pull request, so that a
fix made in the workbench after the review is not lost. A workstream whose
source is W only has to run on a machine that holds the workbench.

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
- **Run the tests from a venv**, not the system interpreter, with the
  `render` and `tools` extras. Without them the tests of
  `surgical_core.geometry.render` and of the tools' statistics and chart are
  skipped, and the run still looks green. The `recon3d` extra is not needed:
  the CPU tests drive the pipeline through a stand-in model.

  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install -e ".[render,tools]" pytest
  pytest
  ```

## Not now: blocked on a decision

`track_metrics` is a tool outside the evaluator ("No measure over time
carries a star"). It waits on a decision not yet made, its GT track: ported
before the decision, it would be ported twice. The stages after tracking
wait on the tracking stage. The questions are named as `docs/porting.md`
heads them.

| piece | waits for |
|---|---|
| `track_metrics` | "The GT track of `track_metrics`" |
| the export stages and the viewer | the tracking stage, whose output they read |

The ported tracking stage reads no GT. Its propagation rule places the seed
on the middle frame or on frame 0, and no seed comes from GT masks. Its check links
the workbench's stored `results.npz` in, so it does not wait on the depth
stages. Where its record holds the ring setting of the tracker's input
stays open ("The edge ring of the tracker's input").

## Now: workstreams that can start today

| branch | brings | source | runs where |
|---|---|---|---|
| `evalkit/metric-guide` | a page that shows each metric on the test scenes | E `archive/metric-guide` | anywhere |
| `docs/short-headers` | the module headers of `main` brought under the cap of `docs/review.md` | this repository | anywhere |

The depth stages can start today too. Their port runs anywhere and their
check needs W and G; they have a section of their own, at the end.

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

## Ported anywhere, checked on W and G: the depth stages

The pipeline moves one stage at a time (step 5 of `docs/porting.md`). A
stage is checked by running it and the workbench's stage on the same clip,
in one environment, and comparing every file the two write. That needs no
evaluator, so these workstreams do not wait on it. Each comes in up to
three parts:

- **Anywhere:** the port, in English, with CPU tests that drive the stage
  through a stand-in that returns fixed depth in place of the model. It is
  reviewed and merged like any other branch.
- **On the development machine, on CPU, for DA3 only:** the byte check with
  the real model on both sides, on `adrenalectomy__16GPCUPkXYQ__gt_0004` and
  the CholecSeg8k clip (below), each cut to its first four frames. It runs
  everything the DA3 stage does, the wrapper and the model included. Four
  frames take about a minute and up to 6 GB there, and both grow faster than
  the frame count, so a whole clip does not fit in the machine's 16 GB. The
  workbench's stage runs twice on each clip first, to show that the CPU
  gives the same bytes twice; cut to four frames, both clips did, every
  file. DA3 declares `xformers`, which does not install everywhere;
  DA3-LARGE's layers do not use it, so this environment may go without it.
  Pi3X skips this check: a CPU run of it is impractically slow, as the
  workbench's wrapper notes.
- **On G:** the byte check on the GPU, on all three clips. It adds what only
  a GPU runs: the precision each model chooses from the GPU (bfloat16 where
  the GPU supports it, float16 otherwise) and the GPU's kernels. A
  difference it finds is fixed in a pull request of its own. The stage is
  done when this check passes.

The three clips are the two the determinism measurement used,
`adrenalectomy__16GPCUPkXYQ__gt_0004` (15 frames) and
`adrenalectomy__16GPCUPkXYQ__tile_0007` (120 frames), and one CholecSeg8k
clip that both machines hold and that "CholecSeg8k clips whose frames run
out of order" in `docs/porting.md` does not name. The measurement ran no
CholecSeg8k clip, so on G the workbench's stage runs twice on it first.

The workbench's side is its stage at the commit the determinism measurement
recorded (`git_head` in its JSON), checked out clean; the code of these
stages has not changed since. That is not always the code the stored outputs
were made with (`docs/porting.md`, "Depth made by two versions of the depth
stage").

| branch | brings | source | checked by |
|---|---|---|---|
| `pipeline/byte-check` | the tool that runs two commands on fresh copies of a clip's inputs and compares what they write, file by file | W `scripts/measure_determinism.py` | its tests, each planting a fault |
| `surgical-core/camera` | the transforms from pixels to the viewer's space, the depth colormap, the GLB writer and the camera axes | W `surgical_core/` | the stages' byte checks |
| `recon3d/da3` | `recon3d_wrapper/` with DA3 behind it, the depth stage, and the install steps | W `da3_surgery_wrapper/`, `scripts/run_cholec_depth.py` | its byte check |
| `recon3d/pi3x` | Pi3X behind the same interface, and its stage | W `pi3_wrapper/` | its byte check, `runtime_sec` aside |

The order is the table's: the stages use the tool and the helpers, and Pi3X
fits the interface DA3 gave. `scripts/regen_cholec_glb.py`, which records
`glb_centroid` and the camera axes after the depth stage, follows
`recon3d/da3` as a branch of its own; the determinism measurement did not
run it, so its check runs the workbench's script twice first.

Rules for these, besides the ones above:

- **No endoscope border in the depth stage.** On CholecSeg8k, the
  workbench's depth stage looks for the black border around the endoscope's
  view and, when it covers 1 % of the frame or more, fills it in before DA3
  sees it. The clips the paper reads never reach that:
  `crop_cholec_frames.py` has already cut each video to a rectangle inside
  the view, one per video, recorded with each clip in `crop_info.json`. On
  the nine clips of the development machine the border found is 0.46 % at
  most, and every manifest says `border_inpaint: false`. The port carries
  neither the detection nor the fill. It writes `border_inpaint: false`, as
  the workbench does, and refuses a CholecSeg8k clip without
  `crop_info.json` rather than give the model a border. On G, the manifests
  of the scored clips are read first: one that says `true` stops this rule.
- **Anyone can install it.** On a machine with a CUDA GPU and Python 3.12,
  the pipeline installs with one command, `pip install -e ".[recon3d]"`, and
  the weights download from Hugging Face on first use. `recon3d_wrapper` and
  `pipeline` are packages the install provides, beside `surgical_core` and
  `evalkit`, and a stage runs as `python -m pipeline.<stage>`, with no
  `sys.path` edits. The README says so, says what the machine needs, and
  names the weights' licence: DA3-LARGE and Pi3X are CC BY-NC 4.0, unlike
  this repository's code. A CI job runs the README's steps on a clean runner
  and imports both backends, with no GPU and no weights. An install that
  works only on G is not done.
- **The extra says what any machine needs; a constraints file says what G
  ran.** `recon3d` names the two model repositories at a commit, everything
  the stages import (`matplotlib` for the colormap, `trimesh` for the GLB
  writer), and what the models import without declaring it: Pi3's package
  declares only `safetensors`. It keeps the models' bounds: DA3 needs `numpy`
  below 2, which is why the extra needs Python 3.12. G's environment, every
  package at its version, Pillow and matplotlib included, is a constraints
  file beside it, read from G. With it, `pip install -e ".[recon3d]" -c
  <that file>` gives G's packages to a Linux machine on x86-64 whose CUDA
  driver is no older than G's, with torch from the index the README names.
  The base install does not change.
- **Depend on the models, never copy them.** The commits are the ones G ran,
  read from pip's record of each install (`direct_url.json`), not from the
  version strings, which say `0.0.0` and `0.1`. Until they are read, the
  extra names the repositories, and the pull request says so.
- **The output stays the workbench's, file for file**: names, formats, the
  manifest's keys and their order. A change to what a stage writes is a
  change to the data contract, made after the stage's check has passed,
  never with the port. Where the workbench passes over an input that breaks
  the stage's contract (a clip without a manifest, a CholecSeg8k clip that
  was not cropped), the port raises. Where it skips a frame for what the
  data is (a frame with no valid depth gets no point cloud), the port skips
  it too: raising there would stop a clip the workbench completes.
- **The tests need no torch and no weights.** The model sits behind the
  interface, and the tests drive the stage through the stand-in. A test that
  needs torch skips without the extra, as the render tests do, and the
  install job, which has torch, runs it, so that the wrappers' torch code is
  tested before G. On G nothing is decided by a test that can skip: the byte
  check decides, and it raises.
- **This repository's helpers replace the workbench's copies**, and the byte
  check says whether that changed anything. One difference is known:
  `surgical_core.geometry.valid_depth_mask` keeps depth above `DEPTH_MIN`
  (1e-6), the workbench's `pointcloud.valid_depth_mask` above 0. Both stages
  and the colormap call it.

### `pipeline/byte-check`

**Source.** W `scripts/measure_determinism.py` (606 lines) and
`tests/test_measure_determinism.py` (366 lines). Standard library and
`numpy`.

**What it is.** It gives each run a fresh working copy of a clip's inputs
(images and masks linked in, a manifest stripped of what the stages write),
collects everything else the run writes under that copy, and compares two
runs byte for byte, reporting how large a difference is. In the workbench
both runs are one stage. Here they are two commands passed in: the
workbench's stage and this repository's, or one stage twice, which measures
determinism.

**Work.** Port the working copy, the collection and the comparison with
their tests, in English. What changes:

- The workbench's table of stages becomes arguments: the two commands, the
  repository each belongs to, the earlier stage's outputs a stage reads, the
  clips' root and the weights. The working copy can keep a clip's first
  frames only, images and manifest alike, for the check on CPU.
- Each run records where it imported `surgical_core` and the stage's modules
  from, and its repository's commit. A machine can hold three packages named
  `surgical_core`: this repository's, the workbench's, and, on the
  workbench's machines, a sibling clone's, which an editable install puts on
  every process's path (the workbench's
  `tests/test_scripts_import_own_surgical_core.py` records it). A run that
  picks up the wrong one compares something else, and can pass. The tool
  refuses a run that imported one of those modules from outside its own
  repository, or whose imports it could not record, and a repository with
  uncommitted changes.
- Two runs whose environments differ (packages, GPU model, driver, weights)
  are not compared; two GPUs of one model in one machine are one
  environment, as in the determinism measurement. The packages recorded add
  Pillow and matplotlib, which write the depth images, and each model's
  commit.
- Only `runtime_sec` is exempt, and a file that differs only there is
  counted apart, never as equal. The workbench's tool also exempted
  `elapsed_sec`, `timestamp` and `date`, which none of the stages writes.
- The linked inputs are hashed before and after each run, and a run that
  changed one is refused: on G they are the files of the paper's data.

**Done when.** The ported tests pass, and each of these is reported or
refused, with a test that plants it: a one-ulp difference, a missing file
and an extra one, a changed manifest field, a changed `timestamp`, a run
that imported `surgical_core` from outside its repository, a repository with
uncommitted changes, two runs in different environments, and a run that
wrote into its linked inputs. Nothing in the tool names a workbench path or
module.

### `surgical-core/camera`

**Source.** What the two stages import from W's `surgical_core`:
`geometry/__init__.py` (89 lines: `backproject_depth`, `cam_to_world`,
`world_to_gltf`) with `tests/test_cholec_geometry.py` (257 lines);
`viewer/depth_vis.py`, `glb.py` and `camera_axes.py`.

**What it is.** The transforms from pixels to the viewer's glTF space, the
depth colormap, the point-cloud writer, and the camera's axes in glTF space,
which the Pi3X stage and `regen_cholec_glb.py` record.

**Work.** W's `surgical_core.geometry` is not the package of that name here,
which came from the tracking experiment; its transforms join that package as
one module. `surgical_core.geometry.project.backproject` already
back-projects, so one of the two definitions goes. They compute the same
thing in a different order (`(p - t) @ R` against `(R.T @ (p.T - t)).T`),
which can move the last bit, so the one that keeps the stages' bytes stays,
and the other is rebuilt on it. `project_world_to_pixel` is not carried: no
stage calls it, and `project_world_to_frame` projects already. The workbench's
`pointcloud.valid_depth_mask` is not carried either; this repository's
replaces it. The viewer modules join `surgical_core.viewer`. The colormap
needs `matplotlib`, which the `render` extra has; the GLB writer imports
`trimesh`, which no extra has yet, so this branch adds it to one, and
`recon3d` includes both.

**Done when.** The ported tests pass from the package, and each function has
one definition.

### `recon3d/da3`

**Source.** W `da3_surgery_wrapper/da3_surgery_wrapper/inference.py` (83
lines) and `scripts/run_cholec_depth.py` (340 lines). The wrapper's other
scripts and its documents, in Japanese, stay behind; what a reader needs from
them goes into `docs/pipeline.md`.

**What it is.** The first stage: depth for every frame of a clip from DA3,
an image of it, and a point cloud per frame. It writes `depth_raw/*.npy`,
`depth_vis/*.jpg`, `exports/mini_npz/results.npz`, `pc_vis/frame_*.glb` and
the manifest's `depth_info`. The workbench's stage also writes
`prep_images/*.png` and `endoscope_content_mask.npy` where it fills a border
in; on a cropped clip it never does, and the port does not ("No endoscope
border in the depth stage", above).

**Work.** `recon3d_wrapper/` holds the interface a reconstruction model
answers to, with DA3 as its first backend; the stage that drives it goes
under `pipeline/`, and both join the packages `pyproject.toml` installs.
This branch also brings what makes it installable by anyone: the `recon3d`
extra and its constraints file, the README's install steps, the first page
of `docs/pipeline.md` (how to run the stage on a clip), and the CI job that
installs the extra.

**Done when.** Anywhere: the CPU tests pass, through the stand-in, on the
files the stage writes and on the manifest's keys, and the install job is
green. On the development machine and on G: every file of the clips each
checks equals the workbench's.

### `recon3d/pi3x`

**Source.** W `pi3_wrapper/pi3_wrapper/inference.py` (381 lines),
`pi3_wrapper/scripts/run_pi3_depth.py` (468 lines), and the tests in
`pi3_wrapper/tests/`: `test_manifest_merge.py` and `test_roundtrip.py` need
no torch, part of `test_inference.py` needs it, and none needs the weights.

**What it is.** Pi3X reconstructs depth and the camera's poses together. The
stage writes `exports/mini_npz/results__pi3x.npz`, `pc_vis/frame_*__pi3x.glb`
and `depth_vis/*__pi3x.jpg`, and adds `geometry_sources.pi3x` to the
manifest: at the top, with its `runtime_sec`, and on every frame, with the
point cloud's centroid, its size and the camera's axes.

**Work.** The second backend behind the same interface. If the poses do not
fit the interface DA3 gave, the interface changes in this branch, and DA3's
tests show that the change costs DA3 nothing. The README's install steps
cover Pi3X too. The `--vo` path, which chains the chunks of a long clip and
replaces a method of upstream's `Pi3XVO` in place, is not ported: the
determinism measurement ran without it, and no check clip runs it.

**Done when.** Anywhere: the ported tests pass, and a CPU test drives the
stage through the stand-in, as for DA3. On G: every file of the three clips
equals the workbench's, `runtime_sec` aside.
