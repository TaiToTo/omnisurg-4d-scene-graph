# Workstreams: what can be ported side by side

**Status: in progress.** `docs/porting.md` gives the order of the port. This
document says which pieces of it can move at the same time, in separate
branches and separate sessions, while the evaluator is being built and
reviewed. It is deleted with `docs/porting.md` once the port is done.

The order in `docs/porting.md` binds two things: nothing is re-scored before
the evaluator is checked against the pilot evaluator, and nothing touches a
GPU before that. It also makes the evaluator itself sequential: class tables,
then metrics, then the entry point. Everything else that neither reads a score
nor imports the evaluator can move now. This document lists what that is.

## Where a workstream can run

Each piece has a source, and the source decides where the work can be done:

- **W, the workbench.** `$OMNISURG_SOURCE`, on one machine, never committed.
  Only a session on that machine can read it.
- **E, the earlier repository.** `TaiToTo/omnisurg-4dsg` (private, on
  GitHub). A session anywhere with access to that repository can read it. Its
  branches hold the versions that were already reviewed there; `docs/porting.md`
  names the branch per file.

Where a file exists in both, start from E: that is the reviewed version. Then
diff it against W before opening the pull request, so that a fix made in the
workbench after the review is not lost. A workstream whose source is W only has
to run on the machine that has the workbench.

## Rules every workstream follows

These repeat what `AGENTS.md` and `docs/porting.md` already say, in the form a
session starting cold needs.

- **Branch from `main`**, never from a branch under review. One concern per
  branch, about five changed files, one pull request. Branch names are
  `<area>/<thing>`, as the history shows (`evalkit/class-table-cholecseg8k`,
  `docs/porting-plan`).
- **Do not touch `evalkit/classes.py` or `evalkit/class_tables/`.** They are
  under review on `evalkit/class-table-atlas120k`. A workstream that needs
  them waits for that pull request to merge, then branches from `main` again.
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
  committed; one of the ATLAS-120k manifests carries an absolute path today.
- **Dependencies stay what `pyproject.toml` declares** (`numpy`,
  `opencv-python`, `Pillow`). A module that needs more is either split so that
  the part the toolkit uses does not, or the dependency is added as an
  optional extra in the same pull request, with the reason in the file.
- **Run the tests from a venv**, not the system interpreter:

  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install -e . pytest
  pytest
  ```

## Not now: blocked on the evaluator

These import the evaluator, read its scores or depend on a decision not yet
made. Starting them early means porting them twice.

| piece | waits for |
|---|---|
| metrics, entry point, `eval_code_sha` | the class tables (under review) |
| `paired_stats`, `compare_eval`, `condition_inventory`, `reeval_diff` | the evaluator's comparability check and the fields its scores carry |
| `track_metrics`, `kmerge` | open question 1 in `docs/porting.md` |
| `surgical_core/viewer/labels.py`, `palette.py` | `labels.py` reads the CholecSeg8k colour table, which is now `evalkit.classes`; wait for the class-table review to settle that module's interface |
| the whole pipeline, the wrappers, the viewer | step 5 of the order; needs step 2 done and the GPU |

Open question 1 is not a branch. It is a decision, and `docs/evaluation.md`
("Consistency over time") lists what deciding it means. A session can prepare
it by writing down, for each candidate, the GT-track definition it needs and
how that coexists with the whole-class object; the decision itself is made by
the author, in `docs/evaluation.md`.

## Now: workstreams that can start today

| branch | brings | source | runs where |
|---|---|---|---|
| `surgical-core/clip-time` | `surgical_core/clip_time.py`, its test | E `extract/03-metrics`, W | anywhere |
| `evalkit/test-scenes` | the hand-derived test scenes | E `archive/metric-guide` | anywhere |
| `surgical-core/geometry` | `surgical_core/geometry/`, its test | E `extract/04-kmerge`, W | anywhere; diff against W needs the workbench |
| `atlas120k-meta/data` | the metadata files, with a README | W | workbench machine |
| `atlas120k-meta/readers` | `clip_rects.py`, `frame_ratio.py`, their tests | W | workbench machine |

The first three have a source on GitHub and suit a session without the
workbench. The last two read the workbench and stay on its machine. The two
`atlas120k-meta` branches can be one if the result stays near five files;
`data` first, since `readers` is tested against it.

### `surgical-core/clip-time`

**Source.** E `extract/03-metrics`: `surgical_core/clip_time.py`. W:
`surgical_core/clip_time.py` (111 lines) and `tests/test_clip_time.py` (129
lines). Depends on `numpy` and the standard library, and, when an ATLAS-120k
manifest records no `frame_ratio`, on the table of measured ratios that
`atlas120k-meta/readers` brings; until then such a clip is refused.

**What it is.** One function, `frame_times(root, clip)`, that turns a clip's
`frame_manifest.json` into the real time of every frame in seconds. It exists
in one place because the extracted CholecSeg8k clips are not all in
chronological order (the workbench's extractor leaves the frames it decodes
to fill gaps between annotation chunks at an unconverted frame number, so
they sit 1 to 3 s later than their neighbours), and ATLAS-120k's frame
numbers need the frame-ratio correction; two copies of that rule would drift.

**Work.** English only. The docstring's reason (the two sides that need it,
the clips that break a naive rule) stays; its references to the workbench's
task documents go. The test file's Japanese names become English ones. The
test that loads a module by path with `importlib.util` is checked for what it
actually tests; if it reaches into workbench layout, it is rewritten to the
package import.

**Done when.** `tests/test_clip_time.py` passes from the package; no Japanese
left; `pyproject.toml` unchanged.

### `evalkit/test-scenes`

**Source.** E `archive/metric-guide`: `evalkit/tests/scenes.py` and
`evalkit/tests/cartoon.py`. Both are already in English. `scenes.py` depends
on `numpy`; `cartoon.py` on `numpy` and `cv2`.

**What it is.** The scenes the evaluator's hand-derived tests are computed
on: `scenes.py` holds small GT/prediction pairs with one controlled fault each
(exact, shifted, split, merged, gap, on background, sliver, spill, moved,
swapped); `cartoon.py` draws a laparoscopic frame with a liver, gallbladder,
fat and an instrument, with its own class ids, and a prediction that gets each
one wrong in a typical way.

**Work.** Bring both under `tests/` as helpers (this repository keeps its
tests flat in `tests/`, not under the package). Add one smoke test per module
that pins what the scenes promise: shapes, the ids present, and the fault each
scene is said to contain (a split scene has two predicted regions over one GT
object, and so on). **Do not bring the expected metric values.** The normal
mode's values are derived again under the evaluator's rules when the metrics
are ported; the pilot mode's come from `test_v2_metric_guide.py` and are
ported with pilot mode. Neither belongs to this branch.

**Done when.** Both helpers import from `tests/`, the smoke tests pass, and
nothing in them names a metric.

### `surgical-core/geometry`

**Source.** E `extract/04-kmerge`: `surgical_core/geometry/__init__.py` and
`evalkit/tests/test_geom_edge_ring.py`. W:
`depth_sam_tracking_experiment/geometry.py` (548 lines) and
`depth_sam_tracking_experiment/tests/test_geom_edge_ring.py` (75 lines).

**What it is.** Camera-space normals from depth and intrinsics, geometric
edge maps, the images the segmenter is prompted with, and back-projection and
label warping between frames. Shared by the pipeline and the toolkit.

**Work.** `docs/porting.md` marks it "English only", but the workbench file
imports three things the package does not declare: `scipy.spatial.cKDTree`
(back-projection and label transfer), `matplotlib` (one colormap call) and
`sam3d_core.num_to_natural` (an eight-line helper that renumbers group ids to
`0..K-1`; `sam3d_core` is pipeline code and is not ported yet). The branch
first reads how E's reviewed version resolved these, then follows it or
improves on it. The recommended shape, if E did not already settle it:

- `surgical_core/geometry/normals.py`: `camera_normals`, `normal_map`,
  `edge_reliable_mask`, `geom_edge_map`, `normal_edge_map`. `numpy` and `cv2`
  only. This is what the test covers and what the toolkit needs.
- `surgical_core/geometry/render.py`: the colormap, relighting, retinex and
  `sam_input_image`. Needs `matplotlib`.
- `surgical_core/geometry/project.py`: `backproject`, `project_*`,
  `warp_labels`. Needs `scipy`; `num_to_natural` is copied in with a docstring
  saying it is the pipeline's helper, so the two are merged when `sam3d_core`
  moves.
- `scipy` and `matplotlib` go in `pyproject.toml` as an optional extra, with
  a comment saying the evaluator does not need them and the pipeline does.

Whether to split, and what to call the extra, is flagged in the pull request
for the reviewer.

**Done when.** `test_geom_edge_ring.py` passes against the package; a fresh
`pip install -e .` without the extra still imports `evalkit` and
`surgical_core.geometry.normals`; the diff against W shows nothing E missed.

### `atlas120k-meta/data`

**Source.** W only. From `docs/porting.md`, "ATLAS-120k metadata":

| workbench file | what it is | size |
|---|---|---|
| `experiment/crop_necessity/verdicts/verdicts_latest.json` | the crop rectangle per clip, as the judging tool wrote it (`rect`, `verdict`, `src_size`) | 216 kB |
| `experiment/crop_necessity/marks/*.jsonl` | the cut marks, two files | small |
| `ipcai2027_experiment/frozen/atlas97_clips.txt` | the clip population, 315 lines | 11 kB |
| `ipcai2027_experiment/frozen/atlas97_depth_manifest.json` | the depth manifest for that population | 91 kB |
| `ipcai2027_experiment/task22_atlas100/out/manifest/` | the 100-video manifest (`videos.json`, `clips_v100.*`, `population.txt`, `production_frozen.json`, `excluded_short.json`, ...) | 230 kB |
| `ipcai2027_experiment/task22_atlas100/out/audit/` | the audit of it (`crop_scope_table`, `gt_coverage`, `ui_residue`, each `.json` and `.md`) | 230 kB |

No video, no frame, no mask.

**Work.** Copy them under `atlas120k_meta/` with names that say what each file
is, not how many of something one experiment used (`AGENTS.md`): the
population is `clips.txt`, not `atlas97_clips.txt`; the manifest directory is
not `task22_atlas100`. Propose the names in the pull request, with the
workbench path each came from recorded in `atlas120k_meta/README.md` so the
provenance is kept once the names change. The README is the one the release
step needs; a first draft here is enough.

Before committing, grep every file for machine paths and addresses.
`videos.json` holds an absolute `atlas_root` path today; it is removed or
replaced by a relative one, and the README says so. The audit `.md` files are
in Japanese; either translate them or leave them out and say in the README
where they are. Leaving them out is the smaller branch.

**Done when.** Every file in `atlas120k_meta/` is named in the README with its
source; no path, no address; `tests/test_no_personal_email.py` passes.

### `atlas120k-meta/readers`

**Source.** W only: `surgical_core/atlas/clip_rects.py` (89 lines),
`surgical_core/atlas/frame_ratio.py` (239 lines), `tests/test_atlas_clip_rects.py`
(85 lines), `tests/test_atlas_frame_ratio.py` (51 lines). Depend on `numpy`,
`cv2` and the standard library.

**What they are.** `clip_rects` reads the per-clip crop rectangles: a video
does not have one rectangle, since the recording changes mid-video in some,
and the recipe's guess was wrong for most clips, so a person judged each clip
and the file is that judgement. `frame_ratio` holds the measured ratio between
a clip index's frame numbers and the mp4's: for some videos the annotation
numbers frames at a lower rate, the ratio cannot be derived from the fps, so
it is a measurement, and an unmeasured video is refused rather than guessed.

**Work.** Port as `surgical_core/atlas120k/clip_rects.py` and `frame_ratio.py`
(the module is `atlas120k`, decision 8 in `docs/porting.md`). English only;
the docstrings' measurements and reasons stay, their pointers into the
workbench's experiment directories become a sentence saying where the data
came from. `clip_rects` takes the rectangles file as an argument and keeps
doing so. `frame_ratio` carries its measured table in code; that is data
about which videos enter a measurement, which `AGENTS.md` keeps under
`atlas120k_meta/`. Recommended: move the table to
`atlas120k_meta/frame_ratio.json`, read it fail-closed, and have the test
plant an unmeasured video and watch it refused. Flag this in the pull request
as the one decision.

One thing to settle with it. `surgical_core.clip_time.frame_times` asks the
table when an ATLAS-120k manifest records no `frame_ratio`, and it calls the
workbench's `frame_ratio()`, which answers 1 both for a video measured at 1
and for a key it does not hold. The two are not the same, and the second
happens: in one workbench tree, manifests carry a display string such as
`"pi3x (ATLAS-120k)"` as `procedure`, so the lookup misses and 1 comes back.
Those clips carry `timestamp_sec` and never reach the table today, but
nothing guarantees the next tree will. Once the table is read fail-closed,
`frame_times` should call the form that refuses an unmeasured key, and its
test for that case (which plants a stand-in table) should be pointed at the
real one.

**Done when.** Both tests pass against the new modules; `clip_rects` is
tested against the file committed by `atlas120k-meta/data`; no Japanese left.

## After the class-table review merges

Once `evalkit/class-table-atlas120k` is in `main`, two more open up, both
branched from `main` at that point:

- `evalkit/metrics`: the next step of the evaluator, from `docs/evaluation.md`,
  on the scenes `evalkit/test-scenes` brought in.
- `surgical-core/viewer-labels`: `surgical_core/viewer/labels.py` and
  `palette.py` (E `extract/03-metrics`), with `labels.py` reading the colour
  table through `evalkit.classes` instead of `surgical_core.cholec`.
