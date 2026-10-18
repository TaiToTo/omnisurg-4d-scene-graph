# omnisurg-4d-scene-graph

This repository builds training-free **4D** (3D + time) scene graphs for
minimally invasive surgery by composing foundation models. The pipeline
takes a clip of surgical video and estimates depth and camera poses for
every frame. It cuts each frame into regions, tracks them through time and
builds scene graphs from them. A frame's scene graph has one node per
region, placed in 3D, and edges that name a spatial relation between two
regions. The evaluator scores the regions, not the scene graphs, against
the hand-labelled masks of ATLAS-120k and CholecSeg8k. Every score records
a hash of the evaluator's code, `eval_code_sha`. Two scores are compared
only when their hashes match. A second scorer scores the camera poses
against StereoMIS's ground truth. A web page shows a clip's point clouds,
regions and scene graphs.

## Status

The code is being moved here from the research workbench where the
measurements were made. The evaluator is not frozen yet, so its scores are
not final.

- Ported: the extraction stages and the D4D census, depth and camera
  poses, segmentation and tracking, the evaluator and its tools, the camera
  trajectory scorer, and the viewer.
- Not ported yet: the export stage, which builds the scene graphs; the
  paper's tables; and the D4D measurements after the census.

This repository alone takes a clip as far as tracked regions and their
scores. The viewer bundle stage reads scene graphs that the workbench's
export stage wrote. [`docs/porting.md`](docs/porting.md) says what each
part was checked against, what is left, and the decisions still to carry
out, such as splitting the ATLAS-120k clips at the cuts in their
recordings.

## Where to start

- To see what is measured: [`evalkit/README.md`](evalkit/README.md) puts
  the three questions on one drawn scene, and
  [`docs/evaluation.md`](docs/evaluation.md) gives the full rules.
- To run the pipeline: [Install](#install) below, then
  [`docs/pipeline.md`](docs/pipeline.md), which lists the stages in the
  order they run.
- To change the code: [`docs/review.md`](docs/review.md), the rules a
  change is reviewed against.

## What is where

| path | what it holds | read next |
|---|---|---|
| [`pipeline/`](pipeline/) | the stages, each run as `python -m pipeline.<stage>` | [`docs/pipeline.md`](docs/pipeline.md) |
| [`evalkit/`](evalkit/) | the evaluator, hashed into `eval_code_sha` | [`evalkit/README.md`](evalkit/README.md), then [`docs/evaluation.md`](docs/evaluation.md) |
| [`evalkit/tools/`](evalkit/tools/) | what reads scores or predictions, never hashed: comparisons, tables, statistics | each module's docstring |
| [`trajectory_eval/`](trajectory_eval/) | the camera trajectory scorer, hashed into `trajectory_code_sha`; its `tools/` score StereoMIS and read the scores, never hashed | [`docs/pipeline.md`](docs/pipeline.md#the-stereomis-camera-trajectory-scores) |
| [`surgical_core/`](surgical_core/) | dataset readers, geometry and the viewer's data model, shared by the pipeline and the tools; the evaluator imports none of it | each module's docstring |
| [`recon3d_wrapper/`](recon3d_wrapper/) | Depth Anything 3 and Pi3X behind one interface | [`docs/pipeline.md`](docs/pipeline.md) |
| [`sam3_wrapper/`](sam3_wrapper/) | tracking with SAM 3 from mask prompts on one frame | [`docs/pipeline.md`](docs/pipeline.md) |
| [`atlas120k_meta/`](atlas120k_meta/) | the clips measured, their crop rectangles and frame ratios; no video | [`atlas120k_meta/README.md`](atlas120k_meta/README.md) |
| [`cholecseg8k_meta/`](cholecseg8k_meta/) | the clips measured and their crop rectangles; no video | [`cholecseg8k_meta/README.md`](cholecseg8k_meta/README.md) |
| [`d4d_meta/`](d4d_meta/) | the clips the census measures and the sides the measurements score; no frame or point cloud | [`d4d_meta/README.md`](d4d_meta/README.md) |
| [`viewer/`](viewer/) | the web page | [`viewer/README.md`](viewer/README.md) |
| [`docs/`](docs/) | the specifications, and how a change is reviewed | [`docs/review.md`](docs/review.md) |

## Install

The evaluation toolkit needs Python 3.12 or newer; `pip` brings the rest:

```bash
pip install -e .
```

The pipeline needs a Linux machine with an NVIDIA GPU and its driver, and
Python 3.12. Python 3.12 is the newest Python it supports, because Depth
Anything 3 (DA3) requires `numpy` below 2. On a newer Python the install fails
while resolving packages, and the error does not name the Python version as
the reason. A machine without a GPU can run the depth stage slowly;
`docs/pipeline.md` gives the steps.

```bash
git clone https://github.com/TaiToTo/omnisurg-4d-scene-graph.git
cd omnisurg-4d-scene-graph
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[recon3d]"
python -m pipeline.depth --input-dir /path/to/clips --clips <clip>
python -m pipeline.pi3x --input-dir /path/to/clips --clips <clip>
```

The model weights download from Hugging Face on first use: about 1.4 GB for
DA3-LARGE and 5.1 GB for Pi3X. Both are licensed CC BY-NC 4.0, for
non-commercial use only, unlike this repository's code. `docs/pipeline.md`
describes a clip and the files each stage writes.

The D4D census reads the dataset through D4D's own loader, which pip cannot
install. Clone the loader at the commit that `surgical_core.d4d` checks, and
put the clone on `PYTHONPATH`. The census refuses to run without the loader,
or with files other than that commit's.

```bash
git clone https://github.com/reubendocea/d4d.git /path/to/d4d
git -C /path/to/d4d checkout efe12d26
pip install -e ".[d4d]"
export PYTHONPATH=/path/to/d4d
```

## Tests

Run the tests with the three extras, as CI does. Without them the tests
that need matplotlib, trimesh or scipy are skipped, and the run still looks
green.

```bash
pip install -e ".[render,tools,stereomis]" pytest
pytest
```

## License

Apache-2.0. Copyright 2026 Yasuto Tamura.
