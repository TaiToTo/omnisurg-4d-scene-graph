# omnisurg-4d-scene-graph

This repository builds training-free **4D** (3D + time) scene graphs for
minimally invasive surgery by composing foundation models. The pipeline
takes a clip of surgical video and estimates depth and camera poses for
every frame. It cuts each frame into regions, tracks them through time and
builds scene graphs from them. The evaluator scores the regions against the
hand-labelled masks of ATLAS-120k and CholecSeg8k. Every score records a
hash of the evaluator's code, `eval_code_sha`. Two scores are compared only
when their hashes match. A web page shows a clip's point clouds, regions
and scene graphs.

## Status

The code is being moved here from the research workbench where the
measurements were made. The evaluator is not frozen yet, so its scores are
not final. The export stage, which builds the scene graphs, is not ported
yet. `docs/porting.md` says what is ported, what it was checked against,
and what is left.

## What is where

| path | what it holds | read next |
|---|---|---|
| `pipeline/` | the stages, each run as `python -m pipeline.<stage>` | `docs/pipeline.md` |
| `evalkit/` | the evaluator, hashed into `eval_code_sha` | `evalkit/README.md`, then `docs/evaluation.md` |
| `evalkit/tools/` | what reads scores or predictions, never hashed: comparisons, tables, statistics, camera trajectories | each module's docstring |
| `surgical_core/` | dataset readers, geometry and the viewer's data model, shared by the pipeline and the tools; the evaluator imports none of it | each module's docstring |
| `recon3d_wrapper/` | Depth Anything 3 and Pi3X behind one interface | `docs/pipeline.md` |
| `sam3_wrapper/` | tracking with SAM 3 from mask prompts on one frame | `docs/pipeline.md` |
| `atlas120k_meta/` | the clips measured, their crop rectangles and frame ratios; no video | `atlas120k_meta/README.md` |
| `cholecseg8k_meta/` | the clips measured and their crop rectangles; no video | `cholecseg8k_meta/README.md` |
| `d4d_meta/` | the clips the census measures and the sides the measurements score; no frame or point cloud | `d4d_meta/README.md` |
| `viewer/` | the web page | `viewer/README.md` |
| `docs/` | the specifications, and how a change is reviewed | `docs/review.md` |

## Install

The evaluation toolkit needs Python 3.12 or newer; `pip` brings the rest:

```bash
pip install -e .
```

The pipeline needs a Linux machine with an NVIDIA GPU and its driver, and
Python 3.12. Python 3.12 is the newest Python it supports, because Depth
Anything 3 requires `numpy` below 2. On a newer Python the install fails
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
