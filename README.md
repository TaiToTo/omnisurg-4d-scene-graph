# omnisurg-4d-scene-graph

Training-free **4D** (3D + time) scene graphs for minimally invasive
surgery, built by composing foundation models. The pipeline takes a clip of
surgical video, estimates depth and camera poses for every frame, cuts each
frame into regions and tracks them through time, with no task-specific
training. The evaluator scores those regions against the hand-labelled masks
of ATLAS-120k and CholecSeg8k. Every score records a hash of the
evaluator's code, and two scores are compared only when the hashes match.
A web page shows a clip's point clouds, regions and scene graphs.

## Status

The code is being moved here from the research workbench where the
measurements were made. "Checked" means the port gave the workbench's
numbers or files on the same input.

| part | state |
|---|---|
| evaluator, `evalkit/` | implemented and checked; not frozen yet |
| score tools, `evalkit/tools/` | implemented, apart from `track_metrics`, which is not ported yet |
| extraction of ATLAS-120k, CholecSeg8k, LapEx, StereoMIS and the D4D population | implemented |
| depth: Depth Anything 3, Pi3X, point clouds | implemented |
| segmentation and tracking with SAM 3 | implemented and checked on three clips |
| seeds made outside the tracking stage (`--seed-labels`) | the tracking stage takes them; the evaluator refuses them until their scoring is decided |
| the export stage, which builds the scene graphs | not ported yet |
| viewer bundle and web page | implemented |

`docs/porting.md` says in detail what is ported, what it was checked
against, and what is left.

## What is where

| path | what it holds | read next |
|---|---|---|
| `pipeline/` | the stages, each run as `python -m pipeline.<stage>` | `docs/pipeline.md` |
| `evalkit/` | the evaluator, hashed into `eval_code_sha` | `evalkit/README.md`, then `docs/evaluation.md` |
| `evalkit/tools/` | what reads scores: comparisons, tables, statistics | each module's docstring |
| `surgical_core/` | what the pipeline and the evaluator share: dataset readers, geometry, the viewer's helpers | each module's docstring |
| `recon3d_wrapper/` | Depth Anything 3 and Pi3X behind one interface | `docs/pipeline.md` |
| `sam3_wrapper/` | tracking with SAM 3 from mask prompts on one frame | `docs/pipeline.md` |
| `atlas120k_meta/`, `cholecseg8k_meta/`, `d4d_meta/` | which clips are measured, and how each is cropped or which sides are scored; no video | the README in each |
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
