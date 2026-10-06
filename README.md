# omnisurg-4d-scene-graph

Open, training-free **4D** (3D + time) scene graphs for minimally invasive
surgery — built by composing foundation models, with no task-specific training.

**Status: private / under construction.** Nothing here is released yet.

## Install

The evaluation toolkit needs Python 3.12 or newer and nothing else:

```bash
pip install -e .
```

The pipeline needs a Linux machine with an NVIDIA GPU and its driver, and
Python 3.12. Python 3.12 is the newest Python it supports, because Depth
Anything 3 requires `numpy` below 2; on a newer Python the install fails
while resolving packages, without saying so. A machine without a GPU can run
the depth stage slowly; `docs/pipeline.md` gives the steps.

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

## License

Apache-2.0. Copyright 2026 Yasuto Tamura.
