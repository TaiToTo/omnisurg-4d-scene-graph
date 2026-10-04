# AGENTS.md

Router for AI coding assistants working in this repository. This file holds what
does not change: the freeze rule, the target layout and the conventions. Task
detail lives in the documents named below — read those before doing real work.

`CLAUDE.md` is a symlink to this file.

---

## What this repo is

`omnisurg-4d-scene-graph` builds **4D (3D + time) scene graphs for minimally
invasive surgery** by composing foundation models — monocular depth,
promptable segmentation, tracking — with **no task-specific training**. It
ships the evaluation toolkit those graphs are measured with, the pipeline that
produces them, and a viewer.

**Status: under construction.** The code is being extracted, file by file, from
a private research workbench where the measurements were made. Until the
extraction lands, most of the layout below does not exist yet.

---

## The one rule that cannot be relaxed

Nothing in this repository is frozen yet. The evaluator specified in
`docs/evaluation.md` is frozen before the paper's numbers are measured, and
from then on the files it hashes into `eval_code_sha` do **not change — not one
byte**, ever, including "while we are tidying up anyway".

`eval_code_sha` is what makes two measurements comparable. Change a frozen file
and every number measured before becomes incomparable with every number
measured after — silently, because nothing crashes. `compare_eval.py` refuses
to mix scores across shas, and that refusal is the only thing standing between
us and a plausible-looking wrong table.

The pilot measurements were scored by the *pilot evaluator* (`eval_code_sha =
1f8a813a…`). It stays frozen in the private workbench and is not part of this
repository. The evaluator is checked against it where it lives, taking the
paths of the pilot evaluator and its scores as arguments.

The verdict rule is already settled and is held to the same standard: **a
claim gets a star only when the video-level bootstrap 95 % CI does not straddle
zero.** p-values are reported alongside, never decisive. One definition, in
`paired_stats`; scripts borrow it rather than reimplementing a star.

---

## Where the source material is

The extraction reads from the private workbench this repo comes from. Its path
is machine-specific, so it is **never hardcoded and never committed**:

```bash
export OMNISURG_SOURCE=/path/to/the/workbench   # see docs/local/ (git-ignored)
```

Documents in this repo refer to it as `$OMNISURG_SOURCE`. If a command needs a
path under it, it takes the path as an argument — this repo does not learn the
shape of another repo.

---

## Target layout

```
omnisurg-4d-scene-graph/
├── LICENSE  README.md  CITATION.cff  pyproject.toml
├── surgical_core/     cholec atlas120k geometry pointcloud preprocess viewer clip_time
├── evalkit/           the evaluator, hashed into eval_code_sha with its class tables
│   └── tools/         what only reads scores, never hashed: paired_stats
│                      compare_eval condition_inventory
│                      (track_metrics, kmerge: which side waits on docs/porting.md)
├── pipeline/          depth → segmentation → tracking → viewer export
├── recon3d_wrapper/   3D reconstruction — DA3 and Pi3 behind one interface
├── sam3_wrapper/      promptable segmentation and tracking (SAM 3)
├── atlas120k_meta/    crop rectangles, cuts, clip population (no video)
├── viewer/
├── docs/              evaluation.md, porting.md, data_contract.md, pipeline.md
├── tests/
└── .github/workflows/ci.yml
```

Names say what a thing is, not how much of it one experiment used. The dataset
is ATLAS-120k, spelled `atlas120k` as upstream spells it; which of its videos
and clips enter a measurement is data, kept in the population file under
`atlas120k_meta/`, never a count in a directory name.

---

## The port

The port follows `docs/porting.md`: what moves, which version of it was
already reviewed and where, the order of the steps, and what must hold before
the next one. Read it before porting anything. The order is load-bearing: the
evaluator is checked against the pilot scores before anything is re-scored,
and nothing touches a GPU before that. Nothing is frozen during the port.

What is permanent is on this page: the freeze rule, the verdict rule, the
layout, and the conventions below.

---

## Conventions

- **Fail closed.** If an invariant cannot be checked, raise. Never skip quietly.
- **A check earns its place by failing when it should**, not by passing. When
  you add one, demonstrate the failure (a self-test that plants the fault).
- **No TODO in code.** A path known to give a wrong answer for some input
  raises on that input, with a test that plants it. A decision not yet made
  is written under "Open questions" in `docs/porting.md`, where it is read,
  not in a comment, where it is not. `tests/test_no_todo_in_code.py` refuses
  the markers (`TODO`, `FIXME`, `XXX`, `HACK`) in every tracked file but the
  planning documents and this one. The reason is a clip extractor that knew
  its gap frames carried the wrong frame number, wrote the TODO, and kept
  producing data: two months later the fault was found again downstream and
  explained wrongly, because a comment is read only by whoever opens that
  file, and a raise is read by whoever runs it.
- Google-style docstrings; imports at module top; comments say *why*.
- **A function of several steps names each step.** A function that
  composes other modules is read to find where one result comes from, not
  top to bottom. Give each block one line saying what it produces and from
  what, so the reader lands on the right lines without reading the rest.
  Say what the block *is*, not why it is so: the reason for a choice inside
  it is a comment of its own, as below. A function short enough to read
  whole needs none.
- **A comment gives the reason, not a reference.** Never cite a section number,
  a ticket, an audit letter or a task id (`see §2.3`, `audit B7`, `task22`):
  they point at documents this repository does not have and will not keep, so
  the reader is left holding a dead pointer instead of a reason. Write the
  reason itself, in a line or two; if it does not fit, it belongs in the
  docstring. The only citable things are the ones that outlive the work — the
  frozen sha, the module that defines a rule (`paired_stats.VERDICT_RULE`), a
  published paper.
- Code and docs in English.
- Never vendor upstream model code — depend on it.
