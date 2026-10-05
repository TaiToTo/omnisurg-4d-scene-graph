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

## Not now: blocked on a decision or on the GPU

`track_metrics` and `kmerge` wait on a decision not yet made; started before
it, they would be ported twice. The pipeline, the wrappers and the viewer wait
on the order of `docs/porting.md`: nothing touches a GPU before the evaluator
is checked against the pilot evaluator.

| piece | waits for |
|---|---|
| `track_metrics`, `kmerge` | open question 1 in `docs/porting.md` |
| the whole pipeline, the wrappers, the viewer | step 5 of the order; needs step 2 done and the GPU |

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
