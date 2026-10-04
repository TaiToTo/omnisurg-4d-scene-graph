# Review — what a change is checked against

This repository is written so that anyone who opens one file cold, the
authors a month later included, sees at once what it does. A change is
reviewed against this page. `AGENTS.md` lists the rules in one line each and
holds the two that never change, the freeze rule and the verdict rule; this
page says what each rule asks for, where what it cuts goes instead, and
which test refuses a breach.

## What a test refuses

| rule | test |
|---|---|
| no `TODO`, `FIXME`, `XXX`, `HACK` in a tracked file | `tests/test_no_todo_in_code.py` |
| a module's header is at most `HEADER_LINES` lines before its Usage | `tests/test_module_headers_are_short.py` |
| no mark (`★`, `✗`, `○`, `×`) printed without borrowing `paired_stats.verdict`, none decided by a p-value | `tests/test_paired_stats.py` |
| no personal address in the history | `tests/test_no_personal_email.py` |

A rule without a test in this table is checked by the reviewer.

## A module's header

The module docstring says what the module does, in one paragraph of at most
twelve lines, and then, for a command, how to run it under `Usage:`. A reader
decides from it whether this is the file they want, and nothing else is
asked of it.

What it does not hold, and where that goes instead:

- **History.** What was once measured, what the workbench's version did,
  what a review changed. The commit message and the pull request keep it,
  and git keeps those. An incident that a check now prevents is recorded by
  the test that plants the fault: its name says what happened.
- **Definitions.** A term is defined once, in `docs/evaluation.md`, and used
  everywhere else in that sense. A term that only this module needs is
  defined in one line, in the docstring that first relies on it, before the
  sentence that does; a second definition of a term the specification
  already has drifts from the first.
- **Reasons.** The reason for a choice is a comment of one or two lines at
  the line that makes the choice, and only where a reader would otherwise
  "fix" it. A reason that needs more than two lines is a decision, and
  belongs in `docs/porting.md` or `docs/evaluation.md`.

A header that still does not fit is a module that does more than one thing:
if saying what it does needs "and", split it, and give each part its own
header. Split for that reason only; every file is one more to open.

The modules longer than the cap when the rule was made are listed in the
test as `STILL_LONG`, and the workstream `docs/short-headers` shortens them.
A module leaves the list when it is shortened, and cannot rejoin it.

## Docstrings and comments

- Google-style docstrings; imports at the module top.
- **A function of several steps names each step.** A function that composes
  other modules is read to find where one result comes from, not top to
  bottom; one line per block, saying what it produces and from what. The
  reason for a choice inside a block is a comment of its own. A function
  short enough to read whole needs none.
- **A term is called by what it says.** A question or a rule is named by
  what it asks or states, never by its place in a list ("the second
  question"): a place means something only to whoever has the list open.
- **A comment gives the reason, not a reference.** Never a section number, a
  ticket, an audit letter or a task id (`see §2.3`, `audit B7`, `task22`):
  they point at documents this repository does not have, and the reader is
  left holding a dead pointer. The only citable things are the ones that
  outlive the work: the frozen sha, the module that defines a rule
  (`paired_stats.VERDICT_RULE`), a published paper.
- Code, comments and docs in English.

## Code

- **Fail closed.** If an invariant cannot be checked, raise. Never skip
  quietly.
- **A check earns its place by failing when it should**, not by passing.
  Every check comes with a test that plants the fault and shows the check
  refusing it.
- **No TODO in code.** A path known to give a wrong answer for some input
  raises on that input, with a test that plants it. A decision not yet made
  is written under "Open questions" in `docs/porting.md`, where it is read,
  not in a comment, where it is not. The reason is a clip extractor that
  knew its gap frames carried the wrong frame number, wrote the TODO, and
  kept producing data: two months later the fault was found again
  downstream and explained wrongly, because a comment is read only by
  whoever opens that file, and a raise is read by whoever runs it.
- Names say what a thing is, not how much of it one experiment used.
- Never vendor upstream model code; depend on it.

## A pull request

- One concern per branch, from `main`, about five changed files; the branch
  is named `<area>/<thing>`.
- The body says, in this order: what it brings; where it departs from its
  source and why; what is not there. It does not narrate the process: not
  what it is stacked on, when `main` was merged in, or how many review rounds
  it took. Those are in the history.
- A document in the repository names no pull request number and no merge
  status. A figure is drawn, and the text is the intent: a figure that
  disagrees with the text is fixed toward the text.
- Commit messages and pull request text are in English.
