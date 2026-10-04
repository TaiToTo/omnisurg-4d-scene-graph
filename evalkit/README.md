# evalkit

The evaluator specified in [`docs/evaluation.md`](../docs/evaluation.md). The
rules live there. This page says what the evaluator decides, and maps its
parts: one box per thing it does, with the module that holds it. What each
module returns, and the counts it keeps, is in that module's docstring.

## What it decides

The pipeline splits every frame of a surgical video into regions and tracks
them over time, without naming them and without training on the task. The
paper compares *conditions*, configurations of that pipeline, and claims that
one yields better regions than another. The evaluator turns "better" into a
yes or no.

"Better" is measured against the datasets' ground-truth masks on two numbers,
chosen before any score of this evaluator was seen:

- `F1_50`: are the GT objects found, with every extra region counted against
  the condition;
- `SQ`: how well the found ones fit.

Both in the *geometric* view, only the tissue that depth and shape can
separate, because that is what a pipeline built on depth and shape can be
asked to find. Tools, blood and the abdominal wall are scored too, in the
other views, but no claim rests on them.

A claim gets a star when the video-level bootstrap 95 % CI of the difference
between two conditions does not straddle zero, the one rule in
`paired_stats`. Every other key says *why* a condition scores as it does, and
is reported but never starred.

## The map

![The evaluator at a glance: a clip's inputs; one frame, scored in three views; one clip, each key's mean over its frames; one condition, one score file with eval_code_sha; two conditions, a star](../docs/figures/evalkit_overview.png)

Six steps, from a clip's inputs to a star. Everything in `evalkit/` decides a
score and is hashed into `eval_code_sha`; the tools in `evalkit/tools/` only
read score files and are not hashed, so fixing one leaves every score
comparable.

The same steps, box by box: each box with the module that holds it and, until
it merges, its pull request. The only arrows are what is computed from what.

```mermaid
flowchart TB
    classDef merged fill:#ffffff,stroke:#8a8a8a,color:#222
    classDef review fill:#fff3d1,stroke:#c4902a,color:#222
    classDef todo fill:#f3f3f3,stroke:#a0a0a0,stroke-dasharray:5 4,color:#666
    classDef ext fill:#e8e8e8,stroke:#a0a0a0,color:#444
    classDef here fill:#d9f0ef,stroke:#0f7b7b,stroke-width:3px,color:#111

    subgraph inputs["1 · a clip's inputs"]
        direction LR
        gt["GT masks, class tables<br/>classes.py"]:::merged
        regions["region maps<br/>the pipeline, after the evaluator"]:::todo
        readers["depth maps, crop rectangles<br/>surgical_core · #17, CholecSeg8k next"]:::review
    end

    subgraph view["2 · one frame, one view"]
        scored["scored pixels<br/>scored.py"]:::merged
        subgraph keys[" "]
            direction LR
            objects["F1_50, SQ ★<br/>objects.py"]:::merged
            classmap["mIoU<br/>classmap.py"]:::merged
            vi["VI_split, VI_merge<br/>vi.py · #40"]:::review
            unlabelled["unlabelled_share<br/>unlabelled.py"]:::merged
            boundary["boundary_F, boundary_R_raw<br/>boundary.py"]:::merged
            inst_bf["inst_BF<br/>inst_bf.py"]:::merged
        end
        scored --> keys
        classmap -- class map --> boundary
        objects -- hits --> inst_bf
        boundary -- boundary rule --> inst_bf
    end

    frame["3 · one frame, all three views<br/>frame.py · #32"]:::review
    pilot["3′ · the same, by the pilot evaluator's rules<br/>pilot.py · #33, its frame driver next"]:::review
    time_iou["time_IoU, reference only<br/>time_iou.py"]:::merged
    identity["identity over time<br/>not decided"]:::todo
    clip["4 · one clip: each key's mean, with the frames behind it<br/>clip.py · #42"]:::review
    code_sha["eval_code_sha<br/>code_sha.py · #34"]:::review
    entry["5 · one condition: every clip, one score file<br/>the entry point · next"]:::todo

    subgraph tools["6 · across conditions, reading score files · tools/"]
        scores["read a score file, refuse every mix<br/>scores.py · #35"]:::review
        paired_stats["★ does A beat B<br/>paired_stats.py · #36"]:::review
        compare_eval["two conditions side by side<br/>compare_eval.py · #37"]:::review
        condition_inventory["what ran under which condition<br/>condition_inventory.py · #38"]:::review
        pilot_check["does pilot mode give the pilot's numbers<br/>pilot_check.py · #39"]:::review
    end
    pilot_scores["the pilot evaluator's scores<br/>the workbench, sha 1f8a813a…"]:::ext

    inputs --> view
    view --> frame
    view -. pilot rules .-> pilot
    regions -- every frame --> time_iou
    frame --> clip
    pilot --> clip
    time_iou --> clip
    identity -.-> clip
    clip --> entry
    code_sha -- recorded in --> entry
    entry --> scores
    scores --> paired_stats
    scores --> compare_eval
    scores --> condition_inventory
    scores --> pilot_check
    pilot_scores --> pilot_check

    style inputs fill:#fafafa,stroke:#c8c8c8
    style view fill:#fafafa,stroke:#c8c8c8
    style keys fill:#ffffff,stroke:#e0e0e0
    style tools fill:#fafafa,stroke:#c8c8c8
```

White: merged. Amber: under review. Dashed: not written yet, or not decided.
★: the two keys a claim is judged on, and the rule that gives the star. The
colours and numbers are updated as pull requests merge, and go once the port
is done, with `docs/porting.md`.

## How a pull request uses this page

One box per pull request. The description names the box and shows the map
with that box highlighted: a copy of the block above with one line added at
its end, `class clip here`, the box's id in place of `clip`. What the module
returns and the counts behind it go in the module's docstring. A pull request
that adds a box the map does not have changes the map first.
