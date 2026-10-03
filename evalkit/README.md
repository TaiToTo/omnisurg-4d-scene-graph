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

![The evaluator, box by box: inputs per clip, the per-frame keys in three views, the clip means, one score JSON per condition, and the claims](../docs/figures/evaluator_map.png)

The outlined boxes are the path a claim takes: scored pixels, `F1_50` and
`SQ` per frame, their means per clip, one score JSON per condition, the star.
Everything else on the map explains a score or checks that two scores may be
compared.

| box | module | state |
|---|---|---|
| GT masks, read through a class table | `classes`, `class_tables/` | merged |
| Region map | the pipeline | after the evaluator (`docs/porting.md`, step 5) |
| Depth map, crop rectangle | `surgical_core/atlas120k` readers | under review (#17); the CholecSeg8k readers are next |
| Scored pixels | `scored` | merged |
| `F1_50`, `SQ` | `objects` | merged |
| `inst_BF` | `inst_bf` | merged |
| `mIoU` | `classmap` | merged |
| `boundary_F`, `boundary_R_raw` | `boundary` | merged |
| `VI_split`, `VI_merge` | `vi` | merged; its return type is under review (#40) |
| `unlabelled_share` | `unlabelled` | merged |
| One frame, every view | `frame` | under review (#32) |
| Pilot mode | `pilot` | objects and domains under review (#33); the pilot frame driver is next |
| Each key's mean per clip, with the frames it covers | `clip` | under review (#42) |
| `time_IoU` | `time_iou` | merged |
| Identity over time | — | not decided (`docs/porting.md`, open question 1) |
| One score JSON | the entry point; `code_sha` | `code_sha` under review (#34); the entry point is next |
| Reading a score JSON, refusing every mix | `tools.scores` | under review (#35) |
| Checked against the pilot evaluator | `tools.pilot_check` | under review (#39) |
| A star | `tools.paired_stats` | under review (#36) |
| Two conditions, one ruler | `tools.compare_eval` | under review (#37) |
| What ran, under which condition | `tools.condition_inventory` | under review (#38) |

The state column is updated as pull requests merge, and goes once the port is
done, with `docs/porting.md`.

## How a pull request uses this page

One box per pull request. The description names the box and shows this map
with that box highlighted; what the module returns and the counts behind it
go in the module's docstring. A pull request that adds a box the map does not
have changes the map first.
