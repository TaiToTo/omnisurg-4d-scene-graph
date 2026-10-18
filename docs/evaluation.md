# Evaluation — specification

**Status: implemented in `evalkit/`, not frozen yet.**

## Summary

The pipeline splits each frame into regions and gives them no class. The
evaluator compares these regions with the ground-truth (GT) masks of
ATLAS-120k and CholecSeg8k. A GT object is all pixels of one class in one
frame, and a predicted object is one region. Every metric but `time_IoU` is
computed in three views: `all` (every class), `tissue` (without tools) and
`geometric` (only the tissue that depth and shape can separate). The paper
asks three questions of the regions and judges two of them each on a
primary metric of its own. The question about following a region over time,
like every measure over time, gets reference values only, which decide no
claim. Every score records `eval_code_sha` and the hashes of its inputs, and
two scores are compared only when they are comparable ("Terms").

| question | primary metric | view |
|---|---|---|
| Given one frame as an example, how far can the regions be followed? | none: reference values only ("Consistency over time: reference values only") | — |
| Given no example, what input puts the regions' boundaries where the GT's class boundaries are? | `boundary_R_raw` | each of the three |
| Given no GT, how much of the labelled structure is already in the regions? | `F1_50` and `SQ` | geometric |

## Contents

- [Terms](#terms): the words this document and the tools use.
- [The evaluator at a glance](#the-evaluator-at-a-glance): the six steps of
  scoring, in two figures.
- [How a frame is scored](#how-a-frame-is-scored): the inputs, which frames
  count, objects, the class map, class types and views.
- [Metrics](#metrics): each key, how frames become a clip's value, the
  primary metrics, and the measures over time.
- [Thresholds, refusals and provenance](#thresholds-refusals-and-provenance):
  the thresholds, what is refused, what every score records, and the check
  against the pilot evaluator.
- [Class tables](#class-tables): every class's type, per dataset.

## Terms

- The *pilot evaluator* scored the pilot measurements (`eval_code_sha =
  1f8a813a…`), and stays frozen in the private research workbench.
  `docs/pilot_evaluator.md` says why this evaluator replaced it. *Pilot
  mode* runs this evaluator with the pilot evaluator's rules, to check that
  it writes the same numbers ("Checked against the pilot evaluator"). Every
  other run is in *normal mode*.
- A *condition* is one configuration of the pipeline that the paper
  compares. It gives one prediction per clip.
- A *video* is one recording in a dataset. A *clip* is a stretch of one
  video that the pipeline processes as a unit; a video can supply several.
- A *class set* is the list of classes a score reads the GT in. `original`
  is the dataset's own labels, and claims are judged on it. ATLAS-120k has
  a second class set, `benchmark`: its benchmark's 30 classes, scored as
  reference values only ("ATLAS-120k: the benchmark's 30 classes").
- A *score JSON* is one condition's scores: a summary and one *row* per
  clip, under `per_clip`. A *key* is one column of the rows: the evaluator
  writes `metric/view` (`F1_50/geometric`) and `time_IoU` once per clip; a
  *pilot JSON*, one the pilot evaluator wrote, is told apart by holding none
  of the class set, views, mode, input hashes, versions and propagation rule
  (`scores.EVALUATOR_FIELDS`) and keeps the pilot evaluator's spellings
  (`inst_F1_50`, `inst_F1_50_tissue`, with the domain after an underscore).
  A JSON this evaluator writes in pilot mode is not a pilot JSON.
- A *view* is the set of classes a key is computed over: `all`, `tissue` or
  `geometric` ("Views").
- A *ruler* is what a score was measured with, as `scores.Ruler` holds it:
  `eval_code_sha`, dataset, mode, class set, views, and for a pilot JSON its
  domain. Two scores are *comparable* when every requirement listed under
  "Recorded with every score" holds.
- A *tag* is a condition's name on disk: the directory under each clip that
  holds its labels, and separately the name of its score JSON; the score
  names the label directory it read in `track_dir_name`.
- In a comparison, *base* is the condition compared against and *cond* the
  one compared; a difference is `cond − base`. A key's *direction* is the
  way it is better: higher, lower, or neither for a reference value, as the
  `better` column of the metrics table gives it. The
  *population* is the clips the two are compared on, `identical` when both
  hold the same clips and `intersection` when compared on the common ones;
  per key it shrinks to the clips on which both define it. The *wins* on a
  key are the clips on which it moved the better way, among those.
- A *claim* says that one condition is better or worse than another on a
  key. "From frames to clips" says how the clip values decide it.
- A condition's *regions per frame* is the number of region ids a frame's
  label map holds, every pixel read and every region counted whatever its
  size, averaged over the frames with a prediction and then over the
  clips. It reads no GT, view or depth. `evalkit/tools/region_counts`
  counts it, and `evalkit/tools/region_curves` lays each input's scores
  along it.

## The evaluator at a glance

Scoring runs in six steps. The evaluator reads a clip's inputs, scores a
frame in one view and then in all three, averages each key over the clip, and
writes one score file per condition. The tools then compare two conditions.

![Scoring in six steps on a drawn scene: a clip's inputs; one frame, scored in each of three views (steps 2 and 3); one clip, each key's mean over its frames; one condition, one score file; two conditions, the keys that decide the questions about boundaries and structure.](figures/evalkit_overview.png)

The second figure shows the same steps part by part. Each box names, in its
corner, the module or package that holds the part, so a rule below can be
found in the code. A module's docstring says what it returns.

![The evaluator part by part: a clip's inputs; scored pixels and the keys of one frame in one view; one frame in every view; one clip; one score file per condition; the tools that compare two conditions. Each box names its module or package.](figures/evaluator_map.png)

## How a frame is scored

### Inputs

- **Ground truth.** A label per pixel, read through the dataset's class
  table the same way for both datasets: the table gives every id a mask can
  hold its colour. CholecSeg8k's masks store colours. ATLAS-120k's store its
  47 original ids, as palette images read by their index, and in 34 clips as
  colours.
- **Prediction.** One region id per pixel, from tracking, with no class; −1
  means no region. The evaluator never reads a class from the pipeline. Each
  condition gives one prediction per clip.
- **Depth.** The Depth Anything 3 (DA3) depth map the pilot evaluator reads
  for the clip, the same for every condition scored on that clip. It sets
  the pixel grid the GT is resized to, and nothing else: this is an
  evaluation of 2D masks, and no pixel is left out because a depth model
  said nothing there. DA3 gives a finite positive depth on every pixel; a
  depth map with a pixel that is not finite or not above `DEPTH_MIN` is a
  fault in the data, and the evaluator refuses the frame rather than scoring
  the rest. It refuses an excluded frame too: the depth map is checked
  before the marker, so that a faulty depth map is found whether or not the
  frame is scored. The pilot evaluator instead scored only the pixels that passed
  that test; pilot mode keeps its test, which is why `invalid_depth` is
  among the counts below (always 0 in normal mode).
- **Crop.** The pipeline cuts each clip to the rectangle around the
  endoscope's circle. The GT mask is cut with the same rectangle before it is
  resized, so that GT and prediction cover the same pixels. The rectangle is an
  input like the mask, and is recorded and hashed with it.
- **Resolution.** GT masks are resized with nearest neighbour to the depth
  map's shape, and so is a prediction of another shape. That shape is DA3's
  input size: the longest side scaled to 504 px, then each side rounded to the
  nearest multiple of 14.
- **Time.** Frames are ordered by their timestamps, not by their file names: in
  11 of the pilot's 27 CholecSeg8k clips the frame numbers do not follow
  time. A dataset that gives no timestamps, ATLAS-120k, is ordered by frame
  number.
  A clip in which two frames share a time, or only some frames have one, is
  refused, since the order of its frames is unknown.
- **Clips and videos.** Which clips enter a measurement is data, kept in
  the population files, never a count in a name. Scores are made per clip;
  the bootstrap that decides a claim resamples videos, so the clips of one
  video are never treated as independent.
- **Frame manifest.** The pipeline writes `frame_manifest.json` into each
  clip. It lists the clip's frames with their times and GT flags, and holds
  the crop rectangle.
- **Propagation rule.** A tracked condition carries its labels from one
  seed frame, under one of two rules. `both_ways_from_centre` seeds on the
  centre of the frames it labels, frame N // 2 of N, and carries both ways;
  it is the offline setting. `forward_from_first` seeds on frame 0 and
  carries forward; it is the causal setting. A condition segmented frame by
  frame has no seed, and its rule is `per_frame`. The tracker records its
  seed frame and its direction beside its labels, in `seed_info.json`, and
  the evaluator reads the rule from there. A seed from anywhere else, the
  GT for one, holds neither rule: such a condition is a comparison of its
  own, and is not scored under either. A score records the rule as
  `propagation`, and a condition whose rule is unknown is not scored.

A clip is a directory. Frame i is `frames[i]` of the frame manifest, counted
from 0, and every input of that frame sits at index i. The index is the
frame's place in the manifest, not in time; "Time" above says how the frames
are ordered. For frame 7:

| input | where | frame 7 |
|---|---|---|
| frame manifest | `frame_manifest.json`, in the clip | `frames[7]`, whose `seq_idx` is 7 |
| depth | `exports/mini_npz/results.npz`, in the clip | `depth[7]` |
| GT mask | `seg_masks/`, in the clip | `000007_color_mask.png` (CholecSeg8k), `000007_class.png` (ATLAS-120k) |
| prediction | the condition's directory for the clip, which the tracking stage writes | `label_0007.npy` |

### Which frames

The frame manifest says which frames are GT frames, and the mask files do
not. A GT frame has the dataset's GT flag (`has_seg_mask` for CholecSeg8k,
`has_gt` for ATLAS-120k), `is_anchor` true and no `seg_provenance`. The mask
files cannot decide: the pipeline also writes the viewer's SAM 3 masks into
`seg_masks/` under the GT's names, and they are not GT. A clip is refused
when a mask file has no GT flag, when a GT flag has no mask file, when a
flagged frame is neither a GT frame nor marked by `seg_provenance`, or when
a GT frame has no prediction.

![Ten frames of a clip in four rows: the raw frames, the GT on six of them, a track seeded from the GT and a track seeded unsupervised. The GT frames are shaded through every row, and the predictions are hatched.](figures/frame_roles.png)

### Objects

In the GT, an object is one class's whole region in one frame. In the
prediction, an object is one region: one id in one frame.

The GT labels classes, not individuals, so a class's region is the finest unit
it can supply. This also settles classes that cover several separate things,
such as ATLAS-120k's `Tools/camera`, used for every instrument: two touching
instruments are one GT object because the dataset gives them one class, not
because they touch. (ATLAS-120k's own benchmark offers no guide here: its AP
scores a frame's whole foreground as one object.)

The definition has two costs:

- A class seen in pieces — a liver cut in two by an instrument — is still one
  object.
- A prediction that separates two things of one class — two instruments, two
  loops of small intestine — gives regions that each cover part of the object.
  At most one of them can be matched; the others count as false positives, and
  `VI_split` counts the same separation. Instruments are the common case: in
  70 % of ATLAS-120k frames with tools, the tool region is in two or more
  pieces of 300 px or more. Tools are outside the geometric view.

Objects are paired greedily, highest IoU first, each object at most once,
whatever its class. Among pairs of equal IoU, the one with the higher GT
object index is taken first, then the higher predicted index; GT objects are
indexed by class id and predicted objects by region id, both ascending. That
is the pilot evaluator's order, kept so that both modes share one rule. A pair
with IoU ≥ `MATCH_IOU` is a *hit*. `F1_50`, `SQ` and `inst_BF` are taken over
these objects.

This figure and the ones below draw one scene and run the modules on it.
Hatched pixels are not scored; "Views" says which.

![Objects on a drawn scene in the geometric view: two GT objects, one per class; three predicted objects, one per region; the pairs taken greedily by IoU, two hits, and F1_50 and SQ computed from them.](figures/objects.png)

[`evalkit/objects.py`](../evalkit/objects.py) computes the numbers on this
scene.

#### Component objects

`evalkit/tools/component_scores` scores every condition on a second kind of
object as well, and `evalkit/tools/track_metrics` follows it over time
("The measures of identity"). Both give reference values that decide no
claim:

- a GT object is one 8-connected piece of one class's scored pixels;
- a predicted object is one region's scored pixels, whole: a region that
  an instrument cuts in two is one object, as it is for the evaluator;
- each is kept only at a minimum size, the same on both sides: 0, 100, 300
  or 400 pixels at the evaluation resolution, each size reported.

The pairing, the hits and `inst_BF` are the evaluator's. A component object
is a piece the frame shows, not a thing: a liver cut in two by an
instrument is two objects, and two touching instruments are one. The pilot
evaluator's objects were component objects of 300 pixels, in its own
domains.

### Naming the regions: the class map

The pipeline gives regions without classes, so `mIoU` and `boundary_F` need a
class for each region. Each region takes the class most of its scored pixels
have in the GT: a tie goes to the smaller id, and a scored pixel with no region
has no class. A region with no scored pixel gets no name and is no object. The
result is the *class map*.

The GT decides the names, so no classifier's mistakes enter `mIoU`: it is an
oracle value, kinder than any real classifier would get, and is reported as
one. It is not a bound in the strict sense: the majority vote maximises the
share of correctly named pixels, and a different naming could score a higher
`mIoU` by favouring small classes. Splitting a class into several regions costs
nothing there; `VI_split` measures splitting.

![The class map on a drawn scene: each region takes the GT class most of its scored pixels have; the gallbladder split into two regions is named gallbladder twice and costs mIoU nothing.](figures/class_map.png)

[`evalkit/classmap.py`](../evalkit/classmap.py) computes the numbers on this
scene.

### Class types

Every class gets exactly one type, listed per dataset in the class tables at
the end:

| type | meaning | scored |
|---|---|---|
| `ignored` | outside the field of view, or not a class at all | never: removed from the GT and the prediction alike |
| `background` | inside the view, labelled as nothing: unlabelled anatomy, and in ATLAS-120k's benchmark classes the ids the 30-class mapping drops | never: removed like `ignored`, so a region lying on it is neither an object nor a false positive (see the views below) |
| `excluded` | a marker that takes the whole frame out of evaluation | the frame is skipped, and counted |
| `tool` | an instrument, or another object that is not anatomy (ATLAS-120k's catheters and non-anatomical structures) | in the `all` view only |
| `tissue` | anatomy that the scene's depth and shape can separate | in every view |
| `appearance` | tissue told apart only by colour or texture (blood, for one) | in the `all` and `tissue` views |
| `expert` | tissue whose boundary is set by anatomical convention — which vessel it is, where one stretch of a tube ends — so that neither shape nor colour shows it without anatomical knowledge | in the `all` and `tissue` views |
| `backdrop` | a surface the scene sits against, close to background (abdominal wall, diaphragm) | in the `all` and `tissue` views |

`expert` and `backdrop` came out of looking at every ATLAS-120k class in the
masks: many classes were neither shape nor colour, but one of these two.

### Views

Every metric but `time_IoU` is computed over each of three views, the same
way:

- `all`: every class that is not `ignored`, `background` or `excluded`
- `tissue`: the same, without `tool`
- `geometric`: `tissue` classes only

A view removes the pixels of the classes it leaves out, from the GT and the
prediction alike, as it does `ignored` pixels: a prediction is neither rewarded
nor penalised there. In the geometric view, a region that runs from the liver
over the blood lying on it is not penalised for the blood, and neither is one
that stops at its edge.

![One drawn scene in the three views: hatched pixels are not scored; the tool leaves the tissue view, and the tool and the blood leave the geometric view. A table counts each removed pixel once, by the first reason that removed it.](figures/scored_pixels.png)

[`evalkit/scored.py`](../evalkit/scored.py) computes the numbers on this
scene.

#### Why removed pixels are not filled in

The removed pixels are not filled in from their neighbours, as if the liver
ran on under the blood. Under a thin smear it does, but a fill would have to
hold for every class a view removes, and under most of them it does not: to
depth, pooled blood is a surface of its own, a tool is another object in
front, and connective tissue runs between organs, so a fill would make up
labels over large areas, by an adjacency rule that would be one more free
choice. It would also lean one way. A pipeline that follows shape keeps the
liver whole across the blood, and one that follows colour cuts the spot out;
a fill would reward the first, the behaviour depth is expected to bring, by a
rule chosen after the pilot evaluator's scores were seen. Removing the pixels
favours neither. What it cannot see is a region lying on removed pixels
only, such as a spot cut out by colour: it is no object, so it is not counted
against the condition as an extra region, and no other metric sees it either.

Where a fill would be defensible at all, a spot of `appearance` pixels whose
whole outer border is one `tissue` class, the GT has almost none. Counted
on the masks as released, before any crop: in CholecSeg8k, such spots hold
0.6 % of the `appearance` pixels and 0.04 % of the `tissue` pixels the
geometric view scores; 6 % of frames have one, and the median spot is 24 px. In ATLAS-120k's population, read every fifth mask,
6 of 21,093 frames have one, holding 0.008 % of the `appearance` pixels.
Removing and filling differ on almost no pixel, so nothing is counted for
the difference.

#### Background

Background is removed in every view, the same way. What the datasets call
background is not empty space: it is anatomy nobody labelled, and in
ATLAS-120k's benchmark classes also the kidney, pancreas and the other classes
the 30-class mapping drops. A region the pipeline places there is not an
error, so it is neither an object nor a false positive, and no metric sees
it. The one surface that is close to nothing, the abdominal wall, is a class
of its own (`backdrop`) and is scored in the `all` and `tissue` views. This is
the pilot evaluator's `labeled` domain, made for the same reason; its `full`
domain, which counted a region over unlabelled anatomy as a false positive, is
not carried over.

So a region that spills from a labelled organ into unlabelled tissue is
scored as if it stopped at the organ's edge: the spill costs nothing. That
is the convention of panoptic quality, where a segment's void pixels leave
the union and a segment lying mostly on void is no false positive. The other
convention, the Cityscapes instance evaluation, keeps the spill in the union
and so penalises it. It is not used here because ATLAS-120k's unlabelled
pixels include organs painted only in part: in some clips a mesentery stops
at a free-hand line while the same fat visibly continues, and there a region
that follows the organ past the paint would be penalised for being right.
What the scores cannot see is counted instead, as `unlabelled_share` below.

## Metrics

| what it measures | key | better | same key in the pilot evaluator | pilot keys it replaces |
|---|---|---|---|---|
| objects found | `F1_50` | higher | `inst_F1_50` | the recognition quality of PQ (panoptic quality) |
| how well found objects fit | `SQ` | higher | `SQ` | `PQ`, `inst_F1_avg`, `inst_F1_75` |
| how well found objects' contours fit | `inst_BF` | higher | `inst_BF` | — |
| class map, by area | `mIoU` | higher | `GT_mIoU` | `GT_mDice` |
| class map, by contour | `boundary_F` | higher | `boundary_F` | `boundary_P`, `boundary_R`, tolerances 1 / 3 / 5 |
| boundaries found before any class is assigned | `boundary_R_raw` | higher | `boundary_R_raw` | `boundary_P_raw`, its tolerances |
| splitting | `VI_split` | lower | `VI_split` | `overseg_mean` |
| merging | `VI_merge` | lower | `VI_merge` | `underseg_error` |
| consistency over time | `time_IoU` | reference only | `time_IoU` | — |
| how much of the regions lies on unlabelled tissue | `unlabelled_share` | reference only | — | — |

### What each key is, per frame

#### Objects found and how well they fit: `F1_50`, `SQ`, `inst_BF`

- `F1_50` = 2 · hits / (GT objects + predicted objects).
- `SQ` is the mean IoU of the hits. `inst_BF` is the mean, over the hits, of
  the boundary F between the predicted object's contour and the GT object's,
  each marked by the boundary rule below over the scored pixels.

The figure under "Objects" shows `F1_50` and `SQ`. The figure of `inst_BF`
comes after the boundary rule it uses, under "Boundaries".

#### The class map, by area: `mIoU`

`mIoU` is the mean IoU over the classes in the GT or the class map,
background excluded. The figure under "Naming the regions: the class map"
shows it.

#### Boundaries: `boundary_F`, `boundary_R_raw`

- `boundary_F` compares the class map's boundaries with the GT's.
  `boundary_R_raw` is the share of the GT's class boundaries that the regions'
  own boundaries recover, before any class is assigned, so an extra cut costs
  it nothing.
- A boundary pixel is one whose left, right, upper or lower neighbour is a
  scored pixel with another label. An edge against a removed pixel (ignored,
  background, or a class the view leaves out) is not a boundary, so a region
  is neither rewarded nor penalised for where it ends against them. Both sides
  of an edge are marked, so a boundary is 2 px wide. The tolerance is a
  square dilation by `BOUNDARY_TOL_PX`. An edge shifted by `BOUNDARY_TOL_PX`
  px scores 1, one shifted by `BOUNDARY_TOL_PX` + 1 px scores 1/2, and one
  further off scores 0. A frame whose scored pixels are all one class has
  no GT boundary, which is common once a view has removed the rest (a frame
  showing only liver, in the geometric view); the boundary metrics are not
  defined on it, and it is counted. When only the prediction's boundary is
  empty, they score 0.

![Boundary pixels on a drawn scene in the geometric view: the GT boundary is only the arc where liver and gallbladder touch; the class map's boundary and the regions' own boundary are scored against it within the tolerance.](figures/boundary.png)

![How far a predicted edge may be off: with a tolerance of 2 px, an edge 2 px off scores 1, one 3 px off scores 1/2, and one 4 px off scores 0.](figures/boundary_tolerance.png)

[`evalkit/boundary.py`](../evalkit/boundary.py) computes the numbers in both
figures.

`inst_BF` applies the same rule to the contour of each hit.

![inst_BF on a drawn scene: the contour of each hit against its GT object's contour, within the tolerance, and inst_BF as the mean F over the hits. With the gallbladder unlabelled, the liver has no contour to recover, and the hit is left out and counted.](figures/inst_bf.png)

[`evalkit/inst_bf.py`](../evalkit/inst_bf.py) computes the numbers on this
scene.

#### Splitting and merging: `VI_split`, `VI_merge`

`VI_split` = H(regions | GT) and `VI_merge` = H(GT | regions), the two halves
of the variation of information, in bits, over the scored pixels. The pixels
with no region count together as one region. On a frame with no scored pixel
they are not defined, and the frame is counted.

![VI on a drawn scene: cutting the gallbladder in two raises VI_split, and cutting the liver as well raises it further; merging liver and gallbladder into one region raises VI_merge. Below, the pixels of each region by GT class.](figures/vi.png)

[`evalkit/vi.py`](../evalkit/vi.py) computes the numbers on this scene.

#### Over time: `time_IoU`

`time_IoU` is a region id's IoU with itself in the next frame. Unlike the
other metrics it is one number per clip: the IoUs of every (id, frame pair)
are pooled over all tracked frames, with or without GT, and averaged.

![time_IoU on two drawn frames: each id present in both frames gives its IoU with itself; an id that disappears or appears gives nothing.](figures/time_iou.png)

[`evalkit/time_iou.py`](../evalkit/time_iou.py) computes the numbers on these
frames.

#### The spill no other key sees: `unlabelled_share`

`unlabelled_share` is, over the regions that have a scored pixel, the share
of their pixels lying on background among their pixels on scored or
background pixels: the spill the other keys cannot see. A region on
background alone is not among them, as it is no object, and pixels the view
removed for another reason are in neither count. It reads no GT class, only
where the GT is unlabelled, and decides no claim. It is reported beside a
comparison whose two conditions differ in it by much, as the pilot
measurements did for the share of pixels left without a region.

![unlabelled_share on a drawn scene: a region that fits the liver and one that spills into unlabelled tissue get the same value on every other key; unlabelled_share is 0 for the first and 0.552 for the second.](figures/unlabelled_share.png)

[`evalkit/unlabelled.py`](../evalkit/unlabelled.py) computes the numbers on
this scene.

### From frames to clips

Every metric but `time_IoU` is computed per frame and averaged over a clip's
GT frames; `time_IoU` is pooled over the clip as described above. The JSON
keeps the per-frame values.

The clip values decide a claim. `paired_stats` takes the 95 % confidence
interval of the difference with a bootstrap that resamples whole videos. A
table then prints:

- ★ when the interval lies wholly on the key's better side of zero;
- ✗ when it lies wholly on the worse side;
- neither when it contains zero, when the key is a reference value, or when
  the interval rests on fewer than six videos.

An interval on fewer than six videos is read for its sign only, and the
table says so. `paired_stats.MIN_VIDEOS_FOR_MARK` holds the number and the
reason.

The p-value is reported too, and never decides. `paired_stats.mark_of` is
the one code that applies this rule.

A metric is not defined on some frames: `F1_50` on a frame with no GT object,
`SQ` and `inst_BF` on a frame with no hit, `mIoU` on a frame with no class,
`unlabelled_share` on a frame with no region on a scored pixel. `inst_BF`
also leaves out a hit whose GT object has no boundary pixel, because its
contour lies wholly against removed pixels (the one object of a frame that
fills the scored pixels, or one cut off from every other class by a tool),
and is not defined when no hit remains; the hits left out are counted.
Those frames do not enter the mean, and the number of frames each mean covers
is recorded. `time_IoU` is not defined on a clip where nothing is pooled: a
clip of one frame, or one in which no id is present in two consecutive frames
with some valid pixel under it. Two conditions can cover different frames, and comparing them
without the counts once flipped the sign of a pilot result.

PQ is not stored. When a table wants it, it is SQ × F1_50 per frame, taken from
the per-frame values, with one rule that the product alone would miss: a frame
that has GT objects but no hit has PQ = 0, although its `SQ` is not defined.
Leaving such frames out would lift PQ above its usual definition, which the
pilot evaluator follows. Counts (frames, objects, regions) are kept as
diagnostics and decide no claim.

### Primary metrics

The paper asks three questions, each as a comparison between two
conditions, and judges two of them on a key of their own, on each dataset's
own labels (the `original` class set). The table in the summary lists the
questions and their primary metrics.

![The three questions on a drawn scene. Q1: a region picked out in one frame follows the gallbladder for two frames, then leaves it. Q2: the regions' boundaries against the GT's class boundaries, giving boundary_R_raw. Q3: regions matched to the GT's objects, giving F1_50 and SQ.](figures/three_questions.png)

The question about boundaries is answered within each view, between
conditions that differ in what the pipeline is given: the image, the depth,
or both. The views sort the classes by what should separate them, so the
answer can differ from one view to the next.

These were chosen before any score of this evaluator was looked at. The
pilot evaluator's scores of the same quantities have been seen, which is why
the choice is fixed before this evaluator scores anything.

The set is kept as small as the claims allow. The evaluator computes every
metric in the metrics table; which of them the paper reports is settled
before any score of this evaluator is seen.

### Consistency over time: reference values only

`time_IoU` is kept only as a reference value and decides no claim. It uses no
GT, and the workbench measured three faults in it:

- it rises as regions get coarser, and one region covering the whole frame
  scores 1.0;
- an id that disappears costs nothing;
- for a condition segmented frame by frame, whose ids do not carry over from
  one frame to the next, it means nothing.

`temporal_f1`, built to close the second fault, keeps the first.

A tool outside the evaluator, `evalkit/tools/track_metrics`, computes the
measures of identity against the GT: hold, IDF1, ID switches, fragmentation
and re-entry. They are reported as reference values, in a table without ★
or ✗. A measure that is to decide a claim would have to be computed by the
evaluator before the evaluator is frozen, and none is.

#### The measures of identity

A *GT track* follows one GT object over a clip's GT frames, in one view.
Frames without GT and excluded frames are left out of the sequence. The
datasets carry no ids for individual things, so the tool builds the tracks
in two ways and reports both:

- on whole-class objects ("Objects"), each class is one track;
- on component objects ("Component objects"), at each of the four sizes, a
  piece is linked to a track of its class when its IoU with the track's
  last piece is at least 0.3 and the track has gone at most three GT frames
  without a piece. The pairs are taken greedily, highest IoU first, each
  piece and each track once. A piece left over starts a track. These are
  the workbench's values.

A track is *present* on a frame when it has an object there. On each frame
the pairing of "Objects" pairs the tracks' objects with the predicted
objects, and a hit pairs a track with a region. On component tracks a
predicted object is a region with at least the size's pixels. Each measure
is computed per clip, per view and per kind of track:

- **hold** follows each track present on the seed frame, the frame
  `seed_info.json` names. The region *picked* for a track is the one the
  seed frame's pairing hits it with. On every other frame on which the
  track is present, the track gives one sample: the IoU of the picked
  region with the track's object, whatever the region's size, and 0 where
  the two do not overlap. `hold_mean` averages the picked tracks' samples.
  `hold_all_mean` averages every sample, a track without a picked region
  giving 0. `picked_share` is the share of the seed frame's tracks that
  have a picked region. `hold_curve` bins the samples of `hold_mean` by
  their distance in seconds from the seed frame.
- **IDF1** gives each track one region over the clip, and each region one
  track: the assignment with the most frames on which the pair is a hit.
  IDF1 = 2 · those frames / (GT objects + predicted objects, summed over
  the frames).
- **ID switches** counts the hits by another region than the track's
  previous hit.
- **fragmentation** counts each frame on which a track is present and not
  hit, when its previous frame with the track was a hit.
- **re-entry**: a gap is a run of frames on which a track is absent,
  between two on which it is present. `reentry` is the share of the gaps
  hit by one region at both ends, among the gaps whose two ends are hits.
  A whole-class track's gap has any length. A component track's gap has
  at most three frames, since the linking rule waits no longer.

Hold starts on the seed frame. The tool refuses:

- a clip whose seed frame has no GT or is excluded;
- a `per_frame` condition, which has no seed frame;
- a condition seeded from GT or from outside the tracking stage, which
  holds no propagation rule ("Propagation rule");
- labels without `seed_info.json`;
- a condition whose clips hold two rules.

Under `both_ways_from_centre` the samples lie before and after the seed
frame, and under `forward_from_first` after it only; as with every key, the
two rules are compared only between conditions that differ in the rule alone.

The workbench measured a fault in four of the measures, on component tracks
of 400 px:

- `hold_mean` and `hold_all_mean` gave one comparison opposite signs;
- IDF1 rises when regions merge;
- ID switches cannot tell a tracker from a floor that never moves: one mask
  pasted on every frame scores close to zero;
- re-entry leaves out every gap whose two ends are not both hits: about a
  third of the gaps for a tracker, and nine in ten for the pasted floor,
  which then scored 1.0 on the rest.

## Thresholds, refusals and provenance

### Thresholds

| name | value | why |
|---|---|---|
| `BOUNDARY_TOL_PX` | 2 | the pilot evaluator's value for its main boundary keys; what it allows is given with the boundary definition above |
| `DEPTH_MIN` | 10⁻⁶ | the pilot evaluator's test of a depth value, finite and above this; in normal mode a pixel that fails it refuses the frame, in pilot mode it is masked out |
| `MATCH_IOU` | 0.5, as IoU ≥ 0.5 after greedy pairing | the pilot evaluator's rule. PQ's convention, IoU > 0.5, makes a pairing unique; at exactly 0.5 the greedy order given with the objects decides |

There is no minimum object size. The pilot evaluator dropped connected
components and predicted regions under 300 px. With whole-class objects, a
sliver joins its class's region instead of becoming an object, so a cut would
only decide whether a small class counts at all, and nothing in the data sets
that size. Pilot mode keeps the cut, as `PILOT_MIN_CC_PX`, and nothing else
uses it. The component objects are scored at four sizes
instead ("Component objects").

### Fail closed

- A colour or id missing from the dataset's class table raises; it is never
  mapped to background. ATLAS-120k's benchmark code sends unknown ids to
  background, and reads a mask through `.convert("L")`, which turns a palette
  mask into the brightness of its palette colours rather than its ids; it is
  not reused as code.
- A palette mask is read by its index, never through the palette it embeds.
  ATLAS-120k's masks embed seven different palettes, and some of them give
  some of ids 43–46 black, or Ligated plexus the colour of Liver.
- A mask reaches the table as the image its file holds, not as an array: an
  array does not say whether its channels are RGB or BGR, and six pairs of
  ATLAS-120k colours swap under that mistake, Artery and Vein among them.
- A single-channel CholecSeg8k mask is refused: those are the watershed
  masks, whose codes are not the table's ids. An RGBA mask is read only when
  every pixel is opaque; four CholecSeg8k colour masks are RGBA, alpha 255.
- The CholecSeg8k table maps (0, 50, 128) to Hepatic Vein. The dataset's
  watershed codes settle it (`docs/pilot_evaluator.md`, "What the pilot
  evaluator's handling changes, measured on the GT").
- (255, 255, 255), the line CholecSeg8k draws between regions, and
  CholecSeg8k's Black Background are `ignored`.
- `Excluded frames` is checked on ATLAS-120k's original ids, before they are
  mapped to the 30 classes, which would turn it into background. In a mask
  stored as colours it cannot be seen: its colour in the dataset's palette is
  Background's, (0, 0, 0). There (0, 0, 0) reads as Background, and a colour
  mask of background alone, the one frame the marker could hide in, is
  refused. The marker occurs in no palette mask of the release, and no colour
  mask is background alone.
- Nothing is dropped without a count: skipped frames, ignored, background and
  invalid pixels, the pixels each view removes, and the frames a metric is not
  defined on are counted in the JSON. A removed pixel is counted once, by the
  first reason that removes it, in this order: invalid depth, then `ignored`,
  then `background`, then a class the view leaves out. So a frame's counts
  are disjoint and sum to its pixels. Invalid depth is counted in pilot mode
  only, since normal mode refuses a frame with invalid depth. In pilot mode,
  a black corner without depth counts as invalid depth, not as ignored. The
  order is a convention: another one would change the counts and no score,
  but the JSON is read against this one.

### Recorded with every score

- `eval_code_sha`, which covers the class tables as well as the code, so a
  changed type is a new evaluator
- the dataset, the class set (`original`, or for ATLAS-120k also
  `benchmark`), the views, and whether the score was made in pilot mode
- a sha256 hash of every input read, per clip, as `input_shas`: the GT
  masks, the depth maps (as `atlas120k_meta/depth_manifest.json`
  fingerprints them), the crop rectangle, the frames in time order, and the
  predictions
- the name of the directory the predictions were read from, as
  `track_dir_name`, which is how `condition_inventory` matches a score to
  its labels; a score that does not say what it scored cannot be inventoried
- the condition's propagation rule, as `propagation` ("Propagation rule"
  above)
- the Python, numpy, OpenCV and Pillow versions

The pilot evaluator's JSONs recorded no input hashes, propagation rule or
versions.

A score JSON holds those fields, `clips` (the population, in its order),
and one row per clip under `per_clip`. A row holds the clip's name, every
key (`metric/view`, then `time_IoU`), and the counts behind them: `n_frames`,
the frames each key's mean covers, by key; the scored and excluded frames;
`pixels`, where each view's pixels went; and `objects`, the GT objects,
predicted objects, hits and hits that entered `inst_BF`, per view. Its
`frames` keeps every GT frame's values, by frame number, in time order.

Two scores are comparable only when all of these requirements hold:

- they share a ruler (Terms);
- they cover the same clips; `compare_eval --allow-subset` compares the
  common clips instead, and records the population as `intersection`;
- they read the same GT masks, depth maps, crop rectangle and frame order:
  both record a hash of each of these in `input_shas`, and every hash but
  the predictions' is equal;
- they share a propagation rule, or one of them is `per_frame`, or they
  differ in the rule alone.

Pilot JSONs record no input hashes and no propagation rule, so between two
of them only the ruler and the clips are compared.

The same `eval_code_sha` is not enough on its own. Pilot mode and the
normal mode share it, and so do the views.

Conditions under two different propagation rules are compared only when
they differ in the rule alone: their names (`condition_names`, read from the
`track_dir_name` each score records) are equal in every other part. In any
other pair, the difference between the rules would look like a difference
between the methods. A `per_frame` condition is the exception: it may be
compared with a condition of either rule. The paper compares tracking with
per-frame segmentation, and the two rules with each other. So one table
holds one rule, or two rules where every condition has, in the table, the
condition that differs from it in the rule alone; `per_frame` conditions
may go in either. A pair or a table of two rules records both as
`propagation`, joined by ` and `.

The tools enforce this:

- `compare_eval` refuses a pair that is not comparable.
- `paired_stats` refuses such a pair too, and a table that holds two rules
  other than `per_frame` where a condition lacks the one that differs from
  it in the rule alone.
- `paired_table` and `arms_paired` refuse a table that holds two
  conditions that are not comparable.
- `claims_table` refuses a column in which any two score JSONs are not
  comparable.
- All five report a difference in library versions but do not refuse it.

The check against the pilot evaluator, below, is not bound by these
requirements: the two `eval_code_sha` differ by construction. Its own script,
outside `compare_eval`, verifies that this evaluator in pilot mode writes the
pilot evaluator's numbers.

### Checked against the pilot evaluator

- Pilot mode runs this evaluator with the pilot evaluator's rules:
  - ATLAS-120k's 47 original ids;
  - the pilot evaluator's CholecSeg8k colour table, which maps unknown colours
    to background — the one place anything becomes background silently,
    allowed because pilot mode exists only for this check;
  - the pilot evaluator's four domains (`full`, `labeled`, `tissue`,
    `labeled_tissue`, with its instrument ids) in place of the views, for the
    instance metrics; the class map and the boundary metrics on the `full`
    domain as the pilot evaluator computed them, with background pixels
    voting on a region's name and an edge against background counted as a
    boundary; VI on the `labeled` domain, the valid pixels whose GT is not
    background, which is the mask the pilot evaluator computed it on, with
    its `extra_ignore` set empty, as every score that records it has it
    (the workshop's; the 38 conditions' JSONs are checked before pilot mode
    relies on it), so the pilot's one `VI_split` is the evaluator's
    `VI_split/labeled`, and the same for `VI_merge`;
  - per-class 8-connected components of at least `PILOT_MIN_CC_PX` as GT
    objects, and regions of at least that size as predicted objects;
  - the pixels whose depth is not finite or not above `DEPTH_MIN` masked
    out and counted, where normal mode refuses the frame;
  - frames in file order for `time_IoU`;
  - the pilot evaluator's values in place of undefined ones, zeros where it
    wrote zeros and None where it wrote None: a frame with no class enters
    the `mIoU` mean as 0; a clip on which no frame has a GT object in a
    domain writes `F1_50` as 0 in the `full` domain and as None in the other
    three, and `SQ` and `inst_BF` as None in every domain, there being no
    hit to average; a frame whose GT boundary is empty scores 0 on the
    boundary metrics rather than being left out; and a clip on which
    `time_IoU` pools nothing writes 0 for it.
- Pilot mode reads a clip as normal mode does, and refuses every input
  normal mode refuses. The pilot evaluator accepted some faulty inputs
  silently, a frame with two prediction files among them; pilot mode does
  not, because a fault in the data is not a rule to reproduce. So the GT
  frames come from the frame manifest too. On the data the pilot evaluator
  scored, its mask files were exactly these frames.
- On the 38 conditions already scored, apart from those that hold neither
  propagation rule, pilot mode must reproduce every key it shares with the
  pilot evaluator — the metrics table names them, and their per-domain
  variants — at zero tolerance: the values written must be equal, and so
  must the number of frames behind each. Ties are broken as the pilot
  evaluator breaks them. The check runs where the pilot evaluator and its
  scores are, and takes their paths as arguments.
- The check passed on 2026-10-11, for `eval_code_sha` `a5d7ea46…`. On the
  23 ATLAS-120k conditions that hold a propagation rule, 315 clips each,
  every shared value was equal, and so was every number of frames the
  pilot evaluator wrote.
- Every difference in the normal mode then comes from a rule this document
  changes, and is listed.
- A score made in pilot mode is marked as such and never enters a comparison
  with a normal one.
- Pilot mode reads a condition's propagation rule as normal mode does. So it
  refuses a condition that holds neither rule, a condition seeded from GT
  among them, and the check leaves such a condition out by name. The paper
  reports none of them, and the other conditions run the same metric code.
- Pilot mode exists for this check alone. It is removed from the evaluator
  once the check has passed and before the evaluator is frozen, so the
  frozen evaluator has one mode.

## Class tables

Each table is a data file, one per dataset: every class's type, the colour of
every id a mask can hold, and for ATLAS-120k the mapping of its 47 ids to the
30 classes. The files are hashed into `eval_code_sha` with the code.

### CholecSeg8k

Every class was looked at in the masks, as for ATLAS-120k:

| id | class | colour | type |
|---:|---|---|---|
| 0 | Black Background | (127, 127, 127) | ignored: the endoscope's black surround, outside the view |
| 1 | Abdominal Wall | (210, 140, 140) | backdrop, as in ATLAS-120k |
| 2 | Liver | (255, 114, 114) | tissue |
| 3 | Gastrointestinal Tract | (231, 70, 156) | tissue |
| 4 | Fat | (186, 183, 75) | tissue |
| 5 | Grasper | (170, 255, 0) | tool |
| 6 | Connective Tissue | (255, 85, 0) | appearance |
| 7 | Blood | (255, 0, 0) | appearance |
| 8 | Cystic Duct | (255, 255, 0) | expert |
| 9 | L-hook Electrocautery | (169, 255, 184) | tool |
| 10 | Gallbladder | (255, 160, 165) | tissue |
| 11 | Hepatic Vein | (0, 50, 128) | expert |
| 12 | Liver Ligament | (111, 74, 0) | tissue |
| 13 | Region line | (255, 255, 255) | ignored: the line drawn between regions. Not a class of the dataset; the table gives it an id so that a GT id map can hold it, and nothing scores it |

Cystic Duct is `expert`, as ATLAS-120k's Cystic duct is: the class ends where
the anatomical stretch ends. Only ATLAS-120k's benchmark, which merges the
ducts into Bile/lymph duct, types them `tissue`. Hepatic Vein is `expert` because telling it from other
vessels takes anatomical knowledge. Neither moves much: Cystic Duct occurs in 3
of the 17 videos, Hepatic Vein in 1.

### ATLAS-120k: ids, classes and colours

ATLAS-120k's label table, `atlas120k_tools/classes.py` in the ATLAS
repository, lists 47 ids, 0 to 46: Background and 46 labels. Every mask of the
release was read: 119,405 masks in 492 of its 502 clips (the other 1,613
frames its clip index lists ship without a mask).

| | |
|---|---|
| ids that occur | every id from 0 to 46 but Hepatic vein (15), Thoracic duct (38), Nerves (39) and Excluded frames (42); none above 46 |
| classes | the 42 labels that occur are the paper's 42 classes, and its Fig. 2 legend lists exactly them |
| storage | 115,024 palette masks, whose index is the id, and 4,381 RGB masks in 34 clips of 7 videos; 7 of those clips hold both kinds |
| colours of the RGB masks | 20 colours, each the one the label table gives its id. In one video, 441 frames belong to two adjacent clips, stored once as a palette mask and once in colour, and both read as the same ids |
| palettes the palette masks embed | seven versions. Against the label table, five give some of ids 43–46 black, one gives Ligated plexus (28) the colour of Liver, and two give Pancreas (252, 186, 3) |

So 47 is the number of ids the masks are written in, and 42 classes and
background are what they contain. The class table lists all 47, each with its
colour, its type and the benchmark class it merges into. Excluded frames has no colour there, because its
colour in the label table is Background's.

The paper describes how the labels were made: the first frame of every clip
was drawn by hand and reviewed by a surgeon, and the rest were propagated by a
video object segmentation model (Cutie) and corrected by hand. A GT frame after
the first is therefore a tracker's output as the annotators corrected it,
which is worth knowing when a tracking pipeline is scored against it.

### ATLAS-120k: the original ids and their types

Claims are judged on the 47 original ids, the `original` class set. ATLAS-120k
defines no types: its benchmark scores `Tools/camera` like any other class.
Every one of the 47 ids was looked at in the masks, and each takes its own
verdict as its type, "unsure" counted as `expert`, unless the note gives a
reason for another: Abdominal wall and Diaphragm, "almost background", are
`backdrop`, and Catheter, which is not anatomy, is a `tool`. Hepatic vein (15),
Thoracic duct (38) and Nerves (39) occur in no mask of the release; they take
the type of the benchmark class they merge into, which no score depends on.

| id | label | benchmark class | verdict, note | type |
|---:|---|---|---|---|
| 0 | Background | Background | — | background |
| 1 | Tools/camera | Tools/camera | tool | tool |
| 2 | Vein (major) | Vein | unsure, "hard even from colour" | expert |
| 3 | Artery (major) | Artery | unsure, "uses shape, but needs a lot of expertise" | expert |
| 4 | Nerve (major) | Nerve | appearance, "very faint" | appearance |
| 5 | Small intestine | Small intestine | tissue, "plainly shape" | tissue |
| 6 | Colon/rectum | Colon/rectum | tissue, "fairly plainly shape" | tissue |
| 7 | Abdominal wall | Abdominal wall | unsure, "almost background" | backdrop |
| 8 | Diaphragm | Diaphragm | unsure, "almost background" | backdrop |
| 9 | Omentum | Fat | tissue, "a coherent shape" | tissue |
| 10 | Aorta | Artery | tissue, "only faintly shape" | tissue |
| 11 | Vena cava | Vein | appearance, "colour works, but it takes expertise" | appearance |
| 12 | Liver | Liver | tissue | tissue |
| 13 | Cystic duct | Bile/lymph duct | unsure, "mostly shape, but where the stretch ends takes expertise" | expert |
| 14 | Gallbladder | Gallbladder | tissue | tissue |
| 15 | Hepatic vein | Vein | in no mask | expert |
| 16 | Hepatic ligament | Hepatic ligament | tissue | tissue |
| 17 | Cystic plate | Cystic plate | appearance | appearance |
| 18 | Stomach | Stomach | tissue | tissue |
| 19 | Ductus choledochus | Bile/lymph duct | unsure, "mostly shape, but where the stretch ends takes expertise" | expert |
| 20 | Mesenterium | Fat | tissue, "barely shape" | tissue |
| 21 | Ductus hepaticus | Bile/lymph duct | appearance | appearance |
| 22 | Spleen | Spleen | tissue | tissue |
| 23 | Uterus | Uterus | tissue | tissue |
| 24 | Ovary | Ovary | tissue | tissue |
| 25 | Oviduct | Oviduct | tissue | tissue |
| 26 | Prostate | Prostate | tissue | tissue |
| 27 | Urethra | Urethra | tissue | tissue |
| 28 | Ligated plexus | Ligated plexus | appearance | appearance |
| 29 | Seminal vesicles | Seminal vesicles | tissue, "shape, probably" | tissue |
| 30 | Catheter | Non anatomical | unsure, "shape as a rule, but invisible when buried in tissue" | tool |
| 31 | Bladder | Bladder | tissue, "sometimes by shape, sometimes not" | tissue |
| 32 | Kidney | Background | tissue | tissue |
| 33 | Lung | Lung | tissue | tissue |
| 34 | Airway (bronchus/trachea) | Airway (bronchus/trachea) | unsure, "shape cannot make this division; specialist" | expert |
| 35 | Esophagus | Esophagus | tissue | tissue |
| 36 | Pericardium | Pericardium | unsure, "specialist" | expert |
| 37 | V azygos | Vein | appearance, "colour might do" | appearance |
| 38 | Thoracic duct | Bile/lymph duct | in no mask | tissue |
| 39 | Nerves | Nerve | in no mask | appearance |
| 40 | Ureter | Background | unsure | expert |
| 41 | Non anatomical structures | Non anatomical | tool, "shape, as a rule" | tool |
| 42 | Excluded frames | Background | a marker, not a class | excluded |
| 43 | Mesocolon | Background | tissue | tissue |
| 44 | Adrenal gland | Background | tissue | tissue |
| 45 | Pancreas | Background | unsure, "neither shape nor colour settles it" | expert |
| 46 | Duodenum | Background | appearance, "neither shape nor colour settles it" | appearance |

The review's main difficulties were stretches of one tube (cystic duct against
ductus choledochus) and one kind of vessel (vena cava against V azygos), told
apart by anatomical convention. Those ids are `expert` or `appearance`, so the
geometric view leaves them out: for the original ids the types do what the
merge does for the benchmark's classes. Cystic duct is `expert` here as
CholecSeg8k's Cystic Duct is.

### ATLAS-120k: the benchmark's 30 classes

ATLAS-120k's own model and benchmark code score 30 classes. The benchmark maps
every mask before scoring, through `datasets/class_mapping.py` in the
ATLAS-bench repository, read at commit `e286a584`: seven ids become
background, and similar classes merge. The table below lists the original ids
each class takes. The class set is then the benchmark's: Background and 29
classes. It is scored as a second class set, `benchmark`, for comparison with
the benchmark: written to a separate block of the JSON, as reference values
only, which decide no claim. Resolution, crop and metrics are this
evaluator's own, so the numbers are not the benchmark's.

`python -m evalkit.tools.check_benchmark_mapping <class_mapping.py>`
compares the class table's mapping with the one that file defines, id by
id, and names each id the two map differently. At `e286a584` all 47 ids
agree.

The paper gives the reason for the 30: categories not represented in all
subsets were excluded, and semantically similar classes consolidated. In the
released splits that rule does not single out exactly the six anatomical
classes dropped: Uterus, Ovary and Oviduct are in neither the validation nor
the test split and are kept, and Diaphragm is not in the training split. The
evaluator takes the mapping as the benchmark's code defines it. The reason is
a training one, which is why claims are judged on the original ids: for a
pipeline trained on nothing, the dropped classes are anatomy like any other.

A class that is one original id has that id's type. A merged class has one
type for all its ids:

| # | class | original ids | type |
|---:|---|---|---|
| 0 | Background | Background, Kidney, Ureter, Excluded frames, Mesocolon, Adrenal gland, Pancreas, Duodenum | background |
| 1 | Tools/camera | Tools/camera | tool |
| 2 | Vein | Vein (major), Vena cava, Hepatic vein, V azygos | expert: telling one vessel from another takes anatomy |
| 3 | Artery | Artery (major), Aorta | expert: the same |
| 4 | Nerve | Nerve (major), Nerves | appearance |
| 5 | Small intestine | Small intestine | tissue |
| 6 | Colon/rectum | Colon/rectum | tissue |
| 7 | Abdominal wall | Abdominal wall | backdrop |
| 8 | Diaphragm | Diaphragm | backdrop |
| 9 | Fat | Omentum, Mesenterium | tissue |
| 10 | Liver | Liver | tissue |
| 11 | Bile/lymph duct | Cystic duct, Ductus choledochus, Ductus hepaticus, Thoracic duct | tissue: the merge removes the stretch boundaries |
| 12 | Gallbladder | Gallbladder | tissue |
| 13 | Hepatic ligament | Hepatic ligament | tissue |
| 14 | Cystic plate | Cystic plate | appearance |
| 15 | Stomach | Stomach | tissue |
| 16 | Spleen | Spleen | tissue |
| 17 | Uterus | Uterus | tissue |
| 18 | Ovary | Ovary | tissue |
| 19 | Oviduct | Oviduct | tissue |
| 20 | Prostate | Prostate | tissue |
| 21 | Urethra | Urethra | tissue |
| 22 | Ligated plexus | Ligated plexus | appearance |
| 23 | Seminal vesicles | Seminal vesicles | tissue |
| 24 | Non anatomical | Catheter, Non anatomical structures | tool |
| 25 | Bladder | Bladder | tissue |
| 26 | Lung | Lung | tissue |
| 27 | Airway (bronchus/trachea) | Airway (bronchus/trachea) | expert |
| 28 | Esophagus | Esophagus | tissue |
| 29 | Pericardium | Pericardium | expert |

Vein and Artery are `expert`: Vein's ids were judged unsure or appearance,
Artery's unsure or tissue, and what every note shares is that telling one
vessel from another takes anatomy. Bile/lymph duct is `tissue`, although two
of its ids are `expert`, because the merge removes the stretch boundaries
that made them so.
