# Evaluation — specification

**Status: agreed, not yet implemented.** Nothing here is frozen yet. One thing
is still undecided: how to score consistency over time (the last point of the
summary).

## Summary

This document specifies how the paper's segmentation results are scored. The
pipeline splits every video frame into regions and tracks them over time,
without naming what they are. The evaluator compares those regions with the
ground-truth (GT) masks of two datasets: ATLAS-120k (laparoscopic videos of
many procedures, 47 classes) and CholecSeg8k (laparoscopic cholecystectomy,
13 classes).

- **One object per class in the GT, one per region in the prediction.** The
  datasets label classes, not individual things, so all pixels of one class in
  one frame form one GT object. Each predicted region is one predicted object.
  A GT object and a predicted object are paired when they overlap enough
  (IoU ≥ 0.5).
- **Every class has a type**: a tool; tissue that depth and shape can
  separate; or tissue they cannot, or that is nearly background, such as blood
  or the abdominal wall. Every metric is computed over three sets of classes,
  called *views*: all classes; tissue only, without tools; and *geometric*,
  only the tissue that depth and shape can separate.
- **ATLAS-120k is scored with its own 30 classes**, the ones its benchmark
  uses. Its 47 original classes are scored too, as reference values only.
- **Nine metrics.** The two *primary* metrics, on which the paper's claims are
  judged, are `F1_50` (the share of objects found) and `SQ` (how well the found
  ones fit), in the geometric view.
- **Regions are named from the GT.** Each region takes the class most of its
  pixels have in the GT, so the class-map metric `mIoU` is an upper bound.
- **Nothing is dropped silently.** A colour or class id the tables do not know
  stops the run, and every skipped frame or pixel is counted. There is no
  minimum object size.
- **Scores say how they were made.** Each carries `eval_code_sha`, a hash of the
  evaluator's code and class tables, with its settings and hashes of its
  inputs. Two scores are compared only when these match.
- **Checked against the earlier evaluator.** The pilot measurements were scored
  by an earlier evaluator, the *pilot evaluator* (`eval_code_sha = 1f8a813a…`),
  which stays frozen in the private research workbench. Run with its rules
  (*pilot mode*), this evaluator must reproduce its scores exactly.
- **Not decided yet: consistency over time.** The pilot evaluator's `time_IoU`
  is kept as a reference value only, because coarse regions score well on it.
  Measures of whether a tracked thing keeps its identity — hold, IDF1, ID
  switches, fragmentation — are candidates, and all are reference values for
  now.

The rest of the document gives the rules in full, then why the pilot evaluator
was replaced, then the class tables.

## How a frame is scored

### Inputs

- **Ground truth.** A class per pixel: ATLAS-120k's 47 original ids, or
  CholecSeg8k's colours read through its class table.
- **Prediction.** One region id per pixel, from tracking, with no class; −1
  means no region. The evaluator never reads a class from the pipeline. Each
  configuration of the pipeline that the paper compares — a *condition* —
  gives one prediction per clip.
- **Valid pixels.** Only pixels whose depth is finite and positive are scored.
  The depth is the Depth Anything 3 (DA3) depth map the pilot evaluator reads
  for the clip, the same for every condition scored on that clip.
- **Resolution.** GT masks are resized with nearest neighbour to the depth
  map's shape, and so is a prediction of another shape. That shape is DA3's
  input size: the longest side scaled to 504 px, then each side rounded to the
  nearest multiple of 14.
- **Time.** Frames are ordered by their timestamps, not by their file names: in
  11 of the 27 CholecSeg8k clips the frame numbers do not follow time.

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
whatever its class. A pair with IoU ≥ `MATCH_IOU` is a *hit*. `F1_50`, `SQ` and
`inst_BF` are taken over these objects.

### Naming the regions: the class map

The pipeline gives regions without classes, so `mIoU` and `boundary_F` need a
class for each region. Each region takes the class most of its scored pixels
have in the GT: a tie goes to the smaller id, background takes part in the
vote, and pixels with no region are background. The result is the *class map*.

The GT decides the names, so they are the best the regions allow. `mIoU` is
therefore an upper bound on the class map, and is reported as one. Splitting a
class into several regions costs nothing there; `VI_split` measures splitting.

### Class types

Every class gets exactly one type, listed per dataset in the class tables at
the end:

| type | meaning | scored |
|---|---|---|
| `ignored` | outside the field of view, or not a class at all | never: removed from the GT and the prediction alike, like a pixel without valid depth |
| `background` | inside the view, labelled as nothing | never |
| `excluded` | a marker that takes the whole frame out of evaluation | the frame is skipped, and counted |
| `tool` | an instrument | in the `all` view only |
| `tissue` | anatomy that the scene's depth and shape can separate | in every view |
| `appearance` | tissue told apart only by colour or texture (blood, for one) | in the `all` and `tissue` views |
| `expert` | tissue whose boundary is set by anatomical convention — which vessel it is, where one stretch of a tube ends — so that neither shape nor colour shows it without anatomical knowledge | in the `all` and `tissue` views |
| `backdrop` | a surface the scene sits against, close to background (abdominal wall, diaphragm) | in the `all` and `tissue` views |

`expert` and `backdrop` came out of looking at every ATLAS-120k class in the
masks: many classes were neither shape nor colour, but one of these two.

### Views

Every metric is computed over each of three views, the same way:

- `all`: every class that is not `ignored`, `background` or `excluded`
- `tissue`: the same, without `tool`
- `geometric`: `tissue` classes only

A view removes the pixels of the classes it leaves out, from the GT and the
prediction alike, as it does `ignored` pixels: a prediction is neither rewarded
nor penalised there. In the geometric view, a region that runs from the liver
over the blood lying on it is not penalised for the blood, and neither is one
that stops at its edge.

## Metrics

| what it measures | key | same key in the pilot evaluator | pilot keys it replaces |
|---|---|---|---|
| objects found | `F1_50` | `inst_F1_50` | the recognition quality of PQ (panoptic quality) |
| how well found objects fit | `SQ` | `SQ` | `PQ`, `inst_F1_avg`, `inst_F1_75` |
| how well found objects' contours fit | `inst_BF` | `inst_BF` | — |
| class map, by area | `mIoU` | `GT_mIoU` | `GT_mDice` |
| class map, by contour | `boundary_F` | `boundary_F` | `boundary_P`, `boundary_R`, tolerances 1 / 3 / 5 |
| boundaries found before any class is assigned | `boundary_R_raw` | `boundary_R_raw` | `boundary_P_raw`, its tolerances |
| splitting | `VI_split` | `VI_split` | `overseg_mean` |
| merging | `VI_merge` | `VI_merge` | `underseg_error` |
| consistency over time, reference only | `time_IoU` | `time_IoU` | — |

### What each key is, per frame

- `F1_50` = 2 · hits / (GT objects + predicted objects).
- `SQ` is the mean IoU of the hits, and `inst_BF` their mean boundary F.
- `mIoU` is the mean IoU over the classes in the GT or the class map,
  background excluded.
- `boundary_F` compares the class map's boundaries with the GT's.
  `boundary_R_raw` is the share of the GT's class boundaries that the regions'
  own boundaries recover, before any class is assigned, so an extra cut costs
  it nothing.
- A boundary pixel is one whose left, right, upper or lower neighbour has
  another label. Both sides of an edge are marked, so a boundary is 2 px wide,
  and the tolerance is a square dilation by `BOUNDARY_TOL_PX`: one-sided, a
  boundary may be off by `BOUNDARY_TOL_PX` + 1 px. An empty boundary scores 0.
- `VI_split` = H(regions | GT) and `VI_merge` = H(GT | regions), the two halves
  of the variation of information, in bits, over the scored pixels that are
  not background. Pixels with no region count as a region of their own.
- `time_IoU` is each region id's IoU with itself in the next frame, averaged
  over the ids present in both frames.

### From frames to clips

Every metric is computed per frame and averaged over a clip's GT frames. The
clip means are what `paired_stats` resamples, by video, to decide whether a
claim gets a star: the rule in `AGENTS.md`, a video-level bootstrap 95 % CI
that does not straddle zero. The JSON keeps the per-frame values.

A metric is not defined on some frames: `F1_50` on a frame with no GT object,
`SQ` and `inst_BF` on a frame with no hit, `mIoU` on a frame with no class.
Those frames do not enter the mean, and the number of frames each mean covers
is recorded. Two conditions can cover different frames, and comparing them
without the counts once flipped the sign of a pilot result.

PQ is not stored. When a table wants it, it is SQ × F1_50 per frame, taken from
the per-frame values. Counts (frames, objects, regions) are kept as diagnostics
and never get a star.

### Primary metrics

`F1_50` and `SQ` in the geometric view. They were chosen before any score of
this evaluator was looked at. The pilot evaluator's scores of the same
quantities have been seen, which is why the choice is fixed before this
evaluator scores anything.

The set is kept as small as the claims allow. The evaluator computes every
metric in the table; which of them the paper reports is settled before any
score of this evaluator is seen.

### Consistency over time: not decided yet

`time_IoU` is kept only as a reference value and never gets a star. It uses no
GT, and the workbench measured three faults in it:

- it rises as regions get coarser, and one region covering the whole frame
  scores 1.0;
- an id that disappears costs nothing;
- for a condition segmented frame by frame, whose ids do not carry over from
  one frame to the next, it means nothing.

`temporal_f1`, built to close the second fault, keeps the first.

The candidates measure identity against the GT, and come from the workbench's
`track_metrics`: hold (whether the region picked on the frame tracking starts
from still covers the same GT thing seconds later), IDF1, ID switches,
fragmentation and re-entry. Choosing among them also means deciding what a GT track is: the
datasets carry no ids for individual things, and the workbench links each
class's connected components over time.

Until that is settled, every temporal measure is a reference value: none is
primary, and none but `time_IoU` is part of the evaluator. A measure that is to
carry a star has to join the evaluator before it is frozen.

## Rules that keep the numbers honest

### Thresholds

| name | value | why |
|---|---|---|
| `BOUNDARY_TOL_PX` | 2 | the pilot evaluator's value for its main boundary keys; what it allows is given with the boundary definition above |
| `MATCH_IOU` | 0.5, as IoU ≥ 0.5 after greedy pairing | the pilot evaluator's rule. PQ's convention, IoU > 0.5, makes a pairing unique; at exactly 0.5 the greedy order decides |

There is no minimum object size. The pilot evaluator dropped connected
components and predicted regions under 300 px. With whole-class objects, a
sliver joins its class's region instead of becoming an object, so a cut would
only decide whether a small class counts at all, and nothing in the data sets
that size. Pilot mode keeps the cut, as `PILOT_MIN_CC_PX`, and nothing else
uses it.

### Fail closed

- A colour or id missing from the dataset's class table raises; it is never
  mapped to background. ATLAS-120k's benchmark code sends unknown ids to
  background, so it is not reused as code.
- The CholecSeg8k table maps (0, 50, 128) to Hepatic Vein. The dataset's
  watershed codes settle it (see the measurements below).
- (255, 255, 255), the line CholecSeg8k draws between regions, and
  CholecSeg8k's Black Background are `ignored`.
- `Excluded frames` is checked on ATLAS-120k's original ids, before they are
  mapped to the 30 classes, which would turn it into background.
- Nothing is dropped without a count: skipped frames, ignored and invalid
  pixels, and the frames a metric is not defined on are counted in the JSON.

### Recorded with every score

- `eval_code_sha`, which covers the class tables as well as the code, so a
  changed type is a new evaluator
- the dataset, the class set (the 30 classes or the 47 original ids), the view,
  and whether the score was made in pilot mode
- the sha of every input read: GT masks, depth maps and predictions
- the Python, numpy, OpenCV and Pillow versions

The pilot evaluator's JSONs recorded neither the input shas nor the versions.

Two scores are comparable only when their `eval_code_sha`, dataset, class set,
view and mode match, they cover the same clips, and they read the same GT masks
and depth maps. `compare_eval` refuses any other pair, and reports a difference
in versions. The same `eval_code_sha` is not enough on its own: pilot mode and
the normal mode share it, and so do the views.

### Checked against the pilot evaluator

- Pilot mode runs this evaluator with the pilot evaluator's rules:
  - ATLAS-120k's 47 original ids;
  - the pilot evaluator's CholecSeg8k colour table, which maps unknown colours
    to background — the one place anything becomes background silently,
    allowed because pilot mode exists only for this check;
  - the pilot evaluator's four domains (`full`, `labeled`, `tissue`,
    `labeled_tissue`, with its instrument ids) in place of the views;
  - per-class 8-connected components of at least `PILOT_MIN_CC_PX` as GT
    objects, and regions of at least that size as predicted objects;
  - frames in file order for `time_IoU`.
- On the 38 conditions already scored, pilot mode must reproduce every key it
  shares with the pilot evaluator — the metrics table names them, and their
  per-domain variants — at zero tolerance: the values written must be equal.
  Ties are broken as the pilot evaluator breaks them. The check runs where the
  pilot evaluator and its scores are, and takes their paths as arguments.
- Every difference in the normal mode then comes from a rule this document
  changes, and is listed.
- A score made in pilot mode is marked as such and never enters a comparison
  with a normal one.

## Why the pilot evaluator was replaced

The pilot evaluator cannot be corrected in place: any changed byte moves its
sha. Three kinds of problem make correcting it worth a new evaluator.

### 1. The metrics overlap

The pilot evaluator writes 48 summary keys, which carry about nine independent
quantities. Four relations hold exactly, frame by frame; each was confirmed by
running the pilot evaluator's own functions on 366 synthetic frames:

| relation | why |
|---|---|
| PQ = SQ × F1@0.5 | F1@0.5 is PQ's recognition quality, RQ |
| 2·PQ − F1@0.5 ≤ F1_avg ≤ 2·PQ − 0.9·F1@0.5 | every F1@t has the same denominator, \|GT\| + \|pred\|, so averaging over t sums the matched IoUs in steps |
| Dice = 2·IoU / (1 + IoU), per class | both come from the same intersection and union |
| boundary_F = harmonic mean of boundary_P and boundary_R | its definition |

The others overlap in meaning rather than by formula. `overseg_mean` and
`VI_split` both measure splitting, and `underseg_error` and `VI_merge` both
measure merging. The boundary tolerances 1, 2, 3 and 5 are points on one
monotone curve. Each instance metric is also repeated over four domains.

Many overlapping keys count one improvement several times. They also make it
likely that some key reaches a star by chance.

### 2. Class handling is ad hoc

- **Classes that cover several things.** The pilot evaluator's GT objects are
  per-class connected components, and that breaks on classes that lump
  distinct things together:
  - ATLAS-120k has one class, `Tools/camera`, for every instrument, so
    touching instruments are one GT object.
  - CholecSeg8k's authors describe *Gastrointestinal Tract* (stomach, small
    intestine, nearby tissue) and *Liver Ligament* (several ligaments and the
    lesser omentum) as broad classes.
- **No class types.** Which classes are tools lives in the pilot evaluator's
  `eval_track.py` (`INSTRUMENT_CLASSES`), and reaches only the instance metrics
  of its `tissue` domains. Classes that geometry cannot recover — blood, for one
  — can be dropped only through `--extra_ignore`. That reaches only the instance
  and VI metrics, and even there it removes the GT objects but not the
  predicted pixels on them. `eval_gt_clips.py` never passes it, so no score it
  wrote used it. ATLAS-120k classes have no type at all beyond tool.
- **The datasets' own protocols are not followed.** ATLAS-120k's benchmark code,
  ATLAS-bench, maps every mask to 30 classes before scoring. The dataset's label
  table carries the same mapping as its `train_id` column. It merges similar
  classes (Omentum and Mesenterium into Fat, Aorta into Artery, Vena cava,
  Hepatic vein and V azygos into Vein), and the label table says the classes
  mapped to 0 "are either excluded from evaluation or merged into the
  background"; the benchmark merges them into background. The pilot evaluator
  scores the 47 original ids. Only id 0 counts as background, so `Excluded
  frames` (42), a marker rather than anatomy, would be scored as foreground.
- **A CholecSeg8k class is never read.** The pilot evaluator's colour table
  gives Hepatic Vein as (0, 255, 0), a colour no mask in the dataset contains.
  Its pixels are (0, 50, 128), which the pilot evaluator reads as background.

### 3. Silent drops

- CholecSeg8k colours missing from the colour table become background without
  a word, and background is excluded from nearly every metric.
- Connected components and predicted regions under 300 px are dropped from the
  instance metrics: neither missed nor false, and not counted. The count is
  taken at the evaluation resolution, so what 300 px means depends on it.
- A comment in the pilot evaluator says a fragment counts towards `overseg` if
  it covers "5 % or 200 px" of a class; the code requires both.

### What the pilot evaluator's handling changes, measured on the GT

The masks were resized with nearest neighbour to the depth maps' shape, as the
pilot evaluator does. Most frames are 504 px wide and 252 to 504 px high; four
portrait ATLAS-120k clips (73 GT frames) are 476 px wide and 504 px high.

**ATLAS-120k: 315 clips, 7,698 GT frames**

| | |
|---|---|
| `Excluded frames` (42) | in no frame |
| classes the 30-class protocol drops (Kidney, Ureter, Mesocolon, Adrenal gland, Pancreas, Duodenum) | in 835 frames (10.8 %), 2.25 % of pixels; the pilot evaluator scores them as foreground |
| ids outside the 47-class table | none |
| the 30 classes instead of the 47 original ids | changes the GT object count of 910 frames (11.8 %); objects −2.3 %, foreground pixels −2.9 % |
| `Tools/camera` connected components ≥ 300 px, per frame with tools | 1 in 2,256 frames, 2 in 3,869, 3 or more in 1,395, none in 15. How many are touching instruments merged into one cannot be told from the GT. |
| 300 px cut | drops 14.2 % of GT components, 0.05 % of foreground pixels. A class vanishes from a frame's instance evaluation in 1.2 % of (frame, class) pairs; the worst is Catheter, 19 of 64. |

**CholecSeg8k: 9 clips, 270 GT frames**

These are the nine clips on the machine the measurements were made on, a part
of the CholecSeg8k population the paper scores. The rows show the kind and
rough size of each effect, not the population's numbers.

| | |
|---|---|
| Hepatic Vein | in 20 frames (7.4 %) of 3 clips, all read as background: 11 GT objects of 300 px or more never scored. |
| colours outside the table | 0.053 % of pixels, all silently background: (0, 50, 128), which is Hepatic Vein, on 28,054 px, and (255, 255, 255) on 3,978 px |
| 300 px cut | drops 48.3 % of GT components, but 2,320 of the 2,832 dropped are under 10 px (slivers from resizing and annotation), so only 0.08 % of foreground pixels. A class vanishes from a frame in 2.4 % of (frame, class) pairs; the worst is Gastrointestinal Tract, 30 of 170. |
| Black Background | the pipeline crops each clip to the rectangle around the endoscope's circle, which keeps its corners: 2.2 % of pixels, 7.8 % in one clip. They have valid depth, so the pilot evaluator scores them as background. |

**Which colour is which CholecSeg8k class.** All 8,080 frames of the original
dataset were checked against the dataset's watershed masks, which carry a class
code per pixel:

- Each of the 12 classes other than Hepatic Vein has one colour and one code.
- The one remaining code, 33, belongs only to (0, 50, 128), on 314,317 px in
  317 frames.
- (0, 255, 0) occurs in no frame.
- (255, 255, 255) carries code 255: it is the line drawn between regions, not a
  class.
- Every pixel of the 8,080 colour masks is one of these 14 colours.

**How CholecSeg8k's labels were made.** The annotators drew strokes of each
class, and the watershed masks fill the rest of the frame from them. Over the
8,080 frames, 24.3 % of the pixels were drawn; the rest took the class of the
stroke the fill reached. So a region's boundary is the watershed's, not the
annotator's, and a stroke placed in the wrong region labels that whole region:
in video12 frame 20024, a Gallbladder stroke labels a region the class review
read as abdominal wall.

What this says:

- The 300 px cut removes slivers from connected components. Under this
  evaluator's objects a sliver joins its class's region, so the cut is not
  needed.
- `Excluded frames` does not occur, but this evaluator handles it explicitly
  anyway.
- Hepatic Vein is a plain error in the pilot evaluator's ground truth. It is
  small in the pixels it touches, but no CholecSeg8k pilot score includes the
  class.
- The 47 original ids against the 30 classes is a choice, not an error, and it
  moves ATLAS-120k's ground truth by 2–3 %.
- CholecSeg8k's black corners are outside the view, not background.
- CholecSeg8k's boundaries are the watershed's, which is worth knowing when its
  boundary metrics are read.
- The other substantive issues are the class types and the classes that cover
  several things.

## Class tables

Each table is a data file, one per dataset: every class's type, and the
colours (CholecSeg8k) or the mapping to the 30 classes (ATLAS-120k). The files
are hashed into `eval_code_sha` with the code.

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
| — | region line | (255, 255, 255) | ignored: the line drawn between regions |

Cystic Duct is `expert` although ATLAS-120k's Bile/lymph duct is `tissue`:
CholecSeg8k does not merge the ducts, so the class still ends where the
anatomical stretch ends. Hepatic Vein is `expert` because telling it from other
vessels takes anatomical knowledge. Neither moves much: Cystic Duct occurs in 3
of the 17 videos, Hepatic Vein in 1.

### ATLAS-120k: the 30 classes

Scores and claims use the 30 classes that ATLAS-120k's own model and benchmark
code score. The benchmark maps every mask before scoring, through
`datasets/class_mapping.py` in the ATLAS-bench repository, read at commit
`e286a584`:

- Seven ids become background: Kidney, Ureter, Excluded frames, Mesocolon,
  Adrenal gland, Pancreas and Duodenum.
- Similar classes merge:
  - Aorta into Artery.
  - Vena cava, Hepatic vein and V azygos into Vein.
  - Cystic duct, Ductus choledochus, Ductus hepaticus and Thoracic duct into
    Bile/lymph duct.
  - Omentum and Mesenterium into Fat.
  - Nerves into Nerve.
  - Catheter and Non anatomical structures into Non anatomical.

The class set is then the benchmark's. Resolution, crop and metrics are this
evaluator's own, so the numbers are not the benchmark's.

The 47 original ids are scored as well, as reference values only. They are
written to a separate block of the JSON, and they never get a star. There, each
original id takes the type of the class it merges into. The seven ids the
30-class mapping turns into background take their own verdicts from the review
below, with "unsure" counted as `expert`: Kidney, Mesocolon and Adrenal gland
are tissue; Ureter and Pancreas are expert; Duodenum is appearance; and
Excluded frames is excluded.

The review supports the 30 classes. Its main difficulties were stretches of
one tube (cystic duct against ductus choledochus) and one kind of vessel (vena
cava against V azygos) told apart by anatomical convention. The merge removes
exactly those boundaries.

### ATLAS-120k: types

ATLAS-120k defines no types: its benchmark scores `Tools/camera` like any other
class. Every one of the 47 classes was looked at in the masks, and the verdicts
are below, per class of the 30. The original ids merged into each are listed,
with the verdict and note for each. Hepatic vein (15), Thoracic duct (38) and
Nerves (39) were found in no mask the review searched, so they have no bearing.

| # | class | merged from: verdict, note | type |
|---:|---|---|---|
| 1 | Tools/camera | tool | tool |
| 2 | Vein | Vein (major): unsure, "hard even from colour"; Vena cava: appearance, "colour works, but it takes expertise"; V azygos: appearance, "colour might do" | expert |
| 3 | Artery | Artery (major): unsure, "uses shape, but needs a lot of expertise"; Aorta: tissue, "only faintly shape" | expert |
| 4 | Nerve | Nerve (major): appearance, "very faint" | appearance |
| 5 | Small intestine | tissue, "plainly shape" | tissue |
| 6 | Colon/rectum | tissue, "fairly plainly shape" | tissue |
| 7 | Abdominal wall | unsure, "almost background" | backdrop |
| 8 | Diaphragm | unsure, "almost background" | backdrop |
| 9 | Fat | Omentum: tissue, "a coherent shape"; Mesenterium: tissue, "barely shape" | tissue |
| 10 | Liver | tissue | tissue |
| 11 | Bile/lymph duct | Cystic duct and Ductus choledochus: unsure, "mostly shape, but where the stretch ends takes expertise"; Ductus hepaticus: appearance | tissue: the merge removes the stretch boundaries |
| 12 | Gallbladder | tissue | tissue |
| 13 | Hepatic ligament | tissue | tissue |
| 14 | Cystic plate | appearance | appearance |
| 15 | Stomach | tissue | tissue |
| 16–21 | Spleen, Uterus, Ovary, Oviduct, Prostate, Urethra | tissue | tissue |
| 22 | Ligated plexus | appearance | appearance |
| 23 | Seminal vesicles | tissue, "shape, probably" | tissue |
| 24 | Non anatomical | Catheter: unsure, "shape as a rule, but invisible when buried in tissue"; Non anatomical structures: tool, "shape, as a rule" | tool |
| 25 | Bladder | tissue, "sometimes by shape, sometimes not" | tissue |
| 26 | Lung | tissue | tissue |
| 27 | Airway (bronchus/trachea) | unsure, "shape cannot make this division; specialist" | expert |
| 28 | Esophagus | tissue | tissue |
| 29 | Pericardium | unsure, "specialist" | expert |
| 0 | Background | also Kidney: tissue; Ureter: unsure; Excluded frames: excluded; Mesocolon and Adrenal gland: tissue; Pancreas: unsure, "neither shape nor colour settles it"; Duodenum: appearance, the same note | background |

Vein and Artery are `expert`: their members' verdicts split between
appearance, tissue and unsure, and what they share is that telling them apart
takes anatomy.
