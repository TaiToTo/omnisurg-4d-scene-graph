# Evaluation v3 — specification

**Status: agreed, not yet implemented.** Nothing here is frozen yet. v3 is the
ruler the paper's numbers will be measured with. It replaces v2 — the evaluator
every score so far was measured with, `eval_code_sha = 1f8a813a…` — which stays
in the repository only as a reference that v3 is checked against.

Every decision the design needed is settled; they are listed at the end.

## Why a new ruler

v2 cannot be corrected in place: any changed byte moves its sha. Three kinds
of problem make correcting it worth a new ruler.

### 1. The metrics overlap

v2 writes 48 summary keys, which carry about nine independent quantities. Four
relations are exact; each was confirmed by running v2's own functions on 366
synthetic frames:

| relation | why |
|---|---|
| PQ = SQ × F1@0.5, per frame | F1@0.5 is PQ's recognition quality, RQ |
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

- **Umbrella classes.** Instances are defined as per-class connected
  components, and that definition breaks on classes that lump distinct objects
  together:
  - ATLAS-120k has one class, `Tools/camera`, for every instrument, so
    touching instruments are one GT instance.
  - CholecSeg8k's authors describe *Gastrointestinal Tract* (stomach, small
    intestine, nearby tissue) and *Liver Ligament* (several ligaments and the
    lesser omentum) as broad classes.
- **No class types.** Which classes are tools lives in `eval_track.py`
  (`INSTRUMENT_CLASSES`), and reaches only the instance metrics of the
  `tissue` domains. Classes that geometry cannot recover — blood, for one — can
  be dropped only through `--extra_ignore`. That reaches only the instance and
  VI metrics, and `eval_gt_clips.py` never passes it, so every batch
  measurement ran without it. ATLAS-120k classes have no type at all beyond
  tool.
- **The datasets' own protocols are not followed.** ATLAS-120k's evaluation
  code first maps every mask to 30 classes (the `train_id` column in
  `surgical_core/atlas/labels.py`). It merges similar classes (Omentum and
  Mesenterium into Fat, Aorta into Artery, Vena cava, Hepatic vein and V azygos
  into Vein), and its classes list says the classes mapped to 0 "are either
  excluded from evaluation or merged into the background". v2 scores the 47 raw
  ids. Only id 0 counts as background, so `Excluded frames` (42), a marker
  rather than anatomy, would be scored as foreground.
- **A CholecSeg8k class is never read.** The palette table gives Hepatic Vein
  as (0, 255, 0), a colour no mask in the dataset contains. Its pixels are
  (0, 50, 128), which v2 reads as background (see below).

### 3. Silent drops

- CholecSeg8k colours missing from the palette table become background without
  a word, and background is excluded from nearly every metric.
- Connected components and predicted regions under 300 px are dropped: neither
  missed nor false. The count is taken at the evaluation resolution, so what
  300 px means depends on it.
- A comment in `eval_track.py` says a fragment counts towards `overseg` if it
  covers "5 % or 200 px" of a class; the code requires both.

## What v2's handling touches, measured on the GT

The masks were resized with nearest neighbour to the evaluation resolution, as
v2 does: 504 px wide, with the height rounded to a multiple of 14.

**ATLAS-120k: 315 clips, 7,698 GT frames**

| | |
|---|---|
| `Excluded frames` (42) | in no frame |
| classes the 30-class protocol drops (Kidney, Ureter, Mesocolon, Adrenal gland, Pancreas, Duodenum) | in 835 frames (10.8 %), 2.25 % of pixels; v2 scores them as foreground |
| ids outside the 47-class table | none |
| the 30-class protocol instead of raw ids | changes the GT instance count of 910 frames (11.8 %); instances −2.3 %, foreground pixels −2.9 % |
| `Tools/camera` connected components ≥ 300 px, per frame with tools | 1 in 2,255 frames, 2 in 3,870, 3 or more in 1,395. How many are touching instruments merged into one cannot be told from the GT. |
| 300 px cut | drops 14.2 % of GT components, 0.05 % of foreground pixels. A class vanishes from a frame's instance evaluation in 1.2 % of (frame, class) pairs; the worst is Catheter, 19 of 64. |

**CholecSeg8k: 9 clips, 270 GT frames**

| | |
|---|---|
| Hepatic Vein | in 20 frames (7.4 %) of 3 clips, all read as background: 11 GT instances of 300 px or more never scored. |
| colours outside the table | 0.053 % of pixels, all silently background: (0, 50, 128), which is Hepatic Vein, on 28,054 px, and (255, 255, 255) on 3,978 px |
| 300 px cut | drops 48.3 % of GT components, but 2,320 of the 2,832 dropped are under 10 px (slivers from resizing and annotation), so only 0.08 % of foreground pixels. A class vanishes from a frame in 2.4 % of (frame, class) pairs; the worst is Gastrointestinal Tract, 30 of 170. |

**Which colour is which class.** All 8,080 frames of the original dataset were
checked against the dataset's watershed masks, which carry a class code per
pixel:

- Each of the 12 classes other than Hepatic Vein has one colour and one code.
- The one remaining code, 33, belongs only to (0, 50, 128), on 314,317 px in
  317 frames.
- (0, 255, 0) occurs in no frame.
- (255, 255, 255) carries code 255: it is the line drawn between regions, not a
  class.

What this says:

- The 300 px cut is mostly denoising, and v3 keeps it, named and justified.
- `Excluded frames` does not occur, but v3 handles it explicitly anyway.
- Hepatic Vein is a plain error in v2's ground truth. It is small in the
  pixels it touches, but no CholecSeg8k score from v2 includes the class.
- The raw ids against the 30-class protocol is a choice, not an error, and it
  moves ATLAS-120k's ground truth by 2–3 %.
- The other substantive issues are the class types and the umbrella classes.

## v3

### Class types: one table per dataset, read by every metric

Every class gets exactly one type:

| type | meaning | scored |
|---|---|---|
| `background` | not annotated as anything | never |
| `excluded` | a marker that takes the frame out of evaluation | the frame is skipped |
| `tool` | an instrument | in `all` only |
| `tissue` | anatomy that the scene's geometry can separate | in every view |
| `appearance` | tissue told apart only by colour or texture (blood, for one), which depth and shape cannot recover | in `all` and `tissue` views |
| `expert` | tissue whose boundary is set by anatomical convention — which vessel it is, where one stretch of a tube ends — so that neither shape nor colour shows it without anatomical knowledge | in `all` and `tissue` views |
| `backdrop` | a surface the scene sits against, close to background (abdominal wall, diaphragm) | in `all` and `tissue` views |

`expert` and `backdrop` came out of looking at every ATLAS-120k class: many
classes were neither shape nor colour, but one of these two.

There are three views, and every metric is computed over each of them the same
way:

- `all`: every class that is not `background` or `excluded`
- `tissue`: without `tool`
- `geometric`: `tissue` classes only

The table is data (one file per dataset), and its sha is recorded with every
score.

**CholecSeg8k types** (decided):

| id | class | type |
|---:|---|---|
| 0 | Black Background | background |
| 1 | Abdominal Wall | backdrop, as in ATLAS-120k |
| 2 | Liver | tissue |
| 3 | Gastrointestinal Tract | tissue |
| 4 | Fat | tissue |
| 5 | Grasper | tool |
| 6 | Connective Tissue | appearance |
| 7 | Blood | appearance |
| 8 | Cystic Duct | tissue |
| 9 | L-hook Electrocautery | tool |
| 10 | Gallbladder | tissue |
| 11 | Hepatic Vein | tissue |
| 12 | Liver Ligament | tissue |

**ATLAS-120k protocol** (decided). Scores and claims use the 30 classes that
ATLAS-120k's own model and benchmark code score. The benchmark maps every mask
before scoring, through `datasets/class_mapping.py` in the ATLAS-bench
repository:

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

Numbers then read the same way as the dataset's own benchmark.

The 47 raw ids are scored as well, as reference values only. They are written
to a separate block of the JSON, and they never get a star. There, each raw id
takes the type of the class it merges into. The seven ids the protocol turns
into background take their own verdicts from the review below, with "unsure"
counted as `expert`: Kidney, Mesocolon and Adrenal gland are tissue; Ureter and
Pancreas are expert; Duodenum is appearance; and Excluded frames is excluded.

The review supports the 30 classes. Its main difficulties were stretches of
one tube (cystic duct against ductus choledochus) and one kind of vessel (vena
cava against V azygos) told apart by anatomical convention. The merge removes
exactly those boundaries.

**ATLAS-120k types.** ATLAS-120k defines no types: its benchmark scores
`Tools/camera` like any other class. Every one of the 47 classes was looked at
in the masks, and the verdicts are below, per class of the 30-class protocol.
The raw ids merged into each are listed, with the verdict and note for each.
Hepatic vein (15), Thoracic duct (38) and Nerves (39) occur in no mask of the
602 clips, so they have no bearing.

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

### Objects (decided)

An object is one class's whole region in one frame, as ATLAS-120k's own AP
counts it. v2's per-class connected components are not used. That removes the
umbrella problem: two touching instruments are one `Tools/camera` object
because the dataset annotates them as one class, not because they touch. The
cost is that a class seen in pieces — a liver cut in two by an instrument — is
still one object. `F1_50`, `SQ` and `inst_BF` are taken over these objects.

### Metrics

| axis | key | replaces in v2 |
|---|---|---|
| instances found | `F1_50` | `inst_F1_50` (and PQ's RQ) |
| matched mask quality | `SQ` | `PQ`, `inst_F1_avg`, `inst_F1_75` |
| matched contour quality | `inst_BF` | — |
| class-map area | `mIoU` | `GT_mIoU`, `GT_mDice` |
| class-map contour | `boundary_F` (2 px) | boundary P and R, tolerances 1 / 3 / 5 |
| boundary recovered before class assignment | `boundary_R_raw` (2 px) | its precision and tolerances |
| splitting | `VI_split` | `overseg_mean` |
| merging | `VI_merge` | `underseg_error` |
| temporal consistency | `time_IoU` | — |

PQ is not stored. When a table wants it, it is SQ × F1_50 per frame, computed
where the table is made. Counts (frames, instances, regions) are kept as
diagnostics and never get a star.

The set is kept as small as the claims allow: a metric stays only if a claim
in the paper uses it. PQ, F1_avg, the Dice and the extra tolerances go.

**Primary metrics** (decided, before any v3 score is looked at): `F1_50`, `SQ`
and `time_IoU` in the `geometric` view. The rest of the table above is cut to
the ones a claim needs.

### Thresholds, named

| name | value | why |
|---|---|---|
| `MIN_OBJECT_PX` | 300 at 504 px wide, i.e. 0.17 % of the frame | a class region smaller than this is not an object: it removes resize and annotation slivers (see the measurements), and scales with resolution |
| `BOUNDARY_TOL_PX` | 2 at 504 px wide | v2's value |
| `MATCH_IOU` | 0.5 | makes matching unique; PQ's convention |

### Fail closed

- A colour or id missing from the dataset's table raises; it is never mapped
  to background.
- The CholecSeg8k table maps (0, 50, 128) to Hepatic Vein. The watershed codes
  settle it (see the measurements).
- (255, 255, 255) is the line between regions. It is ignored: scored as
  neither a class nor background, like an invalid depth pixel (decided).

### Recorded with every score

- `eval_code_sha`
- the class table's sha and the view
- the Python, numpy, OpenCV and Pillow versions

v2's JSONs recorded none of the last group.

### Checked against v2

- Run v3 in a v2-equivalent configuration: raw classes, v2's colour table, no
  views, and v2's connected components as objects.
- On the 38 conditions already scored, it must reproduce every key it shares
  with v2 at zero tolerance.
- Every difference in the real configuration then comes from a rule this
  document changes, and is listed.

## Decisions

All decided:

- CholecSeg8k class types: Fat is tissue, Abdominal Wall is backdrop.
- ATLAS-120k: the 30-class protocol for scores and claims; the 47 raw ids as
  reference values, in a separate block, never starred.
- ATLAS-120k class types: every class looked at (table above). Vein and Artery
  are expert.
- `backdrop` is scored in the `all` and `tissue` views, not in `geometric`.
- An object is a class's whole region in a frame.
- The metric set is kept minimal. The primary metrics are `F1_50`, `SQ` and
  `time_IoU` in the `geometric` view.
- CholecSeg8k region lines are ignored.
