# The pilot evaluator, and why it was replaced

The pilot measurements were scored by an earlier evaluator, the *pilot
evaluator*. `docs/evaluation.md` specifies the evaluator that replaced it,
and its "Checked against the pilot evaluator" says how the two are compared.
This page says what was wrong with the pilot evaluator, and what its
handling changes on the ground truth.

The pilot evaluator cannot be corrected in place: any changed byte moves its
`eval_code_sha`. Three kinds of problem make correcting it worth a new
evaluator.

## The metrics overlap

The pilot evaluator writes about 85 summary keys per clip, counting each
metric once per domain and per tolerance, and they carry about nine
independent quantities. Four relations hold exactly, frame by frame; each was
confirmed by running the pilot evaluator's own functions on 366 synthetic
frames:

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

## Class handling is ad hoc

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
- **The datasets' own protocols are not offered.** ATLAS-120k's benchmark code,
  ATLAS-bench, maps every mask to 30 classes before scoring. The dataset's label
  table carries the same mapping as its `train_id` column. It merges similar
  classes (Omentum and Mesenterium into Fat, Aorta into Artery, Vena cava,
  Hepatic vein and V azygos into Vein), and the label table says the classes
  mapped to 0 "are either excluded from evaluation or merged into the
  background"; the benchmark merges them into background. The pilot evaluator
  scores only the 47 original ids, so none of its numbers can be set beside
  the benchmark's classes. Only id 0 counts as background there, so `Excluded
  frames` (42), a marker rather than anatomy, would be scored as foreground.
- **A CholecSeg8k class is never read.** The pilot evaluator's colour table
  gives Hepatic Vein as (0, 255, 0), a colour no mask in the dataset contains.
  Its pixels are (0, 50, 128), which the pilot evaluator reads as background.

## Silent drops

- CholecSeg8k colours missing from the colour table become background without
  a word, and background is excluded from nearly every metric.
- Connected components and predicted regions under 300 px are dropped from the
  instance metrics: neither missed nor false, and not counted. The count is
  taken at the evaluation resolution, so what 300 px means depends on it.
- A comment in the pilot evaluator says a fragment counts towards `overseg` if
  it covers "5 % or 200 px" of a class; the code requires both.
- A frame is a GT frame when a mask file exists for it, so the viewer's SAM 3
  masks would be scored as GT without a word.

## What the pilot evaluator's handling changes, measured on the GT

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

**CholecSeg8k: 9 clips, 167 GT frames**

These are the nine clips on the machine the measurements were made on, a part
of the CholecSeg8k population the paper scores. The viewer's SAM 3 masks on
103 more frames are not GT and are left out (`docs/evaluation.md`, "Which
frames"). The rows show the kind and rough size of each effect, not the
population's numbers.

| | |
|---|---|
| Hepatic Vein | in 20 frames (12.0 %) of 3 clips, all read as background: 11 GT objects of 300 px or more never scored. |
| colours outside the table | 0.088 % of pixels, all silently background: (0, 50, 128), which is Hepatic Vein, on 28,054 px, and (255, 255, 255) on 3,978 px |
| 300 px cut | drops 17.9 % of GT components, but 180 of the 416 dropped are under 10 px (slivers from resizing and annotation), so only 0.07 % of foreground pixels. A class vanishes from a frame in 1.1 % of (frame, class) pairs; the worst is Gastrointestinal Tract, 4 of 98. |
| Black Background | the pipeline crops each clip to the rectangle around the endoscope's circle, which keeps its corners: 0.1 % of pixels, 0.2 % in one clip. They have valid depth, so the pilot evaluator scores them as background. |

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
