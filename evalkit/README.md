# evalkit

The code that turns the pipeline's output into the numbers the paper
reports. The full rules are in [`docs/evaluation.md`](../docs/evaluation.md);
this page is the short version, with a map of the parts.

## What it measures

The pipeline cuts every frame of a surgical video into regions and follows
them over time. It is not trained on surgery and it never names a region.
Against two public datasets with hand-labelled masks, the evaluator asks
three questions. Two of them are decided by keys named here; the other keys
help explain the answer.

- **Given one frame as an example, how far can the regions be followed?**
  With no training, for how many frames does a region picked out once keep
  covering the same thing? *Decided by:* no key, since no measure over time
  carries a star. `time_IoU` is reported for reference; it rewards coarse
  regions.
- **Given no example at all, what input makes the regions' boundaries fall
  where the labelled classes' boundaries are?** The image, the 3D shape from
  depth, or both; and for which classes and which procedures, since the
  answer differs between them. *Decided by:* `boundary_R_raw`, how many of
  the labelled boundaries the regions find, between conditions that differ
  in their input, within each of the three views below.
- **Given no ground truth, how much of the labelled structure is already in
  the regions?** Whether the things a surgeon would name are found, and how
  closely, before anyone names them. *Decided by:* `F1_50`, how many of the
  labelled things were found, with an extra region counted against it; and
  `SQ`, how closely the found ones match.

![The three questions on a drawn scene. Q1: a region picked out in one frame follows the gallbladder for two frames, then leaves it. Q2: the regions' boundaries against the GT's class boundaries, found and missed, giving boundary_R_raw. Q3: regions matched to the GT's objects, two hits and one extra region, giving F1_50 and SQ](../docs/figures/three_questions.png)

The three questions on a drawn scene. No key decides the first, so its
panel is an illustration; the numbers in the other two are what the modules
give on the scene.

Each question is put as a comparison: the pipeline in one configuration,
a *condition*, against another. A claim gets a star when the video-level
bootstrap 95 % CI of the difference on the deciding key does not straddle
zero: whole videos are resampled, so that the clips of one video are not
counted as independent evidence. The rule is written once, in
`paired_stats`.

Every key is computed on three sets of classes, called *views*: all classes;
tissue only; and the tissue that depth and shape can separate. The third
question is judged on the last view. The second is answered within each
view: the views sort the classes by what should separate them, and the
conditions vary what the pipeline is given, the image, the depth or both.

## The map

![The evaluator at a glance: a clip's inputs; one frame, scored in three views; one clip, each key's mean over its frames; one condition, one score file; two conditions, the key that decides each question](../docs/figures/evalkit_overview.png)

Six steps, from a clip's inputs to the three questions: steps 2 to 5 score
one condition, and step 6 compares two.

The same steps, part by part: what each part computes, and the module that
holds it. Arrows say what is computed from what; Q1 to Q3 mark the parts that
hold the key deciding each question above. What a module returns is in its
docstring.

![The evaluator, part by part: a clip's inputs; scored pixels and the keys of one frame in one view; one frame in every view; one clip; one score file per condition; the tools that compare two conditions](../docs/figures/evaluator_map.png)
