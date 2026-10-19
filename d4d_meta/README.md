# D4D metadata

Which D4D clips the census measures, which sides the measurements score, and
which clips the camera trajectory result measures. No frame, point cloud or
pose is here.

## The dataset

D4D (DOI 10.25532/OPARA-1033) pairs the video of a stereo endoscope with the
shape of the tissue, scanned by a structured-light camera. Its upstream
loader, `d4d.loader`, yields 271 clips of 98 sessions and 5 specimens. It
leaves out the specimens whose directory name holds `ambiguous`.

A **side** is one of a clip's two scans of the tissue: `start`, taken before
the clip's first frame, and `end`, taken after its last. A side has a point
cloud and the camera pose that places it in the endoscope's view.

## The census

The census measures each side's inputs, in these fields:

- `key`, of a clip: `<specimen>/<session>/<clip>`;
- `moved_camera`, of a clip: whether the endoscope moves during the clip's
  session, as the dataset's `info/session_details.txt` records it;
- `present`: whether the side has a point cloud, a camera pose and the
  endoscope's calibration;
- `gt_blk_frac`: the share of the view's 16 × 16-pixel blocks, among those
  mostly free of instruments, that at least 5 of the side's points fall in;
- `frame_minus_gt_s`: the time of the frame nearest to the scan, less the
  time of the scan, in seconds.

`present` and `moved_camera` are true or false. `gt_blk_frac` and
`frame_minus_gt_s` are finite numbers. `pipeline.d4d_census` makes the
census, which is not kept here.

## The population

A side is left out for the first of these reasons that applies, and kept
otherwise:

- `no_gt`: `present` is false;
- `gt_not_visible`: `gt_blk_frac` is below 0.20;
- `gt_stale`: `frame_minus_gt_s` is more than 10 s from zero.

The rule does not judge whether the tissue moved between a scan and its
clip. D4D takes each scan with the scene static and the endoscope's light
off (arXiv:2603.02985), so no frame shows the tissue at the scan. D4D's own
loader pairs each clip with both its scans as they are.

Neither the census nor the rule reads a score. Of the 542 sides of the 271
clips:

| reason | sides |
|---|---|
| kept | 532 |
| `gt_stale` | 8 |
| `no_gt` | 2 |
| `gt_not_visible` | 0 |

The least `gt_blk_frac` of a side with a point cloud is 0.246, so the
coverage threshold leaves out no side. The kept sides belong to all 271
clips, 98 sessions and 5 specimens.

## The trajectory clips

`trajectory_clips.json` lists each clip of `clips.txt`, one clip a line, cut
to 112 frames spread over the whole clip. The truth is the camera's pose
that the optical tracker logs. A frame inside a gap of the log longer than
0.25 s is not tracked, and is left out of the truth and of every estimate.
A clip is left out for its truth only, never for a score, by the first of
these reasons that holds:

| reason | what holds | clips |
|---|---|---|
| `no_log` | the session has no tracker log | 0 |
| `log_not_covering` | the log starts after the clip's first frame or ends before its last | 3 |
| `few_tracked` | the tracker sees fewer than 30 of its frames, or fewer than 60 % | 19 |
| `gt_jump` | between two tracked frames, the truth moves faster than 300 mm/s or turns faster than 300 degrees/s | 1 |
| `gt_static` | the tracked camera centres' RMS radius is below 1 mm | 239 |

The other 9 clips, of 4 specimens, are usable. Each clip holds:

- `name`, `key` and `specimen`;
- `usable`, true exactly when `drop`, the reason it is left out, is "";
- `span_mm`, the RMS radius of the tracked camera centres, and `stratum`:
  `moving` from 10 mm up, `slow` below; both null when no frame is tracked;
- `n_tracked`, the number of tracked frames, null when the log does not
  cover the clip;
- for a usable clip, `frames`, the frames' positions among the clip's
  rectified left frames; `times`, in s, as the frames' file names give
  them; and `tracked`, whether each frame is tracked. A clip left out
  records null for the three.

A score records the list's hash as `clip_list_sha`, and
`trajectory_eval.tools.scores` refuses to compare scores of two lists.

## Files

| file | what it is |
|---|---|
| `census_clips.txt` | every clip the census measures, one `key` per line, in the order the loader yields them |
| `population.json` | the counts above, the thresholds, and under `sides` one entry per kept side: `key`, `side`, `moved_camera`, and `specimen`, the first name of `key` |
| `clips.txt` | the clips with at least one kept side, one per line, sorted |
| `trajectory_clips.json` | the clips of `clips.txt` cut for the camera trajectory result, usable or not, with the fields above |

## What makes and reads these

`pipeline.d4d_census` makes the census from the dataset.
`pipeline.select_d4d_population` writes `population.json` and `clips.txt`
from the census and `census_clips.txt`. `pipeline.d4d_depth` reads
`population.json` for the sides it estimates depth on.
`trajectory_eval.tools.score_d4d select` writes `trajectory_clips.json` from
`clips.txt` and the tracker log, and its `prepare` and `score` read it.
`docs/pipeline.md` says how to run the steps.
`tests/test_select_d4d_population.py` pins what
must hold between the files.
