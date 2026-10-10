# D4D metadata

Which D4D clips the census measures, and which sides the measurements score.
No frame, point cloud or pose is here.

## The dataset

D4D (DOI 10.25532/OPARA-1033) pairs the video of a stereo endoscope with the
shape of the tissue, scanned by a structured-light camera. Its upstream
loader, `d4d.loader`, yields 271 clips of 98 sessions and 5 specimens. It
leaves out the specimens whose directory name holds `ambiguous`.

A **side** is one of a clip's two scans of the tissue: `start`, taken before
the clip's first frame, and `end`, taken after its last. A side has a point
cloud and the camera pose that places it in the endoscope's view.

## The census

The census measures each side's inputs. The rule reads these of its fields:

- `key`, of a clip: `<specimen>/<session>/<clip>`;
- `moved_camera`, of a clip: whether the endoscope moves during the clip;
- `present`: whether the side has a point cloud, a camera pose and the
  endoscope's calibration;
- `gt_blk_frac`: the share of the view's 16 × 16-pixel blocks, among those
  mostly free of instruments, that at least 5 of the side's points fall in;
- `frame_minus_gt_s`: the time of the frame nearest to the scan, less the
  time of the scan, in seconds;
- `active`: whether the tissue moves at the clip's first moment, for
  `start`, or at its last, for `end`.

`present`, `active` and `moved_camera` are true or false. `gt_blk_frac` and
`frame_minus_gt_s` are finite numbers. The census itself is not here.

## The population

A side is left out for the first of these reasons that applies, and kept
otherwise:

- `no_gt`: `present` is false;
- `gt_not_visible`: `gt_blk_frac` is below 0.20;
- `tissue_moving`: `active` is true;
- `gt_stale`: `frame_minus_gt_s` is more than 10 s from zero.

Neither the census nor the rule reads a score. Of the 542 sides of the 271
clips:

| reason | sides |
|---|---|
| kept | 487 |
| `tissue_moving` | 45 |
| `gt_stale` | 8 |
| `no_gt` | 2 |
| `gt_not_visible` | 0 |

The least `gt_blk_frac` of a side with a point cloud is 0.246, so the
coverage threshold leaves out no side. The kept sides belong to 269 clips of
all 98 sessions and 5 specimens.

## Files

| file | what it is | from the workbench |
|---|---|---|
| `census_clips.txt` | every clip the census measures, one `key` per line, in the order the loader yields them | the keys of `ipcai2027_experiment/out/09/census.json` |
| `population.json` | the counts above, the thresholds, and under `sides` one entry per kept side: `key`, `side`, `moved_camera`, and `specimen`, the first name of `key` | `ipcai2027_experiment/out/09/population.json` |
| `clips.txt` | the clips with at least one kept side, one per line, sorted | `ipcai2027_experiment/out/09/pop_all.txt` |

## Who reads these

`pipeline/select_d4d_population.py` writes `population.json` and `clips.txt`
from the census and `census_clips.txt`; `docs/pipeline.md` says how to run it
and what it refuses. `tests/test_select_d4d_population.py` pins what must
hold between the files.
