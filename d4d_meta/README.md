# D4D metadata

What the D4D measurements need to know about the dataset's clips, without the
data: which clips the census measures, and which sides the measurements score.
No frame, point cloud or pose is here. Which sides a measurement reads is
decided by `population.json`, never by a count in a directory name.

## The dataset

D4D, the Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical
Scenes (DOI 10.25532/OPARA-1033), pairs the video of a stereo endoscope with
the shape of the tissue, scanned by a structured-light camera. It is organised
by specimen, session and clip. Its upstream loader, `d4d.loader`, yields the
clips of every specimen whose directory name does not hold `ambiguous`: 271
clips of 98 sessions and 5 specimens. `census_clips.txt` lists them.

A **side** is one of a clip's two scans of the tissue: `start`, taken before
the clip's first frame, and `end`, taken after its last. The release gives a
side a point cloud and a camera pose. The pose places the point cloud in the
endoscope's view.

## The census

The census measures each side's inputs, and reads no score. For each side it
records:

- `present`: whether the side has a point cloud, a camera pose and the
  endoscope's calibration;
- `gt_blk_frac`: how much of the view the point cloud reaches. The point cloud
  is projected into the view through the side's pose, and the view is cut into
  blocks of 16 × 16 pixels. A block counts when more than half of its pixels
  hold no instrument, by the instrument mask of the frame nearest to the scan,
  dilated with a square of 9 × 9 pixels. `gt_blk_frac` is the share of
  counted blocks that at least 5 points fall in;
- `frame_minus_gt_s`: the time of the frame nearest to the scan, less the time
  of the scan, in seconds;
- `active`: whether the clip's first moment, for `start`, or its last, for
  `end`, falls inside an interval in which the tissue moves.

For each clip it records `key`, the clip's `<specimen>/<session>/<clip>`, and
`moved_camera`, whether the endoscope moves during the clip. `moved_camera`
and `active` come from two files made from the dataset beforehand: a table of
the clips, and a record of the tissue's motion over each clip, measured on the
endoscope's stereo depth. The census itself is not here.

`present`, `active` and `moved_camera` are true or false. `gt_blk_frac` and
`frame_minus_gt_s` are finite numbers. Of a side without a point cloud, the
rule reads only `present`.

## The population

A side is left out for the first of these reasons that applies, and kept
otherwise:

- `no_gt`: `present` is false;
- `gt_not_visible`: `gt_blk_frac` is below 0.20;
- `tissue_moving`: `active` is true;
- `gt_stale`: `frame_minus_gt_s` is more than 10 s from zero.

The rule reads no score, and `population.json` records its thresholds under
`thresholds`. Of the 542 sides of the 271 clips:

| reason | sides |
|---|---|
| kept | 487 |
| `tissue_moving` | 45 |
| `gt_stale` | 8 |
| `no_gt` | 2 |
| `gt_not_visible` | 0 |

The least `gt_blk_frac` of a side with a point cloud is 0.246, so the coverage
threshold leaves out no side. The 45 sides `tissue_moving` leaves out are all
`start` sides. The record of the tissue's motion stops sampling a clip 0.5 to
1.7 s before its last frame, and every interval ends at a sample, so no `end`
side is `active`. The time between a scan and its nearest frame is 4.3 s at the
median, and above 10 s for the 8 sides `gt_stale` leaves out.

The kept sides belong to 269 clips of all 98 sessions and 5 specimens. The two
clips with no kept side are `specimen_1/2025_03_06-17_39_51/Clip_5` and
`specimen_4/2025_06_05-15_45_18/Clip_3`: each has no `end` point cloud, and its
`start` side is `active`. The camera moves in 30 of the 271 clips. 29 of them
are kept, together with 240 clips whose camera is still.

## Files

| file | what it is | from the workbench |
|---|---|---|
| `census_clips.txt` | every clip the census measures, one `<specimen>/<session>/<clip>` per line, in the order the loader yields them | the keys of `ipcai2027_experiment/out/09/census.json` |
| `population.json` | the counts above, the thresholds, and under `sides` one entry per kept side: `key`, `side`, `moved_camera`, and `specimen`, the first name of `key` | `ipcai2027_experiment/out/09/population.json` |
| `clips.txt` | the clips with at least one kept side, one per line, sorted | `ipcai2027_experiment/out/09/pop_all.txt` |

`population.json` and `clips.txt` are byte copies of the workbench's, and
`pipeline/select_d4d_population.py` writes them again, byte for byte, from the
workbench's census. `census_clips.txt` holds the keys the upstream loader
yields, in its order, on the development machine's copy of D4D. The census
holds the same keys in the same order, and the step refuses one that does
not, since `population.json` lists the sides in the census's order.

## Who reads these

`pipeline/select_d4d_population.py` reads the census and `census_clips.txt`,
and writes `population.json` and `clips.txt`:

```bash
python -m pipeline.select_d4d_population --census /path/to/census.json \
    --clips d4d_meta/census_clips.txt --out d4d_meta
```

It refuses a census that does not hold each listed clip once and in order,
and a census in which a field the rule reads is missing or holds a value of
the wrong kind.

`tests/test_select_d4d_population.py` pins what must hold between the files.
