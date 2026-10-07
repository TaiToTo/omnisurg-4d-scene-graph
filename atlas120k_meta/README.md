# ATLAS-120k metadata

What the measurements need to know about the ATLAS-120k videos, without the
videos: which clips are measured, how each is cropped, and a fingerprint of
each clip's depth. No video, frame or mask is
here. Which clips a measurement reads is decided by `clips.txt`, never by a
count in a directory name.

## The population

`clips.txt` lists the **315 clips from 91 videos** the paper measures. The
rule, fixed before the extraction ran: from the 97 videos with an mp4 and a
clip index, take one frame every 0.52 s and keep the longest run of
consecutive annotated frames if it has at least 8. Of the 494 clips in the
index, 167 were too short after thinning (six videos lost every clip), 10
have no mask, and 2 are duplicates of another clip. Nothing is excluded on a
score. Crop rectangles select nothing, and a clip with a cut in its
recording is used whole.

`videos/videos.json` is the inventory of the release's tree, made by an
earlier extraction of it (96 videos, 438 clips; stride 3 annotated frames, at
least 15 of them, a clip with a mask gap dropped whole). The overlap check
below reads it.

Some clips of the release overlap:

- **What the inventory records.** For each clip, `videos/videos.json` gives
  `native_range`, its first and last frame as the clip index numbers them,
  and `n_native`, how many frames it holds.
- **A range is not a run.** Seven clips, all in the two videos below, skip
  frames inside their range, so two clips can share fewer frames than their
  ranges have in common.
- **`cholecystectomy/_-aytJndMV4`.** Six pairs of adjacent clips share 441
  frames. The population holds `clip_0005` and `clip_0010`; `clip_0006` and
  `clip_0011` thin to the same frames as these and are the two duplicates.
- **`hemicolectomy/5YDMlxTl0k8`.** The range of `clip_0016` spans
  `clip_0017` and `clip_0018`, but the clip holds every frame of `clip_0018`
  and none of `clip_0017`. None of the three has a mask.
- **The population.** No two of its clips share a frame: each is a run of
  frames inside one clip of the release, and no two come from clips whose
  ranges intersect.

`tests/test_atlas120k_meta.py` pins the clips that skip frames and the pairs
whose ranges intersect, and checks the population against them.

## Files

| file | what it is | from the workbench |
|---|---|---|
| `clips.txt` | the population, one `<procedure>__<video>__gt_<n>` per line | `ipcai2027_experiment/frozen/atlas97_clips.txt` |
| `depth_manifest.json` | per clip, sha256 of depth and intrinsics, frame count, shape | `ipcai2027_experiment/frozen/atlas97_depth_manifest.json` |
| `crop_rects.json` | per clip, the rectangle to use (`rect`, in `src_size` pixels) and whether the automatic one was accepted (`verdict`); `rect_used` and `via` are history | `experiment/crop_necessity/verdicts/verdicts_latest.json` |
| `frame_ratio.json` | per video, the measured ratio between the clip index's frame numbers and the mp4's (1, 2, 3 or 4), with the frame pair it was matched on and their pixel difference; all 97 videos | `outputs/crop_cuts/_align2.json` |
| `videos/videos.json` | the 100-video tree: per video its mp4, clips, frame counts, why any was dropped | `ipcai2027_experiment/task22_atlas100/out/manifest/videos.json` |

Clip names: `<procedure>__<video>__gt_<n>` in the population files and the
depth manifest, `<procedure>/<video>/clip_<n>` as `key` elsewhere.

The files are byte copies of the workbench's, except that an absolute path
(`atlas_root` in `videos.json`) was removed and free-text `note` fields were
translated from Japanese. Nothing reads the notes. The workbench's other
files about this tree (the cut marks, the audits of the earlier extraction,
its clip lists) are not carried for now: the paper's numbers read none of
them. `docs/porting.md` lists them under "Not carried for now".

`frame_ratio.json` is the exception: it was assembled from the measurement's
output, which was git-ignored in the workbench and is no longer on disk. The
workbench's `surgical_core/atlas/frame_ratio.py` carries all 97 video names
and, for the 14 ratios above 1, the ratio and its `diff` as constants; the
frame pair each video was matched on, and the `diff` of the 83 at ratio 1,
exist only here. The ratios follow the dataset's own sampling rule,
`max(1, int(fps / 15))` (see the reader's docstring). The reader's
`verify_against_bundled` checks a ratio against the pixels but does not find
one: re-measuring a video means calling it once per candidate ratio and
keeping the one that matches.

## Who reads these

`surgical_core/atlas120k/clip_rects.py` reads the crop rectangles and
`frame_ratio.py` the frame ratios; both take the file's path as an argument.
`surgical_core/clip_time.py` is handed the frame ratios when a manifest does
not record its own.
`tests/test_atlas120k_meta.py` pins what must hold between the files.
