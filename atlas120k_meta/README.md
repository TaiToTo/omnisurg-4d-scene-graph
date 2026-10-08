# ATLAS-120k metadata

What the measurements need to know about the ATLAS-120k videos, without the
videos: which clips are measured, how each is cropped, where a recording
cuts, and a fingerprint of each clip's depth. No video, frame or mask is
here. Which clips a measurement reads is decided by `clips.txt`, never by a
count in a directory name.

## The population

`clips.txt` lists the **315 clips from 91 videos** the paper measures. The
rule, fixed before the extraction ran: from the 97 videos with an mp4 and a
clip index, take one frame every 0.52 s and keep the longest run of
consecutive annotated frames if it has at least 8. Of the 494 clips in the
index, 167 were too short after thinning (six videos lost every clip), 10
have no mask, and 2 are duplicates of another clip. Nothing is excluded on a
score. Crop rectangles and cut marks select nothing: a clip with a cut is
used whole.

`videos/` is an earlier extraction of the same tree (96 videos, 438 clips;
stride 3 annotated frames, at least 15 of them, a clip with a mask gap dropped
whole). It is kept as the inventory of the raw tree and because `audit/` was
run on its output. Five names of the 315 are not among the 438, and in
`hemicolectomy/5YDMlxTl0k8` its clip numbers from `clip_0049` on are one
below the clip's own, because that extraction numbered outputs by position.

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
| `cut_marks.jsonl` | judged scene changes, append-only (last line per `key` wins); frame numbers are the mp4's. Ten cuts in two videos; `Bj13QcLRCVc#330` is still `dk` | `experiment/crop_necessity/marks/marks_20260913_174731.jsonl` |
| `videos/videos.json` | the 100-video tree: per video its mp4, clips, frame counts, why any was dropped | `ipcai2027_experiment/task22_atlas100/out/manifest/videos.json` |
| `videos/clips.json` | the 438 clips that extraction wrote | `.../manifest/clips_v100.json` |
| `videos/population.txt` | its 96 videos | `.../manifest/population.txt` |
| `videos/skip_list.txt` | the clips the extractor was told to skip (missing or gapped masks); it stops on any other | `.../manifest/jobs.txt` |
| `videos/excluded_short.json` | 39 clips written but under 15 frames | `.../manifest/excluded_short.json` |
| `audit/crop_scope_table.json` | per video, what the crop removes and whether only part of the frame moves | `ipcai2027_experiment/task22_atlas100/out/audit/crop_scope_table.json` |
| `audit/gt_coverage.json` | per clip after cropping, the labelled fraction and empty frames | `.../audit/gt_coverage.json` |
| `audit/ui_residue.json` | per video, burnt-in UI left after cropping | `.../audit/ui_residue.json` |

Clip names: `<procedure>__<video>__gt_<n>` in the population files and the
depth manifest, `<procedure>/<video>/clip_<n>` as `key` elsewhere.

The files are byte copies of the workbench's, except that two absolute paths
(`atlas_root` in `videos.json`, `dst` in `excluded_short.json`) were removed
and ten free-text `note` fields were translated from Japanese. Nothing reads
the notes.

`frame_ratio.json` is the exception: it was assembled from the measurement's
output, which was git-ignored in the workbench and is no longer on disk. The
workbench's `surgical_core/atlas/frame_ratio.py` carries all 97 video names
and, for the 14 ratios above 1, the ratio and its `diff` as constants; the
frame pair each video was matched on, and the `diff` of the 83 at ratio 1,
exist only here. The ratios follow the dataset's own sampling rule,
`max(1, int(fps / 15))` (see "Frame ratios" below). The reader's
`verify_against_bundled` checks a ratio against the pixels but does not find
one: re-measuring a video means calling it once per candidate ratio and
keeping the one that matches.

## Crop rectangles

A rectangle is judged per clip, not per video. The automatic recipe
estimates one rectangle per video from a sample of its frames. When all 494
clips of the 97 videos were checked by eye, that rectangle was wrong for 305
of them. In seven videos the rectangle changes within one mp4, because the
recording conditions switch mid-video. Neither case fits a per-video
rectangle, so each clip was judged, and `crop_rects.json` is that judgement.

Each entry is what the judging tool wrote:

    {"procedure": ..., "video": ..., "clip": "clip_0001",
     "rect": [x, y, w, h], "verdict": "ok" | "ng" | "skip", ...}

`verdict` is one of:

- `ok`: the recipe's rectangle was accepted;
- `ng`: a person redrew the rectangle;
- `skip`: the clip is not to be used.

For `ok` and `ng`, `rect` is the rectangle to use.

## Frame ratios

The frame numbers in a clip index are not the mp4's frame numbers. In 14 of
the 97 videos the annotation numbers frames at a lower rate than the mp4,
and `mp4_frame = native_frame * ratio`. The measured ratios are 2, 3 and 4.
Code that reads the bundled `images/frame_NNNNNN.jpg` uses the number as a
key, and nothing goes wrong. Code that decodes the mp4 by frame number
silently gets the wrong frame: a real frame, from another moment.

### How the ratio was measured

For each video, the bundled JPEG of one frame of its first clip was matched
against the mp4 by mean absolute pixel difference, scanning from the start
or seeking to each candidate ratio. A match differs by 0.7 to 1.9 (the two
compressions). A miss differs by 20 to 190, so there is no ambiguity. The
ratio was checked at three points of each video, early, middle and late, for
all 14 videos and 17 controls with ratio 1. It held at every point.

### Where the ratio comes from

The dataset's extraction script (`download/process_atlas120k.py` in the
ATLAS repository) walks the mp4 and keeps every `max(1, int(fps / 15))`-th
frame. It numbers the kept frames from zero, whether or not they fall in the
surgical section. The division truncates, so 60.00 fps gives 4, 59.94 and
50.00 give 3, 30.00 gives 2, and 29.97, 25, 23.98 and 15 give 1. The 14
videos above ratio 1 are exactly those at 30.00 fps or more, and the
measured ratios agree with the sampling rule for all 97. The dataset's
README says "15 fps", which is loose: a 29.97 fps video is kept at its
native rate.

`frame_ratio.json` is a measurement rather than the sampling rule applied,
because the rule's input is not under our control. The mp4 on disk is
whatever the download produced, not necessarily the file the authors
sampled. The fps OpenCV reports can also fall on either side of the
truncation for a video near 30 fps. The sampling rule says which videos to
suspect and what to expect; the pixels say what is.

## Who reads these

`surgical_core/atlas120k/clip_rects.py` reads the crop rectangles and
`frame_ratio.py` the frame ratios; both take the file's path as an argument.
`surgical_core/clip_time.py` is handed the frame ratios when a manifest does
not record its own.
`tests/test_atlas120k_meta.py` pins what must hold between the files.
