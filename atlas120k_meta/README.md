# ATLAS-120k metadata

What the measurements need to know about the ATLAS-120k videos, without the
videos: which clips are measured, how each is cropped, how each video's clip
index numbers its mp4's frames, and a fingerprint of each clip's depth. No
video, frame or mask is here. Which clips a measurement reads is decided by
`clips.txt`, never by a count in a directory name.

## The population

`clips.txt` lists the **315 clips from 91 videos** the paper measures. The
rule, fixed before the extraction ran: from the 97 videos with an mp4 and a
clip index, split each clip into runs of consecutive annotated frames, take
one frame every 0.52 s, and keep every run that still has at least 8. A clip
with two kept runs would give two clips, `s1` and `s2`; none of the 315 is
one. Of the 494 clips in the index, 167 were too short after thinning (six
videos lost every clip), 10 have no mask, and 2 are duplicates of another
clip. Nothing is excluded on a score. Crop rectangles select nothing. A clip
with a cut in its recording is used whole. `pipeline/extract_atlas120k.py`
applies the rule.

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
its clip lists) are not carried for now. No measurement of the paper reads
them. `docs/porting.md` lists them under "Not carried for now".

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
