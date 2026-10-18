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
clip. The 10 clips without a mask have none in the release either: their
folders are missing from its file list (checked on 2026-10-12 against the
release of 2026-06-30). Nothing is excluded on a score. Crop rectangles
select nothing. `pipeline/extract_atlas120k.py` applies the rule.

A cut is a scene change inside the mp4: the frame before it and the frame
after it show different scenes, or one of them is an inserted slide. A fast
camera move, an instrument entering or leaving, smoke, bleeding or a change
of lighting is not a cut. `cuts.csv` lists every candidate for a cut inside
the 315 clips that was judged, with the judgement on it. All frame numbers
are the mp4's.

- **The search.** A script compared each mp4 frame with the frame 5 after
  it: the mean absolute difference of the two frames in grey, scaled to
  160 pixels wide (0 to 255). A run of consecutive comparisons above 20
  that could separate two frames of a clip is a spot: `spot_first` and
  `spot_last` are its first and last sampled frame, and `d_pix` is the
  largest difference in it.
- **The candidates.** Inside each spot, adjacent frames were compared with
  the same measure. The pair that differs most is the candidate,
  `boundary_before` and `boundary_after`. Another pair of the spot is a
  candidate too when its difference is at least 15 and at least 1.5 times
  the largest difference within three frames of it. The 315 clips hold
  1,506 candidates.
- **What was judged.** The 431 candidates whose spot has `d_pix` above 30,
  and two below 30 that the recheck found to be cuts; only judged candidates
  are listed. The other 1,073 with `d_pix` between 20 and 30 were not
  judged. Since two cuts turned up among them, more may be missing.
- **The judgement.** A person judged each candidate by eye, from the three
  frames on each side of it and, where those did not decide, from the video
  played around it. For a cut, the person chose the exact pair of adjacent
  frames, `cut_before` and `cut_after`; the new scene starts at `cut_after`.
  An earlier pass had judged the candidates above 40 the same way, each in
  a window of three frames on each side. Its judgement of a cut stands
  where the pair is the same; its judgement that a candidate is not a cut
  stands where the candidate falls in its window. Both yield to a new
  judgement of the candidate. 74 rows carry the earlier pass's judgement.
- **The recheck.** Where the cuts of a clip leave a piece of fewer than 8
  frames after thinning, the person watched the piece with a few seconds
  on each side, confirmed or moved the cut at each end, and marked any
  further cut inside it. `rechecked` is 1 for a candidate judged while the
  video was played frame by frame: the 44 cuts at the ends of such pieces,
  and two candidates judged that way before the pieces were listed, one a
  cut and one not (`appendectomy__41RKDh3INiU__gt_0003`, frames 2122 and
  2123).

Of the 433 judged candidates, 72 are cuts, in 39 clips; each falls between
two frames of its clip. The recheck added three of them, all in
`hemicolectomy/5YDMlxTl0k8` (`gt_0029` and `gt_0055`). They are small
changes of scene, and they are listed as cuts like the others: every cut is
treated the same. The extraction stage of this repository does not read
`cuts.csv` yet, so `clips.txt` still holds the 39 clips whole. The paper's
ATLAS-120k run of 2026-10-13 split each clip at its cuts. The change that
brings that extraction stage also replaces `clips.txt`.

`videos/videos.json` is the inventory of the release's tree: all 100 videos
and their 502 clips. An earlier extraction wrote it, so its `keep`,
`dropped` and count fields follow that extraction's rule (stride 3
annotated frames, at least 15 of them, a clip with a mask gap dropped
whole), not the paper's. The overlap check below reads only `native_range`
and `n_native`.

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
| `cuts.csv` | per judged candidate inside the population: its clip, its spot and `d_pix`, the pair of adjacent frames judged, whether it is a cut, for a cut the pair it falls between, and whether it was rechecked; all frame numbers are the mp4's | `experiment/crop_necessity/cut_review_spot_targets.csv` |
| `frame_ratio.json` | per video, the measured ratio between the clip index's frame numbers and the mp4's (1, 2, 3 or 4), with the frame pair it was matched on and their pixel difference; all 97 videos | `outputs/crop_cuts/_align2.json` |
| `videos/videos.json` | the 100-video tree: per video its mp4, clips, frame counts, why any was dropped | `ipcai2027_experiment/task22_atlas100/out/manifest/videos.json` |

Clip names: `<procedure>__<video>__gt_<n>` in `clips.txt` and
`depth_manifest.json`, `<procedure>/<video>/clip_<n>` as `key` in
`crop_rects.json`. `videos.json` gives `procedure`, `video` and `clip` as
separate fields.

The files are byte copies of the workbench's, except that an absolute path
(`atlas_root` in `videos.json`) was removed and the eight `note` fields of
`crop_rects.json` were translated from Japanese. Nothing reads the notes.
`cuts.csv` is rewritten from the workbench's table, which is on its branch
`exp/preproc/02-cut-review`. The columns are renamed: the workbench's
`cut_before` and `cut_after` are `boundary_before` and `boundary_after`
here, its `cut_between` gives `cut_before` and `cut_after`, and `spot_lo`,
`spot_hi`, `is_cut` and `reviewed` are `spot_first`, `spot_last`, `cut` and
`rechecked`. The candidates that were not judged are left out, and so are
the columns the table keeps for the workbench (the key of each candidate,
its one-frame difference, why it became a candidate, where its judgement
came from) and the notes, which are in Japanese. One note asks about a
frame that is not a cut; the rest are typing errors.
The workbench's other files about this tree (the cut marks, the audits of
the earlier extraction, its clip lists) are not carried for now. No
measurement of the paper reads them.

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
`tests/test_atlas120k_meta.py` pins what must hold between the files, and
`tests/test_atlas120k_cuts.py` what must hold in `cuts.csv`.
