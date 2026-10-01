# ATLAS-120k metadata

What this directory holds is everything about the ATLAS-120k videos that the
measurements depend on and that is not the videos themselves: which clips
enter a measurement, how each clip is cropped, where a recording changes
mid-video, and the fingerprints of the depth each clip was reconstructed
with. No video, frame or mask is here; those come from the dataset's own
distribution.

These files are data, not code. Which videos and clips a measurement reads is
decided by the population file below, never by a count in a directory name.
Every file was produced in the private research workbench the code is being
extracted from; the table gives the path there, so the provenance survives
the renaming.

## The two populations

ATLAS-120k was extracted twice, under two protocols, and both are recorded:

- **The clip population the paper measures: `clips.txt`, 315 clips from 91
  videos.** Cut from the 97 videos that have both an mp4 and a clip index,
  using the per-clip crop rectangles in `crop_rects.json`, one frame every
  0.52 s, and keeping only the longest run of consecutive annotated frames
  per clip that has at least 8 of them. The rule was fixed on paper before the
  extraction ran, and the extraction's output was checked against it clip by
  clip before this list was written; the paper's scores are over these clips
  and their depth is fingerprinted in `depth_manifest.json`.
- **The video manifest from the 100-video tree: `videos/`, 96 videos and 438
  clips.** An earlier extraction of the same 100 videos at a stride of 3
  annotated frames, keeping clips with at least 15 of them. Its manifest is
  the inventory of the raw tree (which videos have an mp4, how many clips and
  frames each has, why a clip or video was dropped), and the audit in
  `audit/` was run on its output. It is kept because the inventory and the
  audit describe the videos, not the extraction, and because the two
  populations differ: five clips of the paper's 315 are not among the 438.

Both populations are decided by where ground truth exists, and nothing is
excluded on a score. Every clip of ATLAS-120k is an annotated segment, so the
only exclusions are a video with no obtainable mp4 and a clip with no mask at
all. The two differ on a clip whose masks have a gap: the earlier extraction
dropped it whole, the paper's keeps the longest run of annotated frames. That
is where the five clips of the 315 that are not among the 438 come from: three
have a mask gap (`cholecystectomy/_-aytJndMV4/clip_0005`,
`hemicolectomy/5YDMlxTl0k8/clip_0024` and `clip_0039`), and the other two are
clips of `5YDMlxTl0k8` that the earlier extraction wrote under another number
(see `videos/clips.json` below).

The crop rectangles and the cut marks are not selection rules. The rectangle
decides which pixels of a frame reach the pipeline and the ground truth alike,
and a clip with a cut in it is used whole; neither removes a clip.

## Files

| file | what it is | from the workbench |
|---|---|---|
| `clips.txt` | The paper's clip population: one clip name per line, `<procedure>__<video>__gt_<n>`. 315 clips, 91 videos. | `ipcai2027_experiment/frozen/atlas97_clips.txt` |
| `depth_manifest.json` | For each of the 315 clips, the sha256 of its reconstructed depth and intrinsics, its frame count and array shape. The pipeline derives normals and edges from depth on the fly, so an unchanged depth means unchanged geometry; this is how a re-run that silently changed the depth is caught. | `ipcai2027_experiment/frozen/atlas97_depth_manifest.json` |
| `crop_rects.json` | The crop rectangle of every clip of the 97 videos, 494 entries, as judged by a person in the review tool. `rect` is the rectangle to use; `verdict` says whether the automatic recipe's rectangle (`rect_recipe`) was accepted (`ok`, 189) or redrawn (`ng`, 305); no clip was marked unusable. Rectangles are in the coordinates of the source video (`src_size`). A video does not have one rectangle: seven videos have clips with different rectangles, because the recording changes within one mp4. `rect_used` is the rectangle an earlier extraction had used, kept as history, and `via` records how a redrawn rectangle was entered; nothing reads either. `first`, `last` and `n_gt` are the clip's frame range and annotated frame count in the clip index. | `experiment/crop_necessity/verdicts/verdicts_latest.json` |
| `cut_marks.jsonl` | Where a person judged whether a candidate scene change is a cut. An append-only log, one line per judgement: the video, the frames before and after, the mean absolute pixel difference `d_pix`, and `is_cut` (`yes`, `no`, `dk` for undecided, `null` for not yet judged). Three boundaries were judged twice, so the last line for a key is its verdict. Candidates were the boundaries with `d_pix` above 40, the smallest difference any observed cut had; boundaries below that were not looked at, except one inside a clip (`d_pix` 27.8) that was checked by hand and is not a cut. Ten cuts were found, in the GT clips of two videos; a clip with a cut in it is used whole. | `experiment/crop_necessity/marks/marks_20260913_174731.jsonl` |
| `videos/videos.json` | The inventory of the 100-video tree: per video, its split, procedure, whether it is robotic, whether the mp4 is present, its clips with their frame counts and the reason any was dropped, the pixel source (bundled JPEG or decoded mp4) and the mask format (index or RGB). Also the extraction's parameters (`stride`, `min_frames`). | `ipcai2027_experiment/task22_atlas100/out/manifest/videos.json` |
| `videos/clips.json` | The 438 clips that extraction wrote, with their video and frame totals. `videos.json` is the inventory before extraction and this is what came out; for one video (`hemicolectomy/5YDMlxTl0k8`) the two number the clips differently, because that extraction named its outputs by position in the clip index rather than by the clip's own number. | `.../manifest/clips_v100.json` |
| `videos/population.txt` | The 96 videos of that extraction, `<procedure> <video>` per line. The four left out: three with no mp4, one with no clip long enough. | `.../manifest/population.txt` |
| `videos/skip_list.txt` | `population.txt` with a third column naming the clips the extractor was told to skip because their masks are missing or have gaps, or `-`. The extractor skips only what is named here and stops on any other missing mask, so a clip cannot drop out silently. | `.../manifest/jobs.txt` |
| `videos/excluded_short.json` | The 39 clips that extraction wrote but that had fewer than 15 frames after the stride, and were set aside. | `.../manifest/excluded_short.json` |
| `audit/crop_scope_table.json` | Per video, how much of the frame the crop removes relative to the black surround, how far the fixed rectangle is from the frame content over time, and whether only part of the frame moves (a console recording rather than an endoscope feed). It lists; it excludes nothing. | `ipcai2027_experiment/task22_atlas100/out/audit/crop_scope_table.json` |
| `audit/gt_coverage.json` | Per clip after cropping, the fraction of labelled pixels (mean and minimum over frames), the number of frames with no label at all, and the classes present. Checks that cropping did not cut away annotated anatomy. | `.../audit/gt_coverage.json` |
| `audit/ui_residue.json` | Per video, the rows and columns of static burnt-in UI left after cropping: at the frame edges, where a crop could have removed them, and inside the field of view, where it cannot and the residue is recorded as a covariate. | `.../audit/ui_residue.json` |

Clip names appear in two spellings, both from the dataset's layout:
`<procedure>__<video>__gt_<n>` in the population files and the depth manifest,
and `<procedure>/<video>/clip_<n>` as `key` in the crop rectangles and the
video manifest. The same clip has the same `<n>` in both.

## What was changed on the way in

The files are byte copies of the workbench's except for these edits, each
made because the repository is public and English:

- `videos/videos.json` lost its `atlas_root` key and `videos/excluded_short.json`
  its `dst` key; both held an absolute path on the workbench's machine.
- The free-text `note` fields in `crop_rects.json` (8 entries) and
  `cut_marks.jsonl` (2 lines) were translated from Japanese. Nothing reads
  them.
- The JSON files that were edited were written back with the same formatting
  (indent of one space), so a diff against the workbench shows only the
  removed keys and the translated notes.

## What was left out, and why

- `marks_20260913_150749.jsonl`, the earlier snapshot of the cut marks: the
  later file is the earlier one plus one line.
- `clips_v100.txt`: the same list as `videos/clips.json`, comma-separated.
- `extracted_videos.txt`: a list of 91 videos nothing in the workbench writes
  or reads any more.
- `production_frozen.json`: sha256 fingerprints of the workbench's own source
  files at the time of the extraction. It describes workbench code, not the
  dataset, and the workbench marks it stale.
- The `.md` tables next to the audit JSONs: they are the same rows rendered
  in Japanese.
- `verdicts.jsonl` and `verdicts_before_reset_*.jsonl`, the append-only
  histories behind `crop_rects.json`: the latest judgement per clip is what
  the extraction reads, and that is what is here.

## Who reads these

Nothing in this repository reads them yet. The readers (`clip_rects.py` for
the crop rectangles, `frame_ratio.py` for the measured ratio between a clip
index's frame numbers and the mp4's) are ported next, under
`surgical_core/atlas120k/`. Until then `tests/test_atlas120k_meta.py` pins
what must hold between the files.
