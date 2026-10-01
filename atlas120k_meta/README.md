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

## Files

| file | what it is | from the workbench |
|---|---|---|
| `clips.txt` | the population, one `<procedure>__<video>__gt_<n>` per line | `ipcai2027_experiment/frozen/atlas97_clips.txt` |
| `depth_manifest.json` | per clip, sha256 of depth and intrinsics, frame count, shape | `ipcai2027_experiment/frozen/atlas97_depth_manifest.json` |
| `crop_rects.json` | per clip, the rectangle to use (`rect`, in `src_size` pixels) and whether the automatic one was accepted (`verdict`); `rect_used` and `via` are history | `experiment/crop_necessity/verdicts/verdicts_latest.json` |
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

## Who reads these

Nothing yet. `clip_rects.py` and `frame_ratio.py` are ported next under
`surgical_core/atlas120k/`; until then `tests/test_atlas120k_meta.py` pins
what must hold between the files.
