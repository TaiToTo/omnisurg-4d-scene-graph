# StereoMIS metadata

Which StereoMIS clips the camera trajectory result measures, and, for each
frame of a clip, its place in the video, its ground truth row, its time and
its tissue mask. No frame, mask or pose is here. Every StereoMIS stage and
score reads the clips from `clips.json`, never from the dataset.

## The clips

`clips.json` lists the **86 clips of the ten sequences, 43 of them usable**,
one clip a line. A clip is 112 frames, every 12th frame of the video, and
the clips of a sequence do not overlap. The grid starts at position 4 and
does not depend on the offsets, so measuring an offset again moves no
frame. A clip is left out for its inputs only, never for a score, by the
first of these reasons that holds:

| reason | what holds | clips |
|---|---|---|
| `no_surface` | fewer than half its depth maps show a surface: a median from 30 to 400 mm, more than half the pixels valid | 0 |
| `gt_jump` | the ground truth steps more than 5 mm or 5 degrees between two rows inside it | 17 |
| `gt_static` | the true camera's RMS radius is below 1 mm | 26 |

## Fields

Each clip holds:

- `name` and `seq`;
- `usable`, true exactly when `drop`, the reason it is left out, is "";
- `span_mm`, the RMS radius of the true camera centres, and `stratum`:
  `moving` from 10 mm up, `slow` below;
- `depth_ok`, the share of its depth maps that show a surface;
- `stride` and `frames`, the frames' 0-based positions in the video, as
  FFmpeg's `select=eq(n,N)` decodes them;
- `gt_row_offset`: a frame's row of `groundtruth.txt` is its position plus
  this offset, −3 in every sequence;
- `times`: each frame's timestamp in the video, in s. The P2 videos declare
  59.94 frames a second, but by their timestamps 12 frames span 0.217 s at
  the median: a usable P2 clip lasts 24.2 to 24.9 s, and a P1 clip 22.2 s;
- `masks`: the file number of the tissue mask of the frame nearest each
  frame that has one, within two frames, or null. Every sequence numbers its
  files one ahead of its frames. Each frame of a usable P2 clip has the mask
  of its own number; 504 of the 672 frames of P1's usable clips do, and the
  other 168 lie outside P1's masked frames and have none.

## How the list is made

```bash
python -m pipeline.select_stereomis_clips --root /path/to/StereoMIS --depth-root /path/to/StereoMIS_depth \
    --out stereomis_meta/clips.json
```

The step reads each sequence's video, ground truth and masks, and the
`stats.npy` of a depth export the workbench made, which gives each depth
map's median and valid share. No tracked file makes that export, and no
other step reads it. `surgical_core/stereomis.py` holds the offsets and the
rule. A score records the list's hash as `clip_list_sha`, and
`trajectory_eval.tools.scores` refuses to compare scores of two lists.
