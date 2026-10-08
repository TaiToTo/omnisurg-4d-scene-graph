# CholecSeg8k metadata

What the measurements need to know about the CholecSeg8k videos, without the
videos: which clips are measured, and the rectangle each clip is cropped to. No
video, frame or mask is here.

## The population

`clips.txt` lists the **27 clips from 17 videos** the paper measures, one
`VID<nn>_s15_<start>_crop` per line. A clip is a window of 30 frames of
video `nn`, numbered in CholecSeg8k from `start` in steps of 15, and cropped
to its rectangle. Every `start` is the first frame of one of CholecSeg8k's
annotated chunks of 80 frames. These are the clips the pilot scored, and
the windows the workbench's `run_all17_pipeline.sh` lists.

`python -m pipeline.extract_cholecseg8k --clips VID<nn>_s15_<start>` writes
a window; `python -m pipeline.crop_cholecseg8k` writes its `_crop`.

## The rectangles

`crop_rects.json` gives each clip the rectangle it is cropped to, in the
frame's pixels (`src_w` × `src_h`, 854 × 480 for all 17 videos). `y1` and
`x1` are exclusive. Its keys are the clips before the crop, `VID<nn>_s15_<start>`
without `_crop`. The crop stage writes the clip's rectangle into the clip
as `crop_info.json`, the layout the depth stage requires.

Each rectangle is the largest that lies inside both the frame and the
endoscope's circle, so the clips of one video share it. The circle was
fitted to 60 frames sampled across the cholec80 video. Its radius was
scaled by 0.98 before the rectangle was taken, to keep the rectangle off
the vignette at the circle's edge. The workbench fitted the circle and took
the rectangle when it cropped each clip, with `surgical_core/preprocess`'s
`fit_endoscope_circle` and `circle_frame_inscribed_rect`. The rectangles
here were computed again with those functions, from the same videos. The
table is per clip so that it can record the rectangle each clip was cropped to,
whether or not the clips of a video share one.

## Who reads these

`pipeline/crop_cholecseg8k.py` reads the rectangles. `tests/test_cholecseg8k_meta.py`
pins what must hold between the files.
