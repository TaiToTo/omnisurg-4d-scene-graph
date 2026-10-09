# CholecSeg8k metadata

What the measurements need to know about the CholecSeg8k videos, without the
videos: which clips are measured, and the rectangle each clip is cropped to. No
video, frame or mask is here.

## The population

`clips.txt` lists the **27 clips from 17 videos** the paper measures, one
`VID<nn>_s15_<start>_crop` per line. A clip is a window of 30 frames of
video `nn`, numbered in CholecSeg8k from `start` in steps of 15, and cropped
to its rectangle. Every `start` is the first frame of one of CholecSeg8k's
annotated chunks of 80 frames. These are the windows the pilot scored, and
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
the vignette at the circle's edge. The rectangles were computed with the
workbench's `fit_endoscope_circle` and `circle_frame_inscribed_rect`, the
functions its `crop_cholec_frames.py --method circle` crops with. The table
is per clip, so a clip could be given a rectangle of its own.

The workbench's own clips, which the pilot scored and AE-CAI reports, were
cropped to other rectangles. The run that cropped them found no cholec80
video, so the script fell back to each clip's own 30 frames. It took the
largest rectangle inside the pixels whose 90th-percentile luminance over
those frames is at least 15. Those rectangles differ from these on all 27
clips, by up to 59 px, and the clips of one video differ from one another.
The measurements here crop every clip to the rectangles in this table.

## Who reads these

`pipeline/crop_cholecseg8k.py` reads the rectangles. `tests/test_cholecseg8k_meta.py`
pins what must hold between the files.
