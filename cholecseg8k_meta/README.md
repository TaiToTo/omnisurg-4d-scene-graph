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

Each rectangle is the one the workbench cropped the clip to. The pilot
scored those cropped clips, and the AE-CAI workshop paper reports them. The
workbench found each rectangle on the clip's own 30 frames, as its
extractor wrote them. It took the largest rectangle inside the pixels whose
90th-percentile luminance over those frames is at least 15, with the
workbench's `detect_endoscope_content_mask` and `largest_inscribed_rect`.
The clips of one video can therefore hold different rectangles. A clip
whose frames the extraction stage writes differently from the workbench
keeps its rectangle; the rectangle is not found again on the new frames.

Run on the workbench's 27 clips with this table, the crop stage writes the
workbench's cropped images, masks and `crop_info.json`, byte for byte.

## Why not the endoscope's circle

A circle fitted to the cholec80 video gives other rectangles. The
workbench's `crop_cholec_frames.py --method circle` takes that rectangle
when it finds the video, and it found none in the run that cropped these
clips. The circle's rectangles differ from these on all 27 clips, by up to
59 px. On VID26 and VID43 the circle is larger than the view, and its
rectangle holds black corners of 2 to 3.6 % of the crop.

## Who reads these

`pipeline/crop_cholecseg8k.py` reads the rectangles. `tests/test_cholecseg8k_meta.py`
pins what must hold between the files.
