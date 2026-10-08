# The pipeline

The pipeline turns a clip of surgical video into the geometry that a 4D
scene graph is built on. It runs in stages. The extraction stage cuts the
clips from a dataset's release. The depth stage estimates depth and camera
poses for every frame. Later stages segment the frames, track the segments
through time, and export what the viewer reads. Each stage is a module of
`pipeline` and runs as `python -m pipeline.<stage>`. Each stage reads what
the earlier stages wrote into the clip.

## A clip

A clip is a directory with these files:

- `input_images/`: the frames as PNG files. Their names sort in time order.
- `seg_masks/`: the ground-truth masks of the frames that have one. An
  ATLAS-120k mask holds the class id of each pixel, as `NNNNNN_class.png`.
- `frame_manifest.json`: the clip's record, written by the stage that cut
  the clip from its dataset. Its `dataset` key names the dataset; a
  manifest without one belongs to CholecSeg8k.
- `crop_info.json`, for a CholecSeg8k clip only: the rectangle inside the
  endoscope's view that the frames were cropped to.

## The ATLAS-120k extraction stage

```bash
python -m pipeline.extract_atlas120k --atlas-root /path/to/ATLAS --out /path/to/clips \
    --clip-rects atlas120k_meta/crop_rects.json --frame-ratios atlas120k_meta/frame_ratio.json \
    [--videos <procedure>/<video> ...] [--population atlas120k_meta/clips.txt] [--overwrite]
```

The stage cuts the annotated clips of ATLAS-120k videos from the release,
the way the paper's 315 clips were cut. `--atlas-root` holds the release's
`atlas120k/` and the videos' `raw_data/`. For each clip of a video's
`clip_index.json`, the stage splits the clip into runs of consecutive
annotated frames, thins each run to one frame every 0.52 s, and writes each
run that still has 8 frames or more as `<procedure>__<video>__gt_<n>`. A
clip with two such runs becomes two clips, `s1` and `s2`; none of the 315
is one. A run with the frames and rectangle of one already written is not
written again. Into each clip it writes:

- `input_images/`: the release's JPEGs, cropped to the clip's confirmed
  rectangle (`atlas120k_meta/crop_rects.json`) and resized to a long side
  of 854.
- `seg_masks/`: the class ids, read through the evaluator's class table.
- `frame_manifest.json`: every frame has ground truth and is an anchor.

Each video also gets `<procedure>__<video>__extract_report.json`, one row
per clip of its index, saying whether the clip was kept, split, a duplicate,
too short, or without masks. With no `--videos`, the stage extracts every
video whose frame ratio was measured. With `--population`, it refuses a
video whose clips written are not the population's clips of that video.

The stage refuses:

- a video whose frame ratio was not measured, or whose ratio fails the
  pixel check against one of its JPEGs;
- a stride whose step is more than 20 % off 0.52 s;
- a kept clip with no confirmed rectangle, or for which the release holds
  no JPEGs;
- a frame or a mask whose size is not the mp4's;
- a mask with an id or a colour the class table lacks;
- a video that already has output, unless `--overwrite` is given, which
  removes the video's clips and report first;
- clips written that are not the population's.

A refused video is left as it was: the stage writes and removes nothing
until every check has passed, apart from the class table's, which is made
while a mask is written.

`python -m pipeline.extract_atlas120k --help` lists the options.

## The CholecSeg8k crop stage

```bash
python -m pipeline.crop_cholecseg8k --input-dir /path/to/clips --rects cholecseg8k_meta/crop_rects.json \
    --clips VID01_s15_80 [VID25_s15_162 ...] [--overwrite]
```

The stage crops a CholecSeg8k clip to the rectangle inside the endoscope's
view, the rectangle `cholecseg8k_meta/crop_rects.json` gives the clip. It
reads the clip `python -m pipeline.extract_cholecseg8k` wrote, named
`VID<nn>_s15_<start>` as the table's keys are, and writes
`VID<nn>_s15_<start>_crop` beside it. Into the cropped clip it writes:

- `input_images/`: every frame, cropped to the rectangle.
- `seg_masks/`: every colour mask, cropped to the rectangle.
- `frame_manifest.json`: the clip's manifest, with each frame's new
  `image_size` and the rectangle as `crop_info`.
- `crop_info.json`: the rectangle again, which the depth stage requires of
  a CholecSeg8k clip.

The stage refuses:

- a rectangle that lacks a key, holds a value that is not an integer, or
  is empty or reaches past its frame;
- a clip named on the command line that has no directory with
  `frame_manifest.json`, or no rectangle in the table; either stops the
  run before any clip is cropped;
- a clip without an image;
- a manifest that does not list one frame per image;
- a frame or a mask whose size is not the frame's the rectangle was found
  on;
- a cropped clip that already exists, unless `--overwrite` is given, which
  removes it first.

A refused clip is left as it was: the stage writes and removes nothing
until every check has passed.

`python -m pipeline.crop_cholecseg8k --help` lists the options.

## The depth stage

```bash
python -m pipeline.depth --input-dir /path/to/clips [--clips <clip> ...] [--gpu N] [--overwrite] [--no-glb]
```

The stage runs Depth Anything 3 (DA3) on all frames of a clip in one call.
DA3 is the model whose depth the later stages read. Another reconstruction
model runs as a stage of its own and writes into the same directories, under
a `__<model>` suffix. The stage first resizes each frame to 504 pixels on its
longest side (`--process-res`). It writes into the clip:

- `depth_raw/depth_NNNNNN.npy`: one depth map per frame.
- `depth_vis/NNNN.jpg`: each depth map as an image, near warm and far cool.
- `exports/mini_npz/results.npz`: what the later stages read: `depth`
  (N, H, W), `conf` (N, H, W), `extrinsics` (N, 3, 4) from world to camera
  and `intrinsics` (N, 3, 3), all float32.
- `pc_vis/frame_NNNN.glb`: one point cloud per frame, for the viewer.
  `--no-glb` skips them.
- `depth_info` in the manifest: the model, the resolution and the ranges.

The stage refuses:

- a CholecSeg8k clip without `crop_info.json`;
- a clip that already holds the stage's output. `--overwrite` replaces the
  stage's own files and leaves every other file.

`python -m pipeline.depth --help` lists the options.

### Without a CUDA GPU

The stage runs on a CPU, slowly: four frames take about a minute and 6 GB
on a laptop. DA3 depends on `xformers`, which installs only with CUDA, so
install DA3 without its dependencies and add the ones it imports:

```bash
pip install -e ".[render]" torch torchvision
pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
pip install "numpy<2" addict einops evo huggingface_hub imageio moviepy==1.0.3 omegaconf plyfile pycolmap safetensors
```

## The Pi3X stage

```bash
python -m pipeline.pi3x --input-dir /path/to/clips --clips <clip> [--device auto] [--gpu N] [--overwrite]
```

The stage reconstructs a clip with a second model, Pi3X
(`recon3d_wrapper.pi3x`). Pi3X predicts each frame's points and camera pose
together. The stage resizes each frame to at most 255,000 pixels, with sides
that are multiples of 14. It writes into DA3's directories, each name
carrying the suffix `__pi3x`, and leaves DA3's files alone:

- `exports/mini_npz/results__pi3x.npz`: the same four arrays as DA3's
  bundle.
- `depth_vis/NNNN__pi3x.jpg`: the depth images, coloured as DA3's are.
- `pc_vis/frame_NNNN__pi3x.glb`: one point cloud per frame.
- `geometry_sources.pi3x` in the manifest. Each frame gets its cloud's
  centroid, its number of points and the camera's axes, under
  `frames[i].geometry_sources.pi3x`. The run gets its settings, its runtime
  and the round-trip check.

The round-trip check back-projects the stage's depth through its own poses
and compares the points with the ones Pi3X predicted. It also computes the
error under the inverted reading of the poses. The stage refuses:

- a clip that fails the round-trip check. The error must be finite, its
  99.9th percentile must be under 3 % of the median depth, and the inverted
  reading's error must be at least ten times larger.
- a clip whose manifest does not list each frame of `input_images/` once:
  `seq_idx` must run from 0 to N - 1, and `n_frames` must be N. The stage
  checks this before the model runs.
- a clip that already holds the stage's output; the refusal says what is
  there. `--overwrite` removes the stage's own files and manifest records,
  never DA3's, and writes them again.

A refused clip is left as it was: the stage writes and removes nothing
until every check has passed.

`python -m pipeline.pi3x --help` lists the options.
