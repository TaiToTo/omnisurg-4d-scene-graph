# The pipeline

The pipeline turns a clip of surgical video into the geometry that a 4D
scene graph is built on. It runs in stages. An extraction stage cuts the
clips from a dataset's release, one stage per dataset. The depth stage
estimates depth and camera poses for every frame. Later stages segment the frames, track the segments
through time, and export what the viewer reads. Each stage is a module of
`pipeline` and runs as `python -m pipeline.<stage>`. Each stage reads what
the earlier stages wrote into the clip.

## A clip

A clip is a directory with these files:

- `input_images/`: the frames as PNG files. Their names sort in time order.
- `seg_masks/`: the ground-truth masks of the frames that have one. An
  ATLAS-120k mask holds the class id of each pixel, as `NNNNNN_class.png`.
  A CholecSeg8k mask holds the class colour of each pixel, as
  `NNNNNN_color_mask.png`. A LapEx mask holds the grey level of each
  pixel's class, as `NNNNNN_class.png`.
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

### A video OpenCV cannot decode

```bash
python -m pipeline.prepare_atlas120k_videos --src /path/to/ATLAS --dst /path/to/ATLAS_h264
```

One video of the 97, `rarp/NitKIjCcS7U`, is AV1, which some builds of
OpenCV cannot decode. This command builds a second root for the extraction
to take as `--atlas-root`. Its `atlas120k/` is a symlink to the release's.
Under its `raw_data/`, every H.264 video is a symlink to the release's
file, and every other video is converted to H.264 with FFmpeg, keeping its
size, frame rate and frame count. `video_root.json` records what was done
to each video, with the `--crf` a conversion was made at. The command
refuses:

- a release without `atlas120k/` or without an mp4;
- a video with no video stream, or whose stream does not say how many
  frames it has, since a conversion is checked by that count;
- a converted video whose size, frame rate or frame count is not the
  source's, that is not H.264, or that OpenCV cannot read a frame from;
- a converted video that is already there, unless `video_root.json`
  records it at the `--crf` given;
- a path in the new root that is already something else.

A converted video that is already there is checked, not made again. A
conversion is written under a temporary name and moved into place when
FFmpeg has finished, so a run that stops leaves no partial video. The
extraction's output does not depend on the conversion: it reads the frames
from the release's JPEGs, and takes from the mp4 only its frame rate and
size, which the conversion keeps. Its frame ratio check does read the
converted pixels, and allows for the compression.

## The CholecSeg8k extraction stage

```bash
python -m pipeline.extract_cholecseg8k --seg8k-root /path/to/CholecSeg8k --videos-root /path/to/cholec80/videos \
    --out /path/to/clips --clips VID01_s15_80 [VID25_s15_162 ...] [--count 30] [--overwrite]
```

The stage extracts the clips named, each `VID<nn>_s<stride>_<start>`:
`--count` frames of cholec80 video `nn`, numbered in CholecSeg8k from
`start` in steps of `stride`. `--seg8k-root` holds the release's
`video<nn>/` directories; `--videos-root` holds cholec80's `video<nn>.mp4`.
A frame CholecSeg8k annotated takes the image the mask was drawn on, and
the stage finds the video frame that image is by matching it against the
video. A frame CholecSeg8k did not annotate is decoded from the video, at
the frame interpolated between the clip's annotated frames, or carried on
at their rate past the first or the last. Into each clip it writes:

- `input_images/`: the annotated images and the decoded frames, as PNG.
- `seg_masks/`: the release's colour masks, on the annotated frames only.
- `frame_manifest.json`: for each frame its video frame, its CholecSeg8k
  number where it has one, and whether it is an anchor.

The stage refuses:

- a clip name that is not `VID<nn>_s<stride>_<start>`, before any video
  is read;
- a video, or a CholecSeg8k directory, that is missing;
- a video that cannot be opened or reports no frame count;
- a mask without the image it was drawn on;
- an annotated image that is not the video's size, that matches no frame
  of its search window, or that matches a frame on the window's edge;
- a video frame that cannot be decoded;
- a clip with fewer than two annotated frames, or whose annotated frames
  run at a rate CholecSeg8k numbers no video at;
- a clip whose frames do not rise in the video;
- a clip that already exists, unless `--overwrite` is given.

A clip is written under a temporary name and moved into place when it is
complete. A refused clip leaves the earlier clip as it was. When one clip
is refused, the others are still extracted, and the failures are listed at
the end.

`python -m pipeline.extract_cholecseg8k --help` lists the options.

## The CholecSeg8k crop stage

```bash
python -m pipeline.crop_cholecseg8k --input-dir /path/to/clips --rects cholecseg8k_meta/crop_rects.json \
    --clips VID01_s15_80 [VID25_s15_162 ...] [--overwrite]
```

The stage crops a CholecSeg8k clip to its rectangle in
`cholecseg8k_meta/crop_rects.json`. The rectangle lies inside the
endoscope's view; the README of `cholecseg8k_meta/` says how it was
found. The stage reads the clip that `python -m pipeline.extract_cholecseg8k`
wrote, named `VID<nn>_s15_<start>` as the table's keys are. It writes
`VID<nn>_s15_<start>_crop` in the same directory. Into the cropped clip it
writes:

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
- a frame or a mask whose size is not `src_w` × `src_h`, the size of the
  frames the rectangle was found on;
- a cropped clip that already exists, unless `--overwrite` is given;
- a cropped clip that holds anything a later stage wrote, such as
  `depth_raw/`, even with `--overwrite`. Such a clip is removed by hand.

The stage writes the cropped clip as `<clip>_crop.part` and renames it when
every file is written. Only then does `--overwrite` remove the earlier
cropped clip. A run that fails, on a refusal or while it writes, leaves the
clip and an earlier cropped clip as they were.

`python -m pipeline.crop_cholecseg8k --help` lists the options.

## The LapEx extraction stage

```bash
python -m pipeline.extract_lapex --lapex-root /path/to/LapEx_dataset --out /path/to/clips \
    [--cases 01 02 ...] [--overwrite]
```

The stage extracts every annotated frame of LapEx as a clip of its own.
LapEx annotates single frames of each case's video, seconds apart, so each
clip holds one frame. `--lapex-root` holds the release's `metadata/` and one
directory per case, `01` to `30`. A case holds its video's frames in
`frames/` and its masks in `seg/`, each named by its time in milliseconds.
The frame of the mask at `<ms>` becomes the clip `<case>__gt_<ms>`, which
holds:

- `input_images/000000.png`: the frame.
- `seg_masks/000000_class.png`: the mask's grey levels, each a class of the
  release's `metadata/segmented_entity.csv`. Interstitial space, level 0,
  is written as 11, a level no class uses. LapEx labels every pixel, and
  the workbench's evaluators read id 0 as background.
- `frame_manifest.json`: the frame's time and video frame, the class table,
  the moved level, and the levels of the instruments and of the gauze.

A run that extracts every case of the release writes
`extraction_summary.json`, which counts the clips of each case. Such a run
is given no `--cases`, or names every case with it. Any other run removes
the summary an earlier run wrote:

- a run given only some of the cases;
- a run in which a case is refused.

So a summary always counts the clips of one run over the whole release.

The frames are not cropped: they keep the black surround of the
endoscope's view, so the depth stage refuses the clips.

The stage refuses, before any case is extracted:

- a release whose `metadata/segmented_entity.csv` is not the class table
  the stage was written for, or lists a level twice;
- a case named on the command line that is not one of the release's, `01`
  to `30`, such as `01/` or `../01`;
- a case named on the command line that has no directory.

In each case, the stage refuses:

- a case without a mask;
- a file in `seg/` whose name is not `<ms>_seg.jpg`;
- a mask whose time is not a whole frame at 25 fps;
- a mask without the frame of its time;
- a mask or a frame that cannot be read;
- a mask whose size is not its frame's;
- a mask with a level the class table lacks. The masks are JPEGs, so a
  level the compression shifted is refused too;
- a clip that already exists, unless `--overwrite` is given;
- a clip that holds anything a later stage wrote, such as `depth_raw/`,
  even with `--overwrite`. Such a clip is removed by hand.

A refused case is left as it was: the stage writes and removes nothing until
every frame of the case has been read and checked. A clip is written under a
temporary name and moved into place when it is complete. When one case is
refused, the others are still extracted, the failures are listed at the end,
and no summary is written.

`python -m pipeline.extract_lapex --help` lists the options.

## The StereoMIS extraction stage

```bash
python -m pipeline.extract_stereomis --root /path/to/StereoMIS --depth-root /path/to/StereoMIS_depth \
    --out /path/to/clips [--sequences P1 P2_0 ...] [--mask-instruments]
```

The stage writes the clips the camera trajectory result is measured on.
`--root` holds one directory per sequence, as StereoMIS ships it.
`--depth-root` holds the depth export's `<sequence>/stats.npy`, from which
the stage tells the clips with no surface in view. The clips are those
`surgical_core.stereomis.clips` marks usable: 112 frames 0.2 s apart, cut
from each sequence without overlap. Into each clip the stage writes:

- `input_images/`: the left view of each frame, rectified and at half
  resolution, 640x512, numbered from 0.
- `frame_manifest.json`: for each image its position in the video
  (`native_frame`), its ground truth row (`gt_row`) and its time, and for
  the clip its sequence, stride, stratum and the RMS radius of its true
  trajectory (`span_mm`).

With `--mask-instruments`, each image is painted black outside the tissue
mask nearest its frame, and `instruments_masked` counts the images
painted. A frame with no mask within two frames keeps its image. The
paper's main condition reads the clips without it.

The stage decodes each sequence's frames in one pass of FFmpeg. It writes
a clip as `<clip>.partial` and renames it when its manifest is written.
It refuses a clip whose directory, or `<clip>.partial`, exists already,
for every sequence named, before anything is decoded. A clip is not
replaced: the depth stages write into it.

The stage needs the `stereomis` extra and FFmpeg.

## The D4D population

```bash
python -m pipeline.select_d4d_population --census /path/to/census.json \
    --clips d4d_meta/census_clips.txt --out d4d_meta
```

The D4D measurements score sides, not clips. The step selects the sides by
the rule in `d4d_meta/README.md`, from a census of their inputs. It writes
`population.json`, which lists the kept sides and counts the sides left out
for each reason, and `clips.txt`, which lists the clips with a kept side.

The step refuses, before it writes anything:

- a list of clips that is empty, holds a clip twice, or holds a line that is
  not `<specimen>/<session>/<clip>`;
- a census that does not hold each clip of the list once and in its order,
  such as a census cut short;
- a side that carries an `error`, which the census wrote when it failed to
  measure the side;
- a field the rule reads that is missing or holds a value of the wrong kind,
  such as a side with a point cloud and no `active`, or a `NaN`.

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

- a clip without `crop_info.json`, unless it is an ATLAS-120k clip: a
  CholecSeg8k clip before the crop stage, or a LapEx clip;
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

### Every clip of a population

```bash
python -m pipeline.depth_population --input-dir /path/to/clips --clips atlas120k_meta/clips.txt [--gpus 0 1 2 3]
```

The command runs the depth stage on every clip of a population file, one
process per GPU, on CUDA only, at the stage's default model and resolution,
and without the point clouds. A clip whose manifest already holds
`depth_info` is skipped. A clip that holds the bundle without `depth_info`
is run again; the stage refuses it until its files are removed or
`pipeline.depth --overwrite` is run on it. Each process writes its output
to `<input-dir>/_logs/depth_gpu<N>.log`. A driver that is stopped, by
`kill`, a closed terminal or Ctrl-C, stops its processes with it. The
command refuses:

- before any process starts: a clip of the population that is not under
  `--input-dir` or whose manifest cannot be read; a clip whose
  `depth_info` lacks a key the stage writes, or records another model or
  resolution than the stage runs at; a GPU listed twice;
- after the run: a clip of the population that lacks `depth_info` or its
  bundle, whose bundle cannot be read, holds other keys than the stage
  writes, or has another number of depth maps than the clip has images,
  or whose `depth_info` records a filled border (a `ray_map` in the
  bundle and a filled border each mark another version of the stage);
- a population whose `depth_info` records more than one model or
  resolution;
- a process that exited non-zero; the message names its log.

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

## The tracking stage

```bash
python -m pipeline.track --input-dir /path/to/clips --tracks-root /path/to/tracks --tag <tag> \
    --rule both_ways_from_centre --sam-input normal_edge --track-base rgb --sam-ckpt sam_vit_h_4b8939.pth \
    [--clips <clip> ...] [--seed-labels DIR] [--depth-source pi3] [--keep-edge-ring] [--overwrite]
```

The stage cuts one frame of each clip into regions and carries them through
the clip with SAM 3's video tracker (`sam3_wrapper`). SAM's automatic mask
generator cuts the seed frame, prompted with the `--sam-input` image;
`--seed-labels` reads the regions from a directory instead. The rule places
the seed:

- `both_ways_from_centre` seeds the middle frame and carries both ways.
- `forward_from_first` seeds frame 0 and carries forwards.

The stage reads the depth stage's bundle, and Pi3X's with
`--depth-source pi3`. It reads no GT. It writes, for each clip:

- `<tracks-root>/<clip>/track_<track-base>_<tag>/label_NNNN.npy`: one
  label map per frame, at the depth's resolution, -1 where no object is.
- `seed_info.json` beside the labels: the seed frame, whether the stage
  carried both ways (`bidir`), the frames labelled and the settings the
  seed was made with. Its `seed_source` is `sam`, or `external` when the
  seed was read with `--seed-labels`.
- `<tracks-root>/<clip>/viz/montage_track_<track-base>_<tag>.png`: every
  frame's labels, one colour per object.

The stage refuses:

- a missing image or a missing bundle.
- an unknown input mode, rule or depth source.
- seed regions from `--seed-labels` that are missing, of another shape than
  the depth, or empty.
- Pi3X depth with another number of frames than DA3's.
- a seed frame with no region of `--seed-min-area` pixels or more.
- a condition an earlier run left, unless `--overwrite` is given.

A clip that fails keeps the labels an earlier run left: the stage replaces
them only once it has written every file. The stage runs every clip and
exits with an error if one failed.

The stage needs the `track` extra: `pip install -e ".[track]"`. The SAM
ViT-H weights, `sam_vit_h_4b8939.pth`, are downloaded by hand; SAM 3's
weights, `facebook/sam3`, download from Hugging Face on first use.

`python -m pipeline.track --help` lists the options.

## The per-frame segmentation stage

```bash
python -m pipeline.per_frame --input-dir /path/to/clips --tracks-root /path/to/tracks --tag <tag> \
    --sam-input normal_edge --sam-ckpt sam_vit_h_4b8939.pth \
    [--clips <clip> ...] [--depth-source pi3] [--keep-edge-ring] [--overwrite]
```

The stage cuts every frame of each clip into regions with SAM's automatic
mask generator, prompted with the `--sam-input` image. It carries nothing
from one frame to the next, so an id does not follow an object between
frames. Regions under `--seed-min-area` pixels are dropped.

The stage reads the depth stage's bundle, and Pi3X's with
`--depth-source pi3`. It reads no GT. It writes, for each clip:

- `<tracks-root>/<clip>/track_rgb_<tag>/label_NNNN.npy`: one label map per
  frame, at the depth's resolution. Region `r` is written as `r + 1`, the
  id the tracking stage gives it, and -1 where no region is.
- `seed_info.json` beside the labels: `seed_source` `per_frame`, which the
  evaluator reads as the condition's rule, and the settings the labels
  were made with.

The directory is named as the tracking stage's are, and the evaluator reads
it the same way.

The stage refuses:

- a missing image or a missing bundle.
- an unknown input mode or depth source.
- Pi3X depth with another number of frames than DA3's.
- a condition an earlier run left, unless `--overwrite` is given.
- with `--overwrite`, a condition whose `seed_info.json` does not record
  `seed_source` `per_frame`, such as one the tracking stage wrote.

A clip that fails keeps the labels an earlier run left: the stage replaces
them only once it has written every file. The stage runs every clip and
exits with an error if one failed.

The stage needs the `track` extra and the SAM ViT-H weights, as the
tracking stage does.

`python -m pipeline.per_frame --help` lists the options.

## A condition on every clip of a population

```bash
python -m pipeline.condition_population track --input-dir /path/to/clips --clips atlas120k_meta/clips.txt \
    --tracks-root /path/to/tracks --tag <tag> --rule both_ways_from_centre --sam-input normal_edge \
    --track-base rgb --seed-edge-gain 1.0 --seed-no-smooth --sam-ckpt sam_vit_h_4b8939.pth --gpus 0 1 2 3
python -m pipeline.condition_population per_frame --input-dir /path/to/clips --clips atlas120k_meta/clips.txt \
    --tracks-root /path/to/tracks --tag <tag> --sam-input normal_edge --sam-ckpt sam_vit_h_4b8939.pth \
    [--points-per-side 8] [--depth-source pi3] --gpus 0 1 2 3
```

The command runs one condition of the tracking stage or of the per-frame
stage on every clip of a population file. The subcommand names the stage,
and the stage's options keep their names and defaults.

- Each GPU of `--gpus` runs one clip at a time, on CUDA.
- A clip that already has its labels is skipped, so a stopped run resumes.
- Each clip's log is `<tracks-root>/_logs/<labels>/<clip>.log`. `<labels>`
  is the label directory: `track_<track-base>_<tag>`, or `track_rgb_<tag>`
  for the per-frame stage.
- A stopped driver stops its processes. Under `nohup`, a closed terminal
  does not stop the driver.
- Labels removed by hand are removed with their montage,
  `viz/montage_<labels>.png`.

The command refuses:

- before any process starts: an input a process would fail on, labels made
  with other settings, and a tracker's input that burns edges with a seed
  whose record holds no edge ring ("The edge ring of the tracker's input"
  in `docs/porting.md`);
- after a clip: a record of other settings, which stops the run;
- after the run: labels that are missing, incomplete or of other settings,
  and a process that exited non-zero.

## The granularity conditions

The granularity result tracks seeds merged to K regions. These three commands
make the tracked condition at K = 10. The output of `kmerge`, the per-frame
map merged to 10, is scored as a condition too.

```bash
python -m pipeline.per_frame --input-dir /path/to/clips --tracks-root /path/to/tracks --tag rgb_per_frame \
    --sam-input rgb --points-per-side 24 --sam-ckpt sam_vit_h_4b8939.pth
python -m evalkit.tools.kmerge --dataset atlas120k --clips atlas120k_meta/clips.txt --data-root /path/to/clips \
    --tracks-root /path/to/tracks --tag track_rgb_rgb_per_frame --k 10
python -m pipeline.track --input-dir /path/to/clips --tracks-root /path/to/tracks --tag rgb_k10 \
    --rule both_ways_from_centre --sam-input rgb --track-base rgb \
    --seed-labels '/path/to/tracks/{clip}/track_rgb_rgb_per_frame_k10'
```

The floor condition tracks the unmerged seed:

```bash
python -m pipeline.track --input-dir /path/to/clips --tracks-root /path/to/tracks --tag rgb \
    --rule both_ways_from_centre --sam-input rgb --track-base rgb --points-per-side 24 \
    --sam-ckpt sam_vit_h_4b8939.pth
```

- K = 6, 8 and 12 are made the same way.
- On CholecSeg8k, `kmerge` takes `--dataset cholecseg8k --clips
  cholecseg8k_meta/clips.txt`.
- `kmerge` reads a clip as the evaluator does, so it needs the GT masks and
  valid depth at every pixel of every frame.
- The evaluator refuses a condition tracked with `--seed-labels` ("Propagation
  rule" in `docs/evaluation.md`). It scores the floor and the per-frame
  conditions.
