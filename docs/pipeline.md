# The pipeline

The pipeline turns a clip of surgical video into the geometry that a 4D
scene graph is built on. It runs in stages. The depth stage estimates depth
and camera poses for every frame. Later stages segment the frames, track the
segments through time, and export what the viewer reads. Each stage is a
module of `pipeline` and runs as `python -m pipeline.<stage>`. Each stage
reads what the earlier stages wrote into the clip.

The stages are moving here one at a time from the research workbench where
they were measured. A stage counts as moved once it writes the same files as
the workbench's stage, byte for byte. The depth stage is the first.

## A clip

A clip is a directory with these files:

- `input_images/`: the frames as PNG files. Their names sort in time order.
- `frame_manifest.json`: what the extraction recorded about the clip. Its
  `dataset` key names the dataset, and `frames` lists each frame's place in
  the source video. A manifest without `dataset` belongs to CholecSeg8k.
- `crop_info.json`, for a CholecSeg8k clip only: the rectangle that the
  frames were cut to. The rectangle lies inside the endoscope's view, so the
  frames hold no black border. It gives `x0`, `x1`, `y0` and `y1` in the
  source frame, which is `src_w` by `src_h` pixels. All clips of one video
  share one rectangle.

## The depth stage

```bash
python -m pipeline.depth --input-dir /path/to/clips --clips <clip> [--device auto] [--gpu 0] [--no-glb]
```

The stage runs Depth Anything 3 (`recon3d_wrapper.da3`) on all frames of a
clip in one call. It first resizes each frame to 504 pixels on its longest
side. It writes these files into the clip:

- `depth_raw/depth_NNNNNN.npy`: one depth map per frame.
- `depth_vis/NNNN.jpg`: each depth map as an image. Near is warm and far is
  cool; each frame is normalised on its own.
- `exports/mini_npz/results.npz`: the bundle that the later stages read. It
  holds `depth` (N, H, W), `conf` (N, H, W), `extrinsics` (N, 3, 4) from
  world to camera, and `intrinsics` (N, 3, 3), all as float32.
- `pc_vis/frame_NNNN.glb`: one point cloud per frame, in glTF's frame. Each
  cloud is centred on its centroid and coloured by its frame. A frame with
  no usable depth gets no cloud. `--no-glb` skips the clouds, since only the
  viewer reads them.
- `depth_info` in the manifest: `model`, `process_res`, `depth_shape`,
  `depth_range`, `conf_range` and `border_inpaint`. `border_inpaint` is
  always false.

The stage refuses a CholecSeg8k clip that has no `crop_info.json`. Its frames
would still hold the black border around the endoscope's view, and the model
would see it.

### Without a CUDA GPU

The stage also runs on a CPU, slowly. On a laptop, four frames take about a
minute and 6 GB of memory, and both grow faster than the number of frames.
DA3 lists `xformers` as a dependency, and `xformers` installs only next to
CUDA. DA3-LARGE does not use it. On a machine without CUDA, install DA3
without its dependencies, then add the ones it imports:

```bash
pip install -e ".[render]" torch torchvision
pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
pip install "numpy<2" addict einops evo huggingface_hub imageio moviepy==1.0.3 omegaconf plyfile pycolmap safetensors
```

These packages are the ones that loading and running DA3 imports. The rest
of its list (`open3d`, a web server, `e3nn`) serves parts of DA3 that the
stage does not run.

## The Pi3X stage

```bash
python -m pipeline.pi3x --input-dir /path/to/clips --clips <clip> [--device auto] [--gpu 0]
```

The stage reconstructs a clip with Pi3X (`recon3d_wrapper.pi3x`), a second
model beside DA3. Pi3X predicts each frame's points and camera pose
together. The stage resizes each frame to at most 255,000 pixels, with sides
that are multiples of 14. It writes its files beside DA3's, each name
carrying the suffix `__pi3x`, and leaves DA3's files alone:

- `exports/mini_npz/results__pi3x.npz`: the same four arrays as DA3's
  bundle.
- `depth_vis/NNNN__pi3x.jpg`: the depth images, coloured as DA3's are, so
  that the two can be set side by side.
- `pc_vis/frame_NNNN__pi3x.glb`: one point cloud per frame. The stage first
  removes this source's clouds from an earlier run.
- `geometry_sources.pi3x` in the manifest. Each frame gets its cloud's
  centroid, its number of points and the camera's axes, under
  `frames[i].geometry_sources.pi3x`. The run gets its settings, its runtime
  and the round-trip check.

The round-trip check back-projects the stage's depth through its own poses
and compares the points with the ones Pi3X predicted. It also places the
points under the inverted reading of the poses. The right reading must win
by at least ten times, and its error at the 99.9th percentile must stay
under 3 % of the median depth. The stage refuses a clip that fails the
check, because a wrong pose convention places every later point cloud
wrong without any error.

The manifest's frames are matched to the images by `seq_idx`. The stage
refuses a clip whose manifest lists other frames than `input_images/`.

Pi3X runs on a CPU only in principle; a clip takes too long to be useful.
