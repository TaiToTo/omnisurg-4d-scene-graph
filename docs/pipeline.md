# The pipeline

The pipeline turns a clip of surgical video into the geometry that a 4D
scene graph is built on. It runs in stages. The depth stage estimates depth
and camera poses for every frame. Later stages segment the frames, track the
segments through time, and export what the viewer reads. Each stage is a
module of `pipeline` and runs as `python -m pipeline.<stage>`. Each stage
reads what the earlier stages wrote into the clip.

## A clip

A clip is a directory with these files:

- `input_images/`: the frames as PNG files. Their names sort in time order.
- `frame_manifest.json`: what the stage that cut the clip from its dataset
  recorded about it. Its `dataset` key names the dataset; a manifest
  without one belongs to CholecSeg8k.
- `crop_info.json`, for a CholecSeg8k clip only: the rectangle inside the
  endoscope's view that the frames were cut to.

## The depth stage

```bash
python -m pipeline.depth --input-dir /path/to/clips [--clips <clip> ...] [--gpu N] [--overwrite] [--no-glb]
```

The stage runs Depth Anything 3 on all frames of a clip in one call. It
first resizes each frame to 504 pixels on its longest side (`--process-res`).
It writes into the clip:

- `depth_raw/depth_NNNNNN.npy`: one depth map per frame.
- `depth_vis/NNNN.jpg`: each depth map as an image, near warm and far cool.
- `exports/mini_npz/results.npz`: what the later stages read: `depth`
  (N, H, W), `conf` (N, H, W), `extrinsics` (N, 3, 4) from world to camera
  and `intrinsics` (N, 3, 3), all float32.
- `pc_vis/frame_NNNN.glb`: one point cloud per frame, for the viewer.
  `--no-glb` skips them.
- `depth_info` in the manifest: the model, the resolution and the ranges.

The stage refuses a CholecSeg8k clip without `crop_info.json`, since the
model would see the black border. It refuses a clip that already holds its
output; `--overwrite` replaces the stage's own files and leaves every other
file. `python -m pipeline.depth --help` lists the options.

### Without a CUDA GPU

The stage runs on a CPU, slowly: four frames take about a minute and 6 GB
on a laptop. DA3 depends on `xformers`, which installs only next to CUDA, so
install DA3 without its dependencies and add the ones it imports:

```bash
pip install -e ".[render]" torch torchvision
pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
pip install "numpy<2" addict einops evo huggingface_hub imageio moviepy==1.0.3 omegaconf plyfile pycolmap safetensors
```
