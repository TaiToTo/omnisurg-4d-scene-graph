# The viewer

The viewer is a web page that shows a clip's 4D scene graph:

- the point cloud of each frame, with its regions painted on it or not;
- the scene graph in the camera's view, one node per region at the
  region's place in the image, beside the camera frame and its masks;
- World mode, which stacks a sample of the clip's frames in one space with
  the camera's path, and projects every stacked node into one camera;
- the graph through time: each stacked frame's regions, the same node
  threaded from frame to frame;
- one region followed through time: where it is, how it moves in 3D, and
  how its relation to its neighbour changes, event by event;
- the scene graph drawn on the point cloud.

The page is static. It reads `catalog.json` and the clips' files from a
data directory, and runs no server code.

## Run it on your clips

Node 20 or newer is needed. From `viewer/`:

```bash
npm ci
python -m pipeline.viewer_catalog --root /path/to/outputs --clips <clip path> ... --in-place
VIEWER_DATA=/path/to/outputs npm run dev
```

The page opens at `http://localhost:5180/`. `docs/pipeline.md` describes
the catalog command.

## Publish it

```bash
python -m pipeline.viewer_catalog --root /path/to/outputs --clips <clip path> ... --out site-data
npm run build
```

`dist/` holds the page. Put the catalog directory beside it as `data/`, or
build with `VITE_DATA_ROOT=https://host/path/` to read the data from another
place, which must then allow the page's origin. A segmentation frame is a
JSON file of several megabytes; a host that compresses JSON sends about a
twentieth of it.

## What a clip holds

A clip is the directory a catalog path names. The viewer reads, for frame
`NNNN` of `n` and each track `<track>` of the catalog:

- `frame_manifest.json`: per frame, the cloud's centroid `glb_centroid` and
  the camera's `camera_pos_glb`, `camera_forward_glb` and `camera_up_glb`,
  and the same keys under `geometry_sources.<source>` for a depth model
  other than DA3;
- `input_images/NNNNNN.png`: the camera frame;
- `pc_vis/frame_NNNN.glb`, or `pc_vis/frame_NNNN__<source>.glb`: the point
  cloud, one vertex per pixel of the depth model's grid;
- `pc_vis/seg_frame_NNNN__<track>.json`: the frame's regions, as a colour
  per pixel and a region id per vertex of the DA3 cloud;
- `pc_vis/graph_frame_NNNN__<track>.json`: the frame's scene graph;
- `pc_vis/temporal_graph__<track>.json`: the track's nodes and relations
  through the clip;
- `pc_vis/hierarchy_frame_NNNN.json`, where two tracks describe one frame:
  which annotated region contains which automatic one.

## Tests

```bash
npm test
```

The tests cover the modules under `src/lib/`, which hold no DOM and no
WebGL: the catalog's checks, the region and outline fits, the relations a
node's band reads, the projection of a cloud's labels onto another grid,
and the escaping of text taken from the data.
