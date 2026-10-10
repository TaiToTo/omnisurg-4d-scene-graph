# The viewer

A web page that shows a surgical clip's 4D scene graph: each frame's point
cloud, the regions of each track on it, the scene graph of the frame, and the
graph through time. The page is static. It reads the files that
`python -m pipeline.viewer_bundle` writes (`docs/pipeline.md`, "The viewer
bundle stage") and needs no server of its own.

What a reader can do on it:

- step through a clip's frames, or play them;
- paint a track's regions on the cloud, and draw the scene graph on it;
- see the frame's scene graph from its camera;
- stack the clip's frames in one space (world mode), with the camera's path,
  and see every stacked frame's scene graph in one camera;
- follow a region in three steps: where it is, how it moves through the
  stack, and how its relation to the neighbour it changes against most varies
  through time, on the three axes the relation was chosen from;
- hide the instruments, or show them alone, with the points that hang between
  surfaces at a depth step.

## Run it

Node 22 and npm:

```bash
cd viewer
npm ci
VIEWER_DATA=/path/to/site/data npm run dev    # http://localhost:5180
npm test
```

`VIEWER_DATA` is the directory the bundle stage wrote, the one with
`catalog.json`. The development server serves it at `./data/`, read-only.

## Publish it

```bash
npm run build                                  # writes dist/
VIEWER_DATA=/path/to/site/data npm run preview # the built page, at http://localhost:5181
```

`dist/` is the page. Copy it to any static host, and the bundle to `data/`
beside its `index.html`. A bundle hosted elsewhere is named when the page is
built, `VITE_DATA_URL=https://example.org/bundle/ npm run build`; that host
must then allow the page's origin to fetch from it (CORS). Every URL the page
uses is relative, so it works from any directory of a host.

The address keeps what is on screen, `?clip=<id>&frame=<n>&mode=world&focus=<track>:<id>`,
so a view can be shared as a link. The page opens only clips the catalog
lists.
