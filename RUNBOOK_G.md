# Running `integration/gpu` on the GPU machine

This branch is `main` with every open pull request merged into it, before
review, so that the work on the GPU machine can start now. It is rebuilt
when a pull request changes, and it is never merged. Every result below
records the commit it ran on. A later review changes a result only if it
changes the code that made it; the scores are cheap to make again from the
predictions, which are what costs GPU time.

`<W>` is the workbench's clone on this machine, `<clone>` this repository's.

## 0. Set up, and record the versions

```bash
git clone https://github.com/TaiToTo/omnisurg-4d-scene-graph <clone> && cd <clone>
git checkout integration/gpu && git rev-parse HEAD   # write this commit down with every result
```

Run the stages and the evaluator with the Python the workbench runs on, so
that both sides of every check share one environment. The repository is
found through `PYTHONPATH`; nothing is installed into that environment.

```bash
export PY=<the workbench's python>
PYTHONPATH=<clone> $PY -m pytest -q <clone>/tests    # every test must pass
$PY -m pip freeze > g_packages.txt                   # answers "G's constraints file"
$PY -m pip show -f depth-anything-3 pi3 | grep -i -E "^(name|version|location)"
cat $($PY -c "import depth_anything_3, os; print(os.path.dirname(depth_anything_3.__file__))")/../depth_anything_3-*.dist-info/direct_url.json
```

The last line gives the commit of Depth Anything 3 the workbench ran
("Which commit of Depth Anything 3 the `recon3d` extra pins"); do the same
for `pi3`.

## 1. Before anything writes: read the depth markers

On the clips the paper reads, the depth may come from two versions of the
stage ("Depth made by two versions of the depth stage"). Record which before
any stage runs:

```bash
$PY - <<'EOF'
import json, numpy as np, pathlib
for root in ["<W>/outputs/cholec_gt", "<W>/outputs/atlas97"]:
    for clip in sorted(pathlib.Path(root).iterdir()):
        npz = clip / "exports/mini_npz/results.npz"
        man = clip / "frame_manifest.json"
        if not npz.is_file() or not man.is_file():
            continue
        keys = list(np.load(npz).files)
        info = json.loads(man.read_text()).get("depth_info", {})
        print(clip.name, "ray_map" in keys, info.get("backproject_mode"), info.get("ray_map_available"))
EOF
```

Never pass `--overwrite` to `pipeline.depth` on these clips: it replaces the
depth these markers describe.

## 2. The evaluator against the pilot evaluator (step 2)

For each of the 38 scored conditions, by its tag:

```bash
PYTHONPATH=<clone> $PY -m evalkit.evaluate --pilot --dataset <cholecseg8k|atlas120k> \
    --clips <population file> --data-root <dir of clips> --tracks-root <dir of predictions> \
    --tag <tag> --out pilot_mode/<tag>.json
PYTHONPATH=<clone> $PY -m evalkit.tools.pilot_check --pilot-dir <the pilot's score JSONs> --eval-dir pilot_mode
```

It passes when `pilot_check` finds every shared key and every frame count
equal on every clip, with no tolerance. Report a difference as it is; do not
adjust either side.

## 3. Score every condition (step 3)

```bash
PYTHONPATH=<clone> $PY -m evalkit.evaluate --dataset <dataset> --clips <population file> \
    --data-root <dir of clips> --tracks-root <dir of predictions> --tag <tag> --out scores/<tag>.json
PYTHONPATH=<clone> $PY -m evalkit.tools.condition_inventory --root '<dir of predictions>:scores:<title>' --require-scored
```

The rule each condition was propagated under comes from its
`seed_info.json`. A condition seeded from GT holds neither rule and is
refused: list those, they wait on "Conditions seeded from GT". Labels with
no `seed_info.json` need `--propagation <rule>`.

## 4. The depth stages against the workbench's (step 5)

On `adrenalectomy__16GPCUPkXYQ__gt_0004`, `adrenalectomy__16GPCUPkXYQ__tile_0007`
(under `<W>/outputs/atlas`) and `VID01_s15_80_crop` (under
`<W>/outputs/cholec_gt`, with `--needs crop_info.json`):

```bash
PYTHONPATH=<clone> $PY -m pipeline.byte_check --root <W>/outputs/atlas --clip <clip> \
    --a-repo <W> --a-cmd "$PY scripts/run_cholec_depth.py --input-dir {root} --clips {clip}" \
    --a-env PYTHONPATH=<W>/da3_surgery_wrapper --a-watch surgical_core da3_surgery_wrapper \
    --b-repo <clone> --b-cmd "$PY -m pipeline.depth --input-dir {root} --clips {clip} --gpu 0" \
    --b-watch surgical_core pipeline recon3d_wrapper --json-out da3_<clip>.json
```

- Pass `--gpu 0` to the port: the workbench's stage always sets
  `CUDA_VISIBLE_DEVICES` to its own `--gpu`, 0 by default, and the port
  leaves it alone without one. Both must run on one card.
- Pi3X the same way: `pi3_wrapper/scripts/run_pi3_depth.py --input-dir {root}
  --videos {clip} --model pi3x` against `-m pipeline.pi3x --input-dir {root}
  --clips {clip} --gpu 0`, watching `surgical_core pi3_wrapper` and
  `surgical_core pipeline recon3d_wrapper`. Only `runtime_sec` may differ.
- Point clouds: `scripts/regen_cholec_glb.py --input-dir {root} --videos {clip}`
  against `-m pipeline.point_clouds --input-dir {root} --clips {clip}`, with
  `--needs exports/mini_npz/results.npz`.
- The tracking stage: `pipeline.byte_check` refuses this pair, because
  only the workbench's run imports `trimesh` and `networkx`. Run both into
  fresh roots and compare every file with `diff -r`; no output means equal.
  On `VID01_s15_80_crop` (and any clip with the bundles), at the operating
  point:

  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W>:<W>/sam3_wrapper $PY <W>/depth_sam_tracking_experiment/track_sam3.py \
      --root <W>/outputs/cholec_gt --clip <clip> --out_dir <out>/w --out_tag t --device auto \
      --sam_ckpt <sam_vit_h_4b8939.pth> --max_frames 0 --seed_auto --bidir --seed_inst_thresh 1.0 \
      --sam_input normal_edge --track_base rgb
  PYTHONPATH=<clone> $PY -m pipeline.track --input-dir <W>/outputs/cholec_gt --clips <clip> \
      --tracks-root <out>/p --tag t --gpu 0 --sam-ckpt <sam_vit_h_4b8939.pth> \
      --rule both_ways_from_centre --sam-input normal_edge --track-base rgb
  diff -r <out>/w <out>/p
  ```

  Neither writes into the clip. `forward_from_first` is the same pair
  without `--seed_auto --bidir --seed_inst_thresh 1.0`, and with
  `--rule forward_from_first`.
- The ATLAS-120k extraction (`-m pipeline.extract_atlas120k`) was compared
  with the workbench's on all 97 videos on the development machine, and
  every file was equal; it needs no run here. If it is run here, it reads
  the mp4s for their fps and size, and this machine's OpenCV cannot decode
  the AV1 one (`rarp/NitKIjCcS7U`): point `--atlas-root` at the root with
  that video converted.
- If this machine's packages no longer equal those of the determinism
  measurement, run the workbench's stage twice first: the same command
  without the `--b-*` options.

## 5. The 14 CholecSeg8k clips extracted again

In 14 of the 27 CholecSeg8k clips, the workbench's extractor decoded the
frames without a mask at the wrong moment ("CholecSeg8k clips whose frames
run out of order"). The port converts them, and these 14 are extracted
again and re-run for the paper. The other 13 extract byte for byte as the
workbench's, so their predictions stand.

First, check that the crop rectangles in `cholecseg8k_meta/crop_rects.json`
are the ones the paper's clips were cut with. They were computed again on
the development machine and equal its 9 clips; the 27 here decide. Every
line must say `True`. If any says `False`, put the values the paper's clip
holds into the table before cropping, and say so with the results.

```bash
$PY - <<'PYEOF'
import json, pathlib
table = json.load(open("<clone>/cholecseg8k_meta/crop_rects.json"))
for clip in sorted(table):
    p = pathlib.Path("<W>/outputs/cholec_gt") / f"{clip}_crop" / "crop_info.json"
    print(clip, json.load(open(p)) == table[clip] if p.is_file() else "missing")
PYEOF
```

Then extract and crop the 14 into a new root, never over `<W>/outputs/cholec_gt`:

```bash
CLIPS="VID17_s15_1563 VID18_s15_979 VID24_s15_0 VID24_s15_9596 VID25_s15_162 VID25_s15_402 VID26_s15_1855
       VID27_s15_160 VID27_s15_400 VID37_s15_528 VID43_s15_67 VID48_s15_561 VID52_s15_2746 VID55_s15_508"
PYTHONPATH=<clone> $PY -m pipeline.extract_cholecseg8k --seg8k-root <SourceDatasets>/CholecSeg8k/CholecSeg8k \
    --videos-root <SourceDatasets>/cholec80/cholec80/videos --out <new root> --clips $CLIPS
PYTHONPATH=<clone> $PY -m pipeline.crop_cholecseg8k --input-dir <new root> \
    --rects <clone>/cholecseg8k_meta/crop_rects.json --clips $CLIPS
```

(In bash, `$CLIPS` splits on the spaces; in zsh write `${=CLIPS}`.) Then
run the depth stage on the 14 `_crop` clips, with the version of the stage
that step 1's markers show for the clip it replaces. Then run every
condition the paper reports on CholecSeg8k, with the workbench's code as it
ran them, its input root set to `<new root>`. Score them as in step 3 and
record both commits.

## 6. New measurements

These are new runs, not checks. Score each as in step 3, and record the
commit. Write them to a tracks root of their own, never over the
workbench's predictions. `<tracks>` below is that root, and `<clips>` the
clips' root.

- **`forward_from_first`.** The tracking stage, with the operating point's
  inputs, seeded on frame 0 and carried forwards:

  ```bash
  PYTHONPATH=<clone> $PY -m pipeline.track --input-dir <clips> --tracks-root <tracks> --tag <tag> --gpu 0 \
      --sam-ckpt <sam_vit_h_4b8939.pth> --rule forward_from_first --sam-input <mode> --track-base rgb
  ```

  The labels land in `<tracks>/<clip>/track_rgb_<tag>/`, which the
  evaluator reads with `--tag track_rgb_<tag>`. Their `seed_info.json`
  gives the rule.
- **The merges to K = 10.** `evalkit.tools.kmerge` merges an existing
  condition's predictions and writes `<tag>_k10` beside them:

  ```bash
  PYTHONPATH=<clone> $PY -m evalkit.tools.kmerge --dataset <dataset> --clips <population file> \
      --data-root <clips> --tracks-root <tracks> --tag <tag> --k 10
  ```

- **The granularity result's propagated condition.** The per-frame
  `rgb` condition on a grid of 24, merged to K = 10, seeds the tracker on
  the middle frame:

  ```bash
  PYTHONPATH=<clone> $PY -m pipeline.per_frame --input-dir <clips> --tracks-root <tracks> --tag rgb_pf \
      --sam-input rgb --sam-ckpt <sam_vit_h_4b8939.pth> --gpu 0
  PYTHONPATH=<clone> $PY -m evalkit.tools.kmerge --dataset <dataset> --clips <population file> \
      --data-root <clips> --tracks-root <tracks> --tag track_rgb_rgb_pf --k 10
  PYTHONPATH=<clone> $PY -m pipeline.track --input-dir <clips> --tracks-root <tracks> --tag t5_k10 --gpu 0 \
      --rule both_ways_from_centre --sam-input rgb --track-base rgb \
      --seed-labels '<tracks>/{clip}/track_rgb_rgb_pf_k10'
  ```

  On one 6-frame clip, this chain equals the workbench's
  (`make_t5_seeds.py` and `track_sam3.py --seed_source external`), file
  for file.
- **The camera trajectory on StereoMIS** is not ported yet. Run it with
  the workbench's code (`run_20.sh`, `pose_metrics.py`).
