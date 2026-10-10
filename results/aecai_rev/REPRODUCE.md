# REPRODUCE

Commands that rebuild every table and figure under `results/aecai_rev/`,
one per line, from the repository root. Expected run times are for one
machine with 8 GPUs and 48 cores.

## Setup

```bash
# 1. The AE-CAI code, as tagged in the workbench (never modified).
mkdir -p $AECAI_REV_WORK/aecai_src
git -C $OMNISURG_SOURCE archive paper/aecai2026-endolina-final -- depth_sam_tracking_experiment surgical_core sam3_wrapper scripts da3_surgery_wrapper pyproject.toml | tar -x -C $AECAI_REV_WORK/aecai_src
git -C $OMNISURG_SOURCE rev-parse 'paper/aecai2026-endolina-final^{commit}' > $AECAI_REV_WORK/aecai_src/TAG_COMMIT

# 2. Environment: paths are never committed; config.yaml reads them from here.
export OMNISURG_SOURCE=/path/to/workbench          # holds outputs/cholec_gt and the frozen tracks
export AECAI_REV_WORK=/path/to/work                 # tagged code, new tracks, scores
export CHOLECSEG8K_ROOT=/path/to/CholecSeg8k        # video01/video01_00080/...
export CHOLEC80_VIDEOS=/path/to/cholec80/videos     # video01.mp4 ...
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Python 3.12 with numpy, scipy, pandas, matplotlib, opencv, PyYAML; for the
tracking step also torch, transformers (SAM 3), `segment-anything` and the
SAM checkpoint at `$OMNISURG_SOURCE/outputs/weights/sam_vit_h_4b8939.pth`.
All conditions are in `aecai_rev/config.yaml`.

## Commands

| output | command |
|---|---|
| tests of the scoring against the tagged evaluator | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest aecai_rev/tests` |
| `00_windows/` | `python3 -m aecai_rev.windows --videos-root $CHOLEC80_VIDEOS` |
| tracks of pps 4, 6, 12 (≈ 25 min) | `OMP_NUM_THREADS=4 python3 -m aecai_rev.run_track --pps 4 6 12 --gpus 1,2,3,4,5,6,7 --per_gpu 2` |
| seed maps of the frozen pps 8, 16, 24, 32 | `OMP_NUM_THREADS=4 python3 -m aecai_rev.run_track --pps 8 16 24 32 --seeds --gpus 1,2,3,4,5,6,7 --per_gpu 1` |
| frame and window scores (`$AECAI_REV_WORK/scores/`, ≈ 1 min) | `python3 -m aecai_rev.score --workers 27` |
| `00_check/` (Table 1 reproduced) | `python3 -m aecai_rev.check_legacy --workers 24` |
| `07_filter/` | `python3 -m aecai_rev.filter_sensitivity` |
| `01_sweep/` | `python3 -m aecai_rev.sweep` |
| `03_merge/` | `python3 -m aecai_rev.merge_rescore` |
| `04_identity/` | `python3 -m aecai_rev.identity --workers 14` |
| `05_merge_rate/`, `06_naming/` (rates) | `python3 -m aecai_rev.overlay --workers 14` |
| `06_naming/fig6_named_events.txt` | `python3 -m aecai_rev.naming_events --clip VID12_s15_19900_crop --focus 6` |
| `SUMMARY.md` | `python3 -m aecai_rev.summary` |

Item 2 (statistics) has no command of its own: every `summary.csv` and
`tests.csv` above goes through `aecai_rev/stats.py`.

## Where the inputs come from

- Frames, masks and depth of the 27 windows:
  `$OMNISURG_SOURCE/outputs/cholec_gt/<window>/`, as the AE-CAI run left them.
- Tracks of pps 8, 16, 24, 32, 48: the AE-CAI run's
  `$OMNISURG_SOURCE/depth_sam_tracking_experiment/out_w1_{pps8,pps16,all,pps32,pps48}/`,
  read in place. Tracks of pps 4, 6, 12: `$AECAI_REV_WORK/tracks/pps<N>/`,
  made by `run_track` with the tagged `track_sam3.py`; each directory's
  `meta.json` records the environment, the checkpoint and its sha256.
- Every output directory's `meta.json` records the configuration, this
  repository's commit, the tag's commit and the library versions.
