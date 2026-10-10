# 00 Windows

`windows.csv`, `excluded.csv` and `legacy27_map.csv` are written by
`python3 -m aecai_rev.windows --videos-root <cholec80 videos>`.

## The enumeration rule (for the manuscript)

> We define one evaluation window per CholecSeg8k clip (a run of 80
> consecutive annotated frames), with no selection by content. The seed is
> the clip's middle annotated frame (the earlier of the two middle frames).
> The window holds 30 frames, every 15th CholecSeg8k frame, with the seed as
> its 16th frame, so it extends 225 frames before the seed and 210 after it.
> A window is excluded when any of its frames falls before the first or after
> the last frame of the video, or when fewer than 5 of its frames carry
> annotation. Of the 101 clips of the 17 videos, 86 give a window; the 15
> excluded windows would start before the video does. Tracking runs on all
> 30 frames of a window, and every annotated frame in it is scored, including
> annotated frames of neighbouring clips.

Notes for the author:

- These windows are not extracted, depth-estimated or tracked yet. Their
  frames lie on a different phase of the stride from the AE-CAI windows, so
  none of the AE-CAI data can be reused; `legacy27_map.csv` maps each AE-CAI
  window to the clips it covers and the enumerated windows seeded in them.
- Windows of consecutive clips overlap (their seeds are 80 frames apart and a
  window spans 435), so neighbouring windows share scored frames.
- "Every 15th CholecSeg8k frame" is 0.6 s in the five videos CholecSeg8k
  numbers at 25 fps (01, 09, 12, 20, 35) and 0.5 s in the other twelve,
  which it numbers at about 30 fps (`00_inventory.md`, section 4).

## The rule of the AE-CAI 27 windows (recovered)

> In each of the 17 videos, a window starts at the first frame of a run of
> consecutive CholecSeg8k clips and holds 30 frames, every 15th CholecSeg8k
> frame, with the tracker seeded on the 16th. One window covers the clips
> that follow within its span (435 frames); clips farther away get a window
> of their own. This gives 27 windows, which touch 90 of the 101 clips and
> score 522 annotated frames.

Recovered from the tag's `miccai2026_workshop/docs/paper_clean_rerun_spec.md`
(section "クリップ選定ルール") and `window_definition_and_coverage_review.md`.
The second document lists where the placement departs from that rule. One
five-clip run of VID01 has no window. Two seven-clip runs (VID28, VID37)
spill over the window by one clip. One VID52 window starts one clip after
its run does. Some windows start at a clip inside a run another window
already covers (for example `VID01_s15_240` inside the run `VID01_s15_80`
starts), so "one window per run" does not describe the set exactly.
