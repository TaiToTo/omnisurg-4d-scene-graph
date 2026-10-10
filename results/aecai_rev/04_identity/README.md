# 04 Identity consistency

Written by `python3 -m aecai_rev.identity`. Setting: SAM `points_per_side`
24, the three modalities, the AE-CAI tracks; area cut 0.001 of the valid
area on both sides.

## Definitions (for the manuscript)

Let $A$ be the annotated frames of a window, in temporal order, and $s$ its
seed frame. On each $t \in A$, the tracked regions $\mathcal{P}_t$ (each
carrying its tracker identity $i$) are matched to the frame's ground-truth
instances $\mathcal{G}_t$ (connected components of each non-background
class) by IoU $\ge 0.5$, greedily and one to one, with the matching used for
instance F1. We write $m_t(i)$ for the instance matched to identity $i$ on
frame $t$, when there is one.

**GT tracks.** Instances of consecutive annotated frames $t, t' \in A$ are
linked when they have the same class, greedily by decreasing IoU, one to
one, and IoU $> 0$; linked instances share a GT track $\tau(\cdot)$. An
instance with no link starts a new GT track.

- **ID switch rate.** Over all identities $i$ and consecutive annotated pairs
  $(t, t')$ with both $m_t(i)$ and $m_{t'}(i)$ defined,
  $$\mathrm{IDSW} = \frac{\#\{(i,t,t') : \tau(m_t(i)) \ne \tau(m_{t'}(i))\}}{\#\{(i,t,t')\}}.$$
  The **class switch rate** is the same ratio with the class of the matched
  instance in place of its GT track.
- **Fragmentation.** For each GT track $g$ matched at least once,
  $F(g) = |\{i : \tau(m_t(i)) = g \text{ for some } t\}|$, the number of
  distinct identities matched to it. We report the mean of $F$ and the share
  of GT tracks with $F \ge 2$.
- **Survival.** For each identity present on the seed frame, the share of
  frames after the seed (forward) and before it (backward) on which its mask
  is non-empty; averaged over identities.
- **Reappearance rate.** The share of identities whose mask, followed from
  the seed outward in either direction, becomes empty and later non-empty
  again. For each such event, **same GT** (same class) says whether the GT
  track (class) matched on the last annotated frame before the mask vanished
  equals the one matched on the first annotated frame after it returned.

Every metric is computed per window; tables give the mean over windows with
window-level bootstrap intervals, and paired tests between modalities.

## Conventions and their limits (for Limitations)

> Ground-truth instances carry no identity across frames in CholecSeg8k; we
> link them between consecutive annotated frames by class and overlap. This
> link is ambiguous where a structure splits or merges between frames (for
> example where an instrument crosses an organ): one of the parts then
> starts a new GT track, and a region that follows that part is counted as
> an ID switch although it stays on the same structure. The class switch
> rate, which compares classes instead of GT tracks, is not affected by this;
> it is 7 to 17 times smaller than the ID switch rate, depending on the
> modality.

- `link == consecutive_annotated` (the primary rule) links consecutive
  annotated frames however far apart; `link == adjacent_samples` links only
  annotated frames one sample apart, so a gap in annotation starts new GT
  tracks. ID switches are counted over the same pairs of frames each rule
  links.
- A mask is "empty" in the label map the tracker's output was collapsed to
  (overlaps resolved by score), so a region hidden under a higher-scoring one
  counts as absent.
- 11 of the 27 windows show the tracker frames out of temporal order between
  annotated frames (`00_inventory.md`, section 4); `subset ==
  frames_in_order` repeats every metric on the other 16.

## Files

- `long.csv`, `summary.csv`, `tests.csv`: by `subset`, `link`, `modality`.
- `id_switch_events.csv`: every ID switch (window, link, modality, region,
  frames, GT tracks, classes).
- `frames_in_order.csv`: which windows have their frames in temporal order.
- `examples/`: three ID switches under the primary rule, those that change
  class first; the region is filled, the matched class outlined.
