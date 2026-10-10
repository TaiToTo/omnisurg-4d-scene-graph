// Choose which frames World mode stacks, and snap a frame to the nearest of them.

/** How many clouds World mode stacks unless the page is told otherwise. */
export const TARGET_FRAMES = 12;

/** The most clouds World mode stacks: every stacked cloud is held in memory at once. */
export const MAX_WORLD_FRAMES = 60;

/**
 * Return the frames World mode stacks: 0, every `stride`-th frame, and the last.
 *
 * @param {number} maxFrame  the clip's last frame index.
 * @param {?number} target   about how many frames to stack.
 * @returns {number[]} ascending frame indices.
 */
export function worldIndices(maxFrame, target = TARGET_FRAMES) {
  const stride = Math.max(1, Math.round((maxFrame + 1) / (target ?? TARGET_FRAMES)));
  const out = [0];
  for (let i = stride; i <= maxFrame; i += stride) out.push(i);
  if (out[out.length - 1] !== maxFrame) out.push(maxFrame);
  return out;
}

/** Return the frame of `indices` nearest to `frameIdx`, the earlier one on a tie. */
export function nearestStacked(frameIdx, indices) {
  let best = indices[0], bd = Infinity;
  for (const i of indices) {
    const d = Math.abs(i - frameIdx);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
