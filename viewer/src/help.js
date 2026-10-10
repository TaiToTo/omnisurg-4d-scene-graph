// List the page's mouse and keyboard bindings, and show them with a short
// account of what the page draws.
//
// The help panel is built from `KEY_BINDINGS`, the one list of bindings, so
// it cannot drift from what the keys do.
import { esc } from './lib/format.js';

export const KEY_BINDINGS = [
  { group: 'Mouse' },
  { keys: 'drag', what: 'Rotate the 3D view' },
  { keys: 'right-drag', what: 'Pan' },
  { keys: 'wheel', what: 'Zoom' },
  { keys: 'click', what: 'Follow a region: on the cloud, in the scene graph or in the band below' },
  { group: 'Keys' },
  { keys: '← →', what: 'Previous / next frame' },
  { keys: 'Space', what: 'Play / pause' },
  { keys: 'W', what: 'World mode: stack the clip’s frames in one space' },
  { keys: 'G', what: 'Draw the scene graph on the cloud' },
  { keys: 'S', what: 'Paint the regions on the cloud, or take them off' },
  { keys: 'O', what: 'World mode: show only the followed region' },
  { keys: 'P', what: 'World mode: the camera frames under the band’s regions' },
  { keys: 'R', what: 'Reset the 3D camera' },
  { keys: 'Esc', what: 'Stop following the region, or close this panel' },
  { keys: '?', what: 'Show this panel' },
];

const ABOUT = `Each clip is a short stretch of laparoscopic video. A monocular depth
model turns every frame into a point cloud; a promptable segmentation model
splits the frames into regions and tracks them through time, with no training
for the task. The scene graph has one node per region, placed at the region's
3D centroid, and one edge per pair of regions, named by where one region lies
from the other in the camera's view. World mode stacks the frames in one space,
so a node's path and its edges can be followed through time.`;

/** Fill the help panel. */
export function renderHelp(el) {
  const rows = KEY_BINDINGS.map((b) => (b.group
    ? `<h3>${esc(b.group)}</h3>`
    : `<div class="key-row"><kbd>${esc(b.keys)}</kbd><span>${esc(b.what)}</span></div>`));
  el.innerHTML = `<p>${esc(ABOUT.replace(/\s+/g, ' '))}</p>${rows.join('')}`;
}
