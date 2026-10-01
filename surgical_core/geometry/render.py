"""The images the segmenter is prompted with.

A segmenter input is a base image (RGB, colormapped depth, normals, Retinex
reflectance or flat grey) with the geometry added in one of two ways: the
geometric edges burnt in as dark lines (`burn_geom_edge`), or the surface
tilt applied as shading (`relight_rgb`). `sam_input_image` builds any mode in
`SAM_INPUT_MODES`.

Needs scipy (the Poisson integration in Retinex) and matplotlib (the depth
colormap): the `render` extra. The evaluator never imports this module.
"""

import cv2
import matplotlib
import numpy as np
from scipy.fft import dctn, idctn

from surgical_core.geometry.normals import (
    burn_geom_edge, camera_normals, geom_edge_map, normal_edge_map, normal_map)


def global_depth01(depth_all, lo_p=1, hi_p=99):
    """Normalise a clip's depth to [0, 1] with one scale for every frame, so
    that the colormapped depth is stable over time."""
    # Invalid pixels (depth 0) map below 0 and clip to 0, the colormap's
    # nearest colour, so in the segmenter input they look like a near valid
    # surface (NaN goes to black). The valid mask removes them downstream, so
    # the harm is small; blacken them here if that changes.
    finite = depth_all[np.isfinite(depth_all) & (depth_all > 1e-6)]
    lo, hi = np.percentile(finite, lo_p), np.percentile(finite, hi_p)
    return np.clip((depth_all - lo) / (hi - lo + 1e-6), 0, 1)


def depth_to_colormapped(gray01, cmap="viridis"):
    """Normalised depth (H, W) to RGB uint8; the segmenter takes 3 channels."""
    colored = matplotlib.colormaps[cmap](gray01)
    return (colored[:, :, :3] * 255).astype(np.uint8)


def pseudo_normal_from_rgb(rgb, scale=8.0, smooth=True):
    """A normal map faked from brightness, as graphics does bump-to-normal.

    It paints texture gradients as surface orientation, so physically it is
    a lie. It is a control: if it helps the segmenter as much as real normals
    do, what helps is the look of a normal map, not the geometry.

    Args:
        rgb: (H, W, 3) uint8.
        scale: strength of the height field; larger tilts more.
        smooth: bilateral-filter the height field, so pixel noise does not
            become tilt.

    Returns:
        (H, W, 3) uint8, encoded like `normal_map`.
    """
    h = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if smooth:
        h = cv2.bilateralFilter(h, d=5, sigmaColor=0.08, sigmaSpace=5)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    n = np.stack([-gx * scale, -gy * scale, np.ones_like(h)], -1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    return np.clip((n * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)


def relight_rgb(rgb, depth, K, light=(0.45, -0.45, 0.77), ambient=0.45):
    """Keep the colours and re-shade the image from the geometry.

    `n . l` for an oblique light, then `ambient + (1 - ambient) * n . l`
    multiplied into the RGB. The same factor on all three channels leaves hue
    and saturation untouched; only the shading changes. Where `burn_geom_edge`
    adds boundaries discretely, this adds surface tilt continuously, and the
    result still looks like a photograph, which matters because the
    segmenter was trained on photographs and degrades on inputs that are not.

    Args:
        rgb: (H, W, 3) uint8.
        depth: (H, W) depth.
        K: (3, 3) intrinsics.
        light: light direction in camera space. The default is above and in
            front, oblique; a frontal light shows no creases.
        ambient: floor of the shading, 0 to 1. 1 is no effect, 0 black shadows.

    Returns:
        (H, W, 3) uint8. Invalid pixels keep their RGB, so the effect is only
        the shading.
    """
    n, m = camera_normals(depth, K)
    l = np.asarray(light, np.float32)
    l /= np.linalg.norm(l)
    lam = np.clip(np.sum(np.nan_to_num(n) * l, -1), 0, 1)
    gain = np.where(m, ambient + (1.0 - ambient) * lam, 1.0)
    return np.clip(rgb.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def _poisson_dct(gx, gy):
    """Integrate a gradient field into a scalar field (DCT, Neumann boundary).

    The divergence keeps its boundary term. Writing only
    `fx[:, 1:] = gx[:, 1:] - gx[:, :-1]` leaves `fx[:, 0]` at zero, which
    drops the left and top gradients from the divergence; measured on a known
    field, the reconstruction then came out uniformly 1.47 times too large
    (correlation 0.906, slope 1.47), and the Retinex built on it amplified
    the shading instead of removing it (shading ratio of the reflectance 1.30
    to 1.52 on synthetic data). With the boundary term: correlation 1.000,
    slope 1.000, shading ratio 1.004.

    Args:
        gx, gy: (H, W) forward-difference gradients, last column and row 0.

    Returns:
        (H, W) scalar field, with the DC component fixed at 0: the constant
        is undetermined.
    """
    fx = np.zeros_like(gx)
    fy = np.zeros_like(gy)
    fx[:, 0] = gx[:, 0]
    fx[:, 1:] = gx[:, 1:] - gx[:, :-1]
    fy[0, :] = gy[0, :]
    fy[1:, :] = gy[1:, :] - gy[:-1, :]
    F = dctn(fx + fy, norm="ortho")
    h, w = F.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    den = 2 * np.cos(np.pi * x / w) + 2 * np.cos(np.pi * y / h) - 4
    den[0, 0] = 1.0
    F = F / den
    F[0, 0] = 0.0
    return idctn(F, norm="ortho")


def color_retinex(rgb, t_lum=0.3, t_chrom=0.05):
    """Colour Retinex: an intrinsic-image split into reflectance and shading.

    Of the log-brightness gradients, those at a large step or where the
    chromaticity moves are taken as material boundaries and removed from the
    shading's gradient; what remains is integrated into the shading. In
    endoscopy the light sits at the camera, so the shading is a proxy for
    shape, mixed with distance falloff.

    Args:
        rgb: (H, W, 3) uint8.
        t_lum: forward difference of log brightness above which an edge is a
            material boundary.
        t_chrom: the same threshold for chromaticity (colour normalised by
            brightness).

    Returns:
        `(reflectance, shading)`: (H, W, 3) float, not normalised to [0, 1],
        and (H, W) float in (0, 1].
    """
    img = rgb.astype(np.float64) / 255.0 + 1e-4
    log_l = np.log(img.mean(2))
    chrom = img / img.sum(2, keepdims=True)
    gx = np.zeros_like(log_l)
    gy = np.zeros_like(log_l)
    gx[:, :-1] = log_l[:, 1:] - log_l[:, :-1]
    gy[:-1, :] = log_l[1:, :] - log_l[:-1, :]
    cx = np.zeros_like(log_l)
    cy = np.zeros_like(log_l)
    cx[:, :-1] = np.abs(chrom[:, 1:] - chrom[:, :-1]).sum(2)
    cy[:-1, :] = np.abs(chrom[1:, :] - chrom[:-1, :]).sum(2)
    sx = np.where((np.abs(gx) > t_lum) | (cx > t_chrom), 0.0, gx)
    sy = np.where((np.abs(gy) > t_lum) | (cy > t_chrom), 0.0, gy)
    log_s = _poisson_dct(sx, sy)
    shading = np.exp(log_s - log_s.max())
    return img / (shading[..., None] + 1e-6), shading


def _to_u8(x, p=99.0):
    """A real-valued image to uint8, scaled by its `p`-th percentile so that
    a few bright pixels do not saturate the rest."""
    hi = np.percentile(x, p)
    return np.clip(x / (hi + 1e-8) * 255.0, 0, 255).astype(np.uint8)


def _compose(base, depth, K, edge=None, shade=False, edge_gain=1.0):
    """Apply geometric shading, then geometric dark lines, to a base image;
    either can be left out.

    Only multiplications, so hue and saturation are kept. That property was
    common to every composition that worked, and is made the rule here.

    Args:
        base: (H, W, 3) uint8.
        depth: (H, W) depth.
        K: (3, 3) intrinsics.
        edge: `None`, `"both"`, `"normal"` or `"depth"`: which edges to burn.
        shade: apply `relight_rgb`.
        edge_gain: how dark the edge line is.

    Returns:
        (H, W, 3) uint8.
    """
    img = base
    if shade:
        img = relight_rgb(img, depth, K)
    if edge is not None:
        e = geom_edge_map(depth, K, parts=edge)
        img = np.clip(img.astype(np.float32) * (1.0 - edge_gain * e)[..., None], 0, 255) \
                .astype(np.uint8)
    return img


# The input modes made by composition: base image x how the geometry is
# added. The older names below keep their own code paths; only combinations
# are added here. `base` is how the base is made, `edge` which dark lines,
# `shade` whether to relight.
_COMBOS = {
    # RGB as base
    "rgb_edge_shade":  ("rgb",  "both",   True),
    "rgb_nedge":       ("rgb",  "normal", False),
    "rgb_dedge":       ("rgb",  "depth",  False),
    # Retinex reflectance as base
    "refl_edge":       ("refl", "both",   False),
    "refl_edge_shade": ("refl", "both",   True),
    # geometry as base
    "normal_shade":    ("normal", None,   True),
    "depth_shade":     ("depth",  None,   True),
    # no base: the control
    "shade_only":      ("gray", None,     True),
}


def _base_image(kind, depth, K, gray01, rgb):
    if kind == "rgb":
        return rgb
    if kind == "refl":
        return _to_u8(color_retinex(rgb)[0])
    if kind == "normal":
        return normal_map(depth, K)
    if kind == "depth":
        return depth_to_colormapped(gray01)
    return np.full((*depth.shape, 3), 200, np.uint8)          # gray


# The modes `sam_input_image` accepts, in one place. Pass this to argparse as
# the choices: when callers each kept their own literal list, one of them was
# extended and the other was not, and a whole propagation stage failed on
# every clip after the per-frame stage had passed.
SAM_INPUT_MODES = ("depth", "normal", "normal_edge", "rgb",
                   "rgb_edge", "depth_edge", "edge_only",
                   "rgb_normal", "rgb_shade",
                   "rgb_refl", "rgb_shading", "refl_shade",
                   *sorted(_COMBOS))

# The inputs the tracker can propagate on: a subset of `SAM_INPUT_MODES`. Kept
# here, not in each caller, for the same reason as above: the propagation
# side and the side that composes conditions once held separate lists, one
# gained a mode and the run died in argparse before touching the GPU.
TRACK_BASE_MODES = ("rgb", "depth", "normal", "normal_edge", "rgb_edge")
assert set(TRACK_BASE_MODES) <= set(SAM_INPUT_MODES), \
    "TRACK_BASE_MODES names an input that is not in SAM_INPUT_MODES"


def uses_geom_edge(mode):
    """Whether `sam_input_image(mode, ...)` burns geometric edges in.

    For the provenance record: `edge_ring_masked` describes how the edges
    were made, so writing True or False for an input that burns no edges
    (`rgb`, `normal`, `depth`) would read as "made with that setting". For
    those inputs the record writes `None`, and this is how it tells.

    Args:
        mode: one of `SAM_INPUT_MODES`.
    """
    if mode in _COMBOS:
        return bool(_COMBOS[mode][1])
    return mode in ("normal_edge", "rgb_edge", "depth_edge", "edge_only")


def sam_input_image(mode, depth, K, gray01, rgb, edge_gain=0.85, smooth=True):
    """The 3-channel image the segmenter is prompted with, per input mode.

    Args:
        mode: `"depth"` (colormapped depth), `"normal"`, `"normal_edge"`
            (normals with the geometric edges burnt in), `"rgb"`, `"rgb_edge"`
            and `"depth_edge"` (the same edges burnt into RGB or depth),
            `"edge_only"` (flat grey with only the edges: the control for the
            base's contribution), `"rgb_normal"` (the normal map faked from
            brightness: the control that uses no geometry), `"rgb_shade"`
            (RGB relit from the geometry), `"rgb_refl"` and `"rgb_shading"`
            (the Retinex reflectance and shading), `"refl_shade"`
            (reflectance relit from the geometry), or a name in `_COMBOS`.
        depth: (H, W) depth.
        K: (3, 3) intrinsics.
        gray01: (H, W) depth normalised by `global_depth01`, for the depth modes.
        rgb: (H, W, 3) uint8.
        edge_gain: how dark the edge lines are, 0 to 1.
        smooth: bilateral-filter the normal image in `normal_edge`.

    Returns:
        (H, W, 3) uint8.
    """
    if mode in _COMBOS:
        kind, edge, shade = _COMBOS[mode]
        return _compose(_base_image(kind, depth, K, gray01, rgb), depth, K,
                        edge=edge, shade=shade, edge_gain=edge_gain)
    if mode == "normal":
        return normal_map(depth, K)
    if mode == "normal_edge":
        return normal_edge_map(depth, K, edge_gain=edge_gain, smooth=smooth)
    if mode == "rgb":
        return rgb
    if mode == "rgb_edge":
        return burn_geom_edge(rgb, depth, K, edge_gain=edge_gain)
    if mode == "depth_edge":
        return burn_geom_edge(depth_to_colormapped(gray01), depth, K, edge_gain=edge_gain)
    if mode == "rgb_refl":
        return _to_u8(color_retinex(rgb)[0])
    if mode == "rgb_shading":
        sh = color_retinex(rgb)[1]
        return np.repeat(_to_u8(sh)[..., None], 3, axis=2)
    if mode == "refl_shade":
        refl = _to_u8(color_retinex(rgb)[0])
        return relight_rgb(refl, depth, K)
    if mode == "rgb_normal":
        return pseudo_normal_from_rgb(rgb)
    if mode == "rgb_shade":
        return relight_rgb(rgb, depth, K)
    if mode == "edge_only":
        base = np.full((*depth.shape, 3), 200, np.uint8)
        return burn_geom_edge(base, depth, K, edge_gain=edge_gain)
    # TODO: an unknown mode lands here and comes back as colormapped depth,
    # so a misspelt mode in a config would run the depth condition under
    # another name. Callers are guarded by `SAM_INPUT_MODES` as argparse
    # choices; this function should refuse too (`"depth"` matched by name,
    # anything else a ValueError). A behaviour change, so not in the port.
    return depth_to_colormapped(gray01)             # "depth", the default
