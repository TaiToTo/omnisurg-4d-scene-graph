"""Check that every segmenter input renders the same bytes as when the hashes were recorded.

A change that moves one pixel of one input mode changes a measured
condition, and nothing else would notice. The hashes in
`render_outputs.sha256.json` were recorded from the code on `main` before
the unused modes were removed. Each mode runs at both edge gains the
conditions used, with and without smoothing, on one synthetic frame.

Usage:
    python -m tests.test_render_outputs_are_pinned > tests/render_outputs.sha256.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from surgical_core.geometry import render  # noqa: E402

PINNED = Path(__file__).with_name("render_outputs.sha256.json")

# The modes the paper's conditions ran.
MODES = ("depth", "normal", "normal_edge", "rgb", "rgb_edge")


def inputs(h=48, w=64):
    """Return a frame's depth, intrinsics, normalised depth and RGB.

    The depth has a slope, a bump, a step, a hole of zeros and one NaN, so that the creases, the steps, the ring
    and the invalid pixels all reach the output.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    depth = 0.3 + 0.002 * xx + 0.001 * yy + 0.01 * np.sin(xx / 6.0) * np.cos(yy / 5.0)
    depth[12:20, 20:30] += 0.05
    depth[30:34, 40:46] = 0.0
    depth[5, 50] = np.nan
    K = np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1]])
    rng = np.random.default_rng(0)
    ramp = np.stack([xx * 3, yy * 4, np.full_like(xx, 128.0)], -1)
    rgb = np.clip(ramp + rng.normal(0, 20, (h, w, 3)), 0, 255).astype(np.uint8)
    return depth, K, render.global_depth01(depth), rgb


def cases():
    """Return every mode at each edge gain, with and without smoothing, as (key, keyword arguments)."""
    return [(f"{mode} edge_gain={gain} smooth={smooth}", {"mode": mode, "edge_gain": gain, "smooth": smooth})
            for mode in MODES for gain in (0.85, 1.0) for smooth in (True, False)]


def sha(img):
    """Return the image's dtype, shape and the SHA-256 hash of its bytes, as one string."""
    return f"{img.dtype}{list(img.shape)} {hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()}"


def render_case(kwargs):
    depth, K, gray01, rgb = inputs()
    return render.sam_input_image(kwargs["mode"], depth, K, gray01, rgb, edge_gain=kwargs["edge_gain"],
                                  smooth=kwargs["smooth"])


def test_every_case_is_pinned_and_nothing_else():
    assert sorted(json.loads(PINNED.read_text())) == sorted(key for key, _ in cases())


@pytest.mark.parametrize("key, kwargs", cases(), ids=[key for key, _ in cases()])
def test_the_mode_renders_the_pinned_bytes(key, kwargs):
    got = sha(render_case(kwargs))
    assert got == json.loads(PINNED.read_text())[key], f"{key} rendered {got}"


if __name__ == "__main__":
    print(json.dumps({key: sha(render_case(kwargs)) for key, kwargs in cases()}, indent=2, sort_keys=True))
