"""Check that every segmenter input renders the same bytes as when the shas were recorded.

A change to how the inputs are composed that moves one pixel of one mode
changes a measured condition, and nothing else would notice. The shas in
`render_outputs.sha256.json` were recorded from the code on `main` before the
composition was refactored. Each mode runs at both edge gains the conditions
used, with and without smoothing, on one synthetic frame. `rgb_normal` is
pinned per CPU architecture, as `platform.machine()` names it: OpenCV's
bilateral filter on float32 gives other last bits on ARM and on x86.

Usage:
    python -m tests.test_render_outputs_are_pinned > tests/render_outputs.sha256.json
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

from surgical_core.geometry import render  # noqa: E402

PINNED = Path(__file__).with_name("render_outputs.sha256.json")


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
            for mode in render.SAM_INPUT_MODES for gain in (0.85, 1.0) for smooth in (True, False)]


def sha(img):
    """Return the image's dtype, shape and the sha256 of its bytes, as one string."""
    return f"{img.dtype}{list(img.shape)} {hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()}"


def render_case(kwargs):
    depth, K, gray01, rgb = inputs()
    return render.sam_input_image(kwargs["mode"], depth, K, gray01, rgb, edge_gain=kwargs["edge_gain"],
                                  smooth=kwargs["smooth"])


def record():
    """Return the shas this machine renders. A case pinned per architecture keeps the other architectures' shas."""
    old = json.loads(PINNED.read_text()) if PINNED.is_file() else {}
    new = {}
    for key, kwargs in cases():
        got = sha(render_case(kwargs))
        new[key] = {**old[key], platform.machine(): got} if isinstance(old.get(key), dict) else got
    return new


def test_every_case_is_pinned_and_nothing_else():
    assert sorted(json.loads(PINNED.read_text())) == sorted(key for key, _ in cases())


@pytest.mark.parametrize("key, kwargs", cases(), ids=[key for key, _ in cases()])
def test_the_mode_renders_the_pinned_bytes(key, kwargs):
    pinned = json.loads(PINNED.read_text())[key]
    if isinstance(pinned, dict):
        assert platform.machine() in pinned, f"{key} has no sha for {platform.machine()}, only for {sorted(pinned)}"
        pinned = pinned[platform.machine()]
    got = sha(render_case(kwargs))
    assert got == pinned, f"{key} rendered {got}"


if __name__ == "__main__":
    print(json.dumps(record(), indent=2, sort_keys=True))
