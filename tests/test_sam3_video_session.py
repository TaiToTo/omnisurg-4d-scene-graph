"""Check how `sam3_wrapper.Sam3VideoInstanceSession` reads presence logits and starts a video, through stand-ins.

The stand-in session stores outputs as transformers' video session does: a
frame the session holds no output for gives None. The stand-in processor
records the arguments a video is started with.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from sam3_wrapper import Sam3VideoInstanceSession


class StandInStore:
    """Hold non-conditioning and conditioning outputs per (object, frame), returning None for a missing frame."""

    def __init__(self, non_cond, cond):
        self.outputs = {False: non_cond, True: cond}

    def get_output(self, obj_idx, frame_idx, output_key, is_conditioning_frame=True):
        out = self.outputs[is_conditioning_frame].get((obj_idx, frame_idx))
        return None if out is None else out[output_key]


def _session(non_cond, cond):
    s = Sam3VideoInstanceSession.__new__(Sam3VideoInstanceSession)
    s._session = StandInStore(non_cond, cond)
    return s


def test_a_propagated_frame_gives_its_non_conditioning_logit():
    s = _session({(0, 3): {"object_score_logits": 2.0}}, {(0, 3): {"object_score_logits": -5.0}})
    assert s._presence_logit(0, 3) == 2.0


def test_the_prompted_frame_gives_no_logit_though_it_holds_a_conditioning_one():
    s = _session({}, {(0, 2): {"object_score_logits": 2.0}})
    assert s._presence_logit(0, 2) is None


def test_an_output_without_a_logit_is_refused():
    s = _session({(0, 3): {"pred_masks": 0}}, {(0, 3): {"object_score_logits": 2.0}})
    with pytest.raises(KeyError):
        s._presence_logit(0, 3)


class StandInProcessor:
    """Record the arguments a video is started with, and start a session of a 4 x 6 video."""

    def init_video_session(self, video, inference_device, dtype, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(video_height=4, video_width=6)


def _starting(state_on_cpu):
    s = Sam3VideoInstanceSession.__new__(Sam3VideoInstanceSession)
    # A stand-in for torch, of which init_video reads only the dtype, so that these tests run without torch.
    s._torch = SimpleNamespace(float32="float32")
    s.device, s.processor, s.state_on_cpu = "cuda", StandInProcessor(), state_on_cpu
    s.init_video(np.zeros((2, 4, 6, 3), np.uint8))
    return s


def test_state_on_cpu_asks_the_processor_to_keep_frames_and_masks_in_host_memory():
    s = _starting(True)
    assert s.processor.kwargs == {"inference_state_device": "cpu", "video_storage_device": "cpu"}
    assert s._size == (4, 6)


def test_without_state_on_cpu_the_processor_is_given_no_storage_device():
    assert _starting(False).processor.kwargs == {}
