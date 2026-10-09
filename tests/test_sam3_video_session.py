"""Check how `sam3_wrapper.Sam3VideoInstanceSession` reads presence logits, through a stand-in session.

The stand-in stores outputs as transformers' video session does: a frame the
session holds no output for gives None.
"""

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
