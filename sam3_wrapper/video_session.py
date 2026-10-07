"""SAM 3's video tracker, seeded with one binary mask per object on one frame.

The tracker is `transformers`' `Sam3TrackerVideoModel`, the part of the
`facebook/sam3` checkpoint that takes point, box and mask prompts and keeps
a memory bank. A session loads it once, in float32, and runs any number of
videos one after another.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL_ID = "facebook/sam3"


@dataclass(frozen=True)
class FrameResult:
    """One frame of a propagation.

    Attributes:
        frame_idx: the frame's index in the video.
        object_ids: the ids of the objects in this frame; `masks[i]` is `object_ids[i]`'s.
        masks: (N, H, W) bool, at the video's resolution.
        scores: (N,) float32, the sigmoid of each object's presence logit; NaN where the session holds none.
    """

    frame_idx: int
    object_ids: list[int]
    masks: np.ndarray
    scores: np.ndarray


class Sam3VideoInstanceSession:
    """Load SAM 3's video tracker once onto a device.

    Args:
        model_id: the Hugging Face model id. The weights download on first use.
        device: "auto", "cpu" or "cuda". "auto" takes CUDA when there is one.
        gpu: the index of the GPU to use, set through `CUDA_VISIBLE_DEVICES`. None leaves the environment as it is.

    Raises:
        RuntimeError: `gpu` is given after torch was imported. `CUDA_VISIBLE_DEVICES` would then have no effect.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto", gpu: int | None = None) -> None:
        if gpu is not None:
            if "torch" in sys.modules:
                raise RuntimeError("torch is already imported, so CUDA_VISIBLE_DEVICES cannot select the GPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Imported here, not at the top: CUDA_VISIBLE_DEVICES takes effect only before torch's first import.
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.model_id = model_id
        self.device = torch.device(device)
        self.model = Sam3TrackerVideoModel.from_pretrained(model_id).to(self.device, dtype=torch.float32)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
        self._session = None
        self._size: tuple[int, int] | None = None

    def init_video(self, frames: np.ndarray) -> None:
        """Start a video: (T, H, W, 3) uint8 frames. Any earlier video and its prompts are dropped."""
        self._session = self.processor.init_video_session(video=frames, inference_device=self.device,
                                                          dtype=self._torch.float32)
        self._size = (self._session.video_height, self._session.video_width)

    def add_masks(self, frame_idx: int, masks: Sequence[np.ndarray], obj_ids: Sequence[int]) -> None:
        """Prompt one object per mask on frame `frame_idx`.

        Args:
            frame_idx: the frame the masks belong to.
            masks: (H, W) bool or uint8 arrays at the video's resolution, one per object.
            obj_ids: each object's id.

        Raises:
            RuntimeError: no video was started.
            ValueError: no mask is given, or not one id per mask.
        """
        if self._session is None:
            raise RuntimeError("init_video comes before add_masks")
        if not masks or len(masks) != len(obj_ids):
            raise ValueError(f"{len(masks)} masks and {len(obj_ids)} ids; one id per mask, at least one")
        self.processor.add_inputs_to_inference_session(
            inference_session=self._session, frame_idx=frame_idx, obj_ids=list(obj_ids),
            input_masks=[np.asarray(m, dtype=np.uint8) for m in masks], clear_old_inputs=True)

    def propagate(self, start_frame_idx: int, reverse: bool = False) -> Iterator[FrameResult]:
        """Yield each frame's result, from `start_frame_idx` to the end, or to the start when `reverse`.

        Raises:
            RuntimeError: no video was started.
        """
        if self._session is None:
            raise RuntimeError("init_video comes before propagate")

        @self._torch.inference_mode()
        def frames():
            for output in self.model.propagate_in_video_iterator(
                    self._session, start_frame_idx=start_frame_idx, max_frame_num_to_track=None, reverse=reverse):
                yield self._result(output)

        return frames()

    def _result(self, output) -> FrameResult:
        """Upsample one output's mask logits to the video's resolution and read each object's presence score."""
        torch = self._torch
        h, w = self._size
        logits = output.pred_masks
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        masks = self.processor.post_process_masks([logits.to(dtype=torch.float32).cpu()],
                                                  original_sizes=[[h, w]])[0].squeeze(1)
        object_ids = list(output.object_ids)
        scores = np.full(len(object_ids), np.nan, dtype=np.float32)
        for i, obj_id in enumerate(object_ids):
            logit = self._presence_logit(self._session.obj_id_to_idx(obj_id), output.frame_idx)
            if logit is not None:
                scores[i] = float(torch.as_tensor(logit).reshape(-1)[0].sigmoid())
        return FrameResult(frame_idx=output.frame_idx, object_ids=object_ids, masks=masks.numpy(), scores=scores)

    def _presence_logit(self, obj_idx: int, frame_idx: int):
        """Return an object's presence logit on a frame, or None where the session holds none.

        The frames propagated to hold their outputs as non-conditioning ones, and those are read. A frame with no
        such output gives None, without looking among the conditioning outputs: so the prompted frame, whose
        outputs are conditioning ones, has no score, and its objects are painted in the order the tracker lists
        them. The workbench's labels were made this way. The conditioning outputs are read only where a
        non-conditioning output lacks the logit.
        """
        for conditioning in (False, True):
            try:
                return self._session.get_output(obj_idx, frame_idx, "object_score_logits",
                                                is_conditioning_frame=conditioning)
            except KeyError:
                continue
        return None
