"""Track objects through a video with SAM 3, from mask prompts on one frame.

`Sam3VideoInstanceSession` loads SAM 3's video tracker once. For each video
it takes the frames, then one binary mask per object on one frame, and
yields every frame's masks and presence scores as it propagates. It depends
on torch and transformers, the `track` extra, and imports them only when a
session is built.
"""

from sam3_wrapper.video_session import DEFAULT_MODEL_ID, FrameResult, Sam3VideoInstanceSession  # noqa: F401
