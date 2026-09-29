"""Instance colour palette for viewer overlays, used for class-agnostic segmentation.

Fixed colours for semantic classes such as CholecSeg8k's live in
`surgical_core.cholec`. This palette is for the other case: instance ids that
carry no meaning of their own (what SAM 3 tracking produces), which only need to
be told apart at a glance.
"""
import numpy as np

# 24 easily distinguished colours (tab20 plus four). Looked up deterministically
# by instance id, so the same id gets the same colour in every frame and every
# clip — temporal consistency is what the colour is there to show.
INSTANCE_PALETTE = np.array([
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.68, 0.78, 0.91], [1.00, 0.73, 0.47],
    [0.60, 0.87, 0.54], [1.00, 0.60, 0.59], [0.77, 0.69, 0.84], [0.77, 0.61, 0.58],
    [0.97, 0.71, 0.82], [0.78, 0.78, 0.78], [0.86, 0.86, 0.55], [0.62, 0.85, 0.90],
    [0.20, 0.30, 0.49], [0.90, 0.33, 0.05], [0.30, 0.69, 0.29], [0.60, 0.10, 0.20],
], dtype=np.float64)

# Id 0 is background. The viewer paints it this dark grey when the palette gives
# it no colour, leaves it out of the segment labels, and draws it transparent in
# the 2D panel.
BACKGROUND_COLOR: list[float] = [0.1, 0.1, 0.1]


def instance_color(new_id: int) -> list[float]:
    """Map a remapped instance id (1..) to a stable float RGB colour.

    Id 0 is background and is never passed here.

    Args:
        new_id: Instance id, starting at 1.

    Returns:
        [r, g, b] as floats in 0–1.
    """
    # TODO(fail-closed): new_id <= 0 (background or invalid) still gets a
    #   foreground colour through the modulo — instance_color(0) is the last
    #   palette entry. Callers never pass 0 by convention, but one that did would
    #   colour the background. Raise for new_id < 1, with a test that passes 0.
    return INSTANCE_PALETTE[(int(new_id) - 1) % len(INSTANCE_PALETTE)].tolist()
