"""Instance colours for viewer overlays, for class-agnostic segmentation.

The fixed colour of a semantic class comes with its label table, whether
built from plain data (`labels`) or from a dataset's class table
(`gt_tables`). This palette is for the other case: instance ids that carry
no meaning of their own, which is what tracking produces, and only need to
be told apart at a glance.
"""

import numpy as np

# 24 easily distinguished colours (tab20 plus four). Looked up
# deterministically by instance id, so the same id gets the same colour in
# every frame and every clip: temporal consistency is what the colour shows.
INSTANCE_PALETTE = np.array([
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.68, 0.78, 0.91], [1.00, 0.73, 0.47],
    [0.60, 0.87, 0.54], [1.00, 0.60, 0.59], [0.77, 0.69, 0.84], [0.77, 0.61, 0.58],
    [0.97, 0.71, 0.82], [0.78, 0.78, 0.78], [0.86, 0.86, 0.55], [0.62, 0.85, 0.90],
    [0.20, 0.30, 0.49], [0.90, 0.33, 0.05], [0.30, 0.69, 0.29], [0.60, 0.10, 0.20],
], dtype=np.float64)

# Vertex label 0 is background. The viewer paints it dark grey when no colour
# is given for 0 and leaves it out of the legend; the 2D panel treats the
# same colour as transparent.
BACKGROUND_COLOR: list[float] = [0.1, 0.1, 0.1]


def instance_color(new_id: int) -> list[float]:
    """The stable float RGB colour of a remapped instance id.

    Args:
        new_id: an instance id, 1 or more.

    Returns:
        `[r, g, b]` as floats in 0 to 1.

    Raises:
        ValueError: `new_id` is below 1. Id 0 is the background and must not
            be coloured as a foreground instance; through the modulo it would
            silently get the palette's last colour.
    """
    new_id = int(new_id)
    if new_id < 1:
        raise ValueError(f"instance ids start at 1; {new_id} is the background or invalid")
    return INSTANCE_PALETTE[(new_id - 1) % len(INSTANCE_PALETTE)].tolist()
