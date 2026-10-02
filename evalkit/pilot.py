"""Pilot mode's objects and domains: the pilot evaluator's rules, kept to reproduce its scores.

Pilot mode exists for one check: run with the pilot evaluator's rules, this
evaluator must write the pilot evaluator's numbers on the 38 conditions it
scored, at zero tolerance (`docs/evaluation.md`, "Checked against the pilot
evaluator"). This module holds the two rules by which the pilot's objects
differ from the evaluator's. Its pairing, `F1_50` and `SQ` are the
evaluator's own, shared through `evalkit.objects`.

- **Objects.** A GT object is one 8-connected component of one class, of at
  least `PILOT_MIN_CC_PX` pixels within the domain; a predicted object is
  one region with at least that many pixels in the domain. The evaluator
  has whole-class objects and no minimum size, and nothing but this module
  reads the cut. The components come from `cv2.connectedComponents`, as the
  pilot's did, numbered class by class ascending and component by component
  in that function's label order; the pairing's tie rule reads the numbers,
  so the order is part of the rule.
- **Domains.** In place of the views, four masks: `full`, the valid pixels;
  `labeled`, without the GT's background; `tissue`, without the dataset's
  instrument classes; `labeled_tissue`, both. The instrument ids are the
  pilot's own (`PILOT_INSTRUMENT_IDS`), not the tables' types: for
  ATLAS-120k the pilot knew one tool class, `Tools/camera`, where the
  table types Catheter and Non anatomical structures as tools too.

The pilot removed its `EXTRA_IGNORE` ids from the GT objects as well. Every
score it wrote with the option recorded has it empty, so the set is not an
argument here; the check against the 38 conditions will say if that
assumption fails.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import cv2
import numpy as np

from evalkit.objects import Objects, _check

# The pilot evaluator's minimum size of a connected component and of a
# predicted region, in pixels at the evaluation resolution. Nothing but
# pilot mode reads it.
PILOT_MIN_CC_PX = 300

# The pilot evaluator's background id, the same in both datasets' id maps.
PILOT_BACKGROUND = 0

# The GT ids the pilot evaluator's `tissue` domains removed, per dataset,
# under the names the pilot used.
PILOT_INSTRUMENT_IDS: Mapping[str, frozenset[int]] = MappingProxyType({
    "cholec": frozenset({5, 9}),   # Grasper, L-hook Electrocautery
    "atlas": frozenset({1}),       # Tools/camera
})

PILOT_DOMAINS = ("full", "labeled", "tissue", "labeled_tissue")


def pilot_gt_objects(gt: np.ndarray, domain: np.ndarray) -> Objects:
    """The pilot's GT objects: per-class 8-connected components of at least `PILOT_MIN_CC_PX`.

    Args:
        gt: An (H, W) integer map of GT class ids; `PILOT_BACKGROUND` is no
            object.
        domain: An (H, W) bool mask, one of `pilot_domains`.

    Returns:
        The objects, numbered class by class ascending and, within a class,
        in `cv2.connectedComponents` label order. `ids` holds each object's
        class id, so several objects share an id.
    """
    gt, domain = _check(gt, domain)
    present = np.unique(gt[domain])
    if present.size and present.min() < 0:
        raise ValueError("a GT class map holds no negative ids; -1 is a prediction's 'no region'")
    index = np.zeros(gt.shape, dtype=np.int32)
    ids, areas = [], []
    for c in present.tolist():
        if c == PILOT_BACKGROUND:
            continue
        # The components of the class inside the domain: a class cut in two
        # by the domain's edge is two objects, as it was for the pilot.
        n, labels = cv2.connectedComponents(((gt == c) & domain).astype(np.uint8))
        counts = np.bincount(labels.ravel(), minlength=n)
        for i in range(1, n):
            if int(counts[i]) >= PILOT_MIN_CC_PX:
                ids.append(c)
                areas.append(int(counts[i]))
                index[labels == i] = len(areas)
    return Objects(index=index, ids=tuple(ids), areas=tuple(areas), scored=domain)


def pilot_predicted_objects(regions: np.ndarray, domain: np.ndarray) -> Objects:
    """The pilot's predicted objects: regions with at least `PILOT_MIN_CC_PX` pixels in the domain.

    Args:
        regions: An (H, W) integer map of region ids, -1 for no region.
        domain: An (H, W) bool mask, one of `pilot_domains`.

    Returns:
        The objects, by region id ascending, each the region's pixels in
        the domain.
    """
    regions, domain = _check(regions, domain)
    if regions.min(initial=0) < -1:
        raise ValueError("a region map holds region ids >= 0 and -1 for no region")
    member = domain & (regions >= 0)
    present, counts = np.unique(regions[member], return_counts=True)
    keep = counts >= PILOT_MIN_CC_PX
    present, counts = present[keep], counts[keep]
    index = np.zeros(regions.shape, dtype=np.int32)
    for k, r in enumerate(present.tolist(), start=1):
        index[member & (regions == r)] = k
    return Objects(
        index=index, ids=tuple(int(r) for r in present.tolist()),
        areas=tuple(int(a) for a in counts.tolist()), scored=domain,
    )


def pilot_domains(gt: np.ndarray, valid: np.ndarray, instrument_ids: frozenset[int]) -> dict[str, np.ndarray]:
    """The pilot evaluator's four domains, by name.

    Args:
        gt: An (H, W) integer map of GT class ids.
        valid: The (H, W) bool mask of pixels with valid depth, as
            `valid_depth(depth, pilot=True)` returns it.
        instrument_ids: The GT ids the `tissue` domains remove, one entry of
            `PILOT_INSTRUMENT_IDS`. The pilot scored no `tissue` domain
            without them, so they are required.
    """
    gt, valid = _check(gt, valid)
    if not instrument_ids:
        raise ValueError("the pilot's tissue domains need the dataset's instrument ids; none were given")
    labeled = valid & (gt != PILOT_BACKGROUND)
    not_instrument = ~np.isin(gt, sorted(instrument_ids))
    return {
        "full": valid,
        "labeled": labeled,
        "tissue": valid & not_instrument,
        "labeled_tissue": labeled & not_instrument,
    }
