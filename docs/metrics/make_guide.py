"""Build docs/metrics/index.html: what each v2 metric measures, drawn on a cartoon frame.

The main figures use the cartoon in `evalkit/tests/cartoon.py`, a liver,
gallbladder, fat and an instrument, predicted with one typical mistake each.
The appendix repeats the metrics on the small scenes of `evalkit/tests/scenes.py`,
whose numbers `evalkit/tests/test_v2_metric_guide.py` checks against values
worked out by hand. Every number on the page is computed by v2 when the page is
built. Rebuild after changing any of those files:

    python docs/metrics/make_guide.py
"""
import base64
import html
import io
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "evalkit", "tests"))

import cartoon as C         # noqa: E402
import scenes as S          # noqa: E402
import v2_import            # noqa: E402

E = v2_import.load()

GT_COLOURS = {0: (228, 228, 224), 1: (176, 74, 62), 2: (96, 150, 84), 3: (232, 200, 100), 4: (160, 166, 178)}
REGION_COLOURS = [(78, 121, 167), (176, 122, 161), (242, 142, 43), (118, 183, 178), (255, 157, 167)]
GT_LINE, PRED_LINE, BAND, BOTH = (30, 80, 200), (235, 110, 20), (170, 200, 245), (130, 60, 170)


# ── Pixels ───────────────────────────────────────────────────────────────────

def paint_gt(gt):
    out = np.zeros(gt.shape + (3,), np.uint8)
    for c, col in GT_COLOURS.items():
        out[gt == c] = col
    return out


def paint_regions(lab, fade=0.0):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for r in np.unique(lab):
        if r >= 0:
            out[lab == r] = REGION_COLOURS[int(r) % len(REGION_COLOURS)]
    yy, xx = np.indices(lab.shape)
    hatch = ((yy + xx) % 6) < 2
    out[(lab < 0) & hatch] = (70, 70, 70)
    out[(lab < 0) & ~hatch] = (125, 125, 125)
    if fade:
        out = (out * (1 - fade) + 255 * fade).astype(np.uint8)
    return out


def class_map(gt, lab, valid):
    pred = np.zeros_like(gt)
    for r, c in E._region_to_class(lab, gt, valid).items():
        pred[lab == r] = c
    return pred


def outline(img, mask, colour):
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    img[edge & mask] = colour
    return img


# Object-figure labels that would collide, moved by (dx, dy) image px.
OBJ_NUDGE = {"G1": (22, -8), "R0": (-6, 14), "R1": (6, 14), "G3": (-18, -2), "R2": (30, 14),
             "G4": (-10, -4), "G5": (26, -12), "R3": (8, 16), "G2": (0, -8), "R4": (0, 4)}

# Labels that would collide with a neighbour's, moved by (dx, dy) image px.
NUDGE = {C.GALLBLADDER: (-22, 9)}


def anchor(mask):
    """The point deepest inside `mask` — where a label reads best."""
    d = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return x + 0.5, y + 0.5


def nudged(mask, cls):
    x, y = anchor(mask)
    dx, dy = NUDGE.get(cls, (0, 0))
    return x + dx, y + dy


# ── SVG panels ───────────────────────────────────────────────────────────────

def _png(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def panel(rgb, marks=(), title="", scale=3, crop=None):
    """An image with text and lines drawn over it, as one SVG.

    Args:
        rgb: (H, W, 3) pixels.
        marks: ("text", x, y, str, cls) and ("line", x0, y0, x1, y1, cls) in pixel coordinates.
        title: Caption under the panel.
        scale: Display px per image px.
        crop: (x0, y0, x1, y1) to show only part of the image, enlarged.
    """
    if crop:
        x0, y0, x1, y1 = crop
        rgb = rgb[y0:y1, x0:x1]
    else:
        x0 = y0 = 0
    h, w = rgb.shape[:2]
    parts = [f'<image href="{_png(rgb)}" width="{w * scale}" height="{h * scale}" style="image-rendering:pixelated"/>']
    # Text is sized against the panel, so it reads the same however wide the panel is shown.
    size = w * scale * 0.032
    for m in marks:
        if m[0] == "text":
            _, x, y, s, cls = m
            fs = size * (0.8 if "small" in cls else 1.0)
            # Keep the whole label inside the panel: centred text near an edge would be cut off.
            half = 0.3 * fs * len(s)
            left = "iou" in cls
            xd = (x - x0) * scale
            xd = min(max(xd, 4 if left else half + 4), w * scale - (2 * half if left else half) - 4)
            parts.append(f'<text x="{xd:.1f}" y="{(y - y0) * scale:.1f}" class="{cls}" '
                         f'font-size="{fs:.1f}" stroke-width="{fs * 0.28:.1f}">{html.escape(s)}</text>')
        else:
            _, xa, ya, xb, yb, cls = m
            parts.append(f'<line x1="{(xa - x0) * scale:.1f}" y1="{(ya - y0) * scale:.1f}" '
                         f'x2="{(xb - x0) * scale:.1f}" y2="{(yb - y0) * scale:.1f}" class="{cls}"/>')
    svg = f'<svg viewBox="0 0 {w * scale} {h * scale}" role="img">{"".join(parts)}</svg>'
    return f'<div class="panel">{svg}<span>{html.escape(title)}</span></div>'


def figure(panels, caption):
    single = " single" if len(panels) == 1 else f" n{len(panels)}"
    return f'<figure><div class="panels{single}">{"".join(panels)}</div><figcaption>{caption}</figcaption></figure>'


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.2f}"


# ── The cartoon ──────────────────────────────────────────────────────────────

def cartoon_sections():
    gt, valid = C.ground_truth(), C.valid()
    lab = C.prediction(gt)
    fr = E.eval_frame(lab, gt, valid)
    inst = fr["inst"]

    # Inputs
    gt_marks = [("text", *anchor(gt == c), name, "lbl") for c, name in C.CLASS_NAMES.items()]
    gt_marks.append(("text", 214, 146, "background (0)", "lbl small"))
    reg_marks = [("text", *anchor(lab == r), f"region {r}", "lbl") for r in range(5)]
    reg_marks.append(("text", 44, 78, "no region (−1)", "lbl small"))
    inputs = figure([panel(paint_gt(gt), gt_marks, "GT: one class per pixel"),
                     panel(paint_regions(lab), reg_marks, "prediction: one region id per pixel, no classes")],
                    "The mistakes, one per structure: the liver is cut in two (regions 0 and 1); the gallbladder is merged "
                    "with the fat (region 2); the instrument is drawn 3 px too wide (region 3); region 4 lies on unannotated "
                    "background; and a band between liver and gallbladder has no region.")

    # Objects
    gid, ga = E._gt_instance_map(gt, valid)
    pid, pa, ids = E._pred_region_map(lab, valid)
    pairs = [(iou, gi, ids[pj]) for iou, gi, pj in E._match_pairs(gid, ga, pid, pa)]
    matched_g = {gi for iou, gi, _ in pairs if iou >= 0.5}
    matched_r = {r for iou, _, r in pairs if iou >= 0.5}
    g_anchor = {gi: anchor(gid == gi + 1) for gi in range(len(ga))}
    r_anchor = {r: anchor(lab == r) for r in ids}
    names = {gi: C.CLASS_NAMES[int(np.bincount(gt[gid == gi + 1]).argmax())] for gi in range(len(ga))}

    base = paint_regions(lab, fade=0.55)
    for gi in range(len(ga)):
        base = outline(base, gid == gi + 1, GT_LINE)
    # Where GT objects and regions share a spot, their labels are moved apart by hand (image px).
    g_anchor = {gi: (x + OBJ_NUDGE.get(f"G{gi + 1}", (0, 0))[0], y + OBJ_NUDGE.get(f"G{gi + 1}", (0, 0))[1])
                for gi, (x, y) in g_anchor.items()}
    r_anchor = {r: (x + OBJ_NUDGE.get(f"R{r}", (0, 0))[0], y + OBJ_NUDGE.get(f"R{r}", (0, 0))[1])
                for r, (x, y) in r_anchor.items()}
    obj_marks = []
    for gi in range(len(ga)):
        x, y = g_anchor[gi]
        tag = "  FN" if gi not in matched_g else ""
        obj_marks.append(("text", x, y, f"G{gi + 1} {names[gi]}{tag}", "lbl gt" + (" bad" if tag else "")))
    for r in ids:
        x, y = r_anchor[r]
        tag = "  FP" if r not in matched_r else ""
        obj_marks.append(("text", x, y, f"R{r}{tag}", "lbl reg" + (" bad" if tag else "")))
    for iou, gi, r in pairs:
        if iou >= 0.5:
            (xa, ya), (xb, yb) = g_anchor[gi], r_anchor[r]
            obj_marks.append(("line", xa, ya + 2, xb, yb - 5, "match"))
            obj_marks.append(("text", xa + 0.55 * (xb - xa) + 3, ya + 0.55 * (yb - ya) - 3, f"IoU {iou:.2f}", "lbl iou"))
    liver_parts = [gi for gi in range(len(ga)) if names[gi] == "liver"]
    objects = figure(
        [panel(base, obj_marks, "GT objects (blue outlines, G) over the predicted regions (R)", scale=3.4)],
        f"The instrument cuts the liver into two connected components, so v2 sees {len(liver_parts)} liver objects "
        f"(G{liver_parts[0] + 1}, G{liver_parts[1] + 1}); the small one has {ga[liver_parts[1]]} px, just above v2's "
        f"{E.INSTANCE_MIN_CC} px threshold. Lines join the pairs with IoU ≥ 0.5.")
    tp = inst["tp"]
    fp, fn = inst["n_pred"] - tp, inst["n_gt"] - tp
    obj_scores = table(["", "value", "from the figure"], [
        ["TP / FP / FN", f"{tp} / {fp} / {fn}", "matched pairs / unmatched regions / unmatched GT objects"],
        ["F1@0.5", fmt(inst["f1@0.5"]), f"2·{tp} / (2·{tp} + {fp} + {fn})"],
        ["SQ", fmt(inst["sq"]), "mean IoU of the matched pairs"],
        ["PQ", fmt(inst["pq"]), f"SQ × F1@0.5 = {inst['sq']:.2f} × {inst['f1@0.5']:.2f}"],
        ["F1@0.75", fmt(inst["f1@0.75"]), "no pair reaches IoU 0.75" if inst["f1@0.75"] == 0 else "pairs with IoU ≥ 0.75 only"],
        ["inst_BF", fmt(inst["inst_bf"]), "boundary F of each matched pair, averaged"],
    ])

    # Class map
    cmap = class_map(gt, lab, valid)
    vote = {r: C.CLASS_NAMES.get(c, "background") for r, c in E._region_to_class(lab, gt, valid).items()}
    cm_marks = [("text", *anchor(lab == r), f"R{r} → {vote[r]}", "lbl") for r in ids]
    iou_marks = [("text", *nudged(gt == c, c), f"{C.CLASS_NAMES[c]}: IoU {fr['ious'].get(c, 0):.2f}", "lbl")
                 for c in C.CLASS_NAMES]
    classmap = figure([panel(paint_regions(lab), cm_marks, "each region takes the class most of its pixels have"),
                       panel(paint_gt(cmap), iou_marks, "the class map that results, scored against the GT")],
                      f"Cutting the liver in two costs almost nothing: both regions vote liver (IoU {fr['ious'][C.LIVER]:.2f}; "
                      f"the band with no region is what is lost). Merging the gallbladder into the fat costs the gallbladder "
                      f"everything: the merged region votes fat, and the gallbladder is gone (IoU {fr['ious'].get(C.GALLBLADDER, 0):.2f}). "
                      f"mIoU = {np.mean(list(fr['ious'].values())):.2f}.")

    # Boundary
    gt_b = E._boundary(gt) & valid
    pb = E._boundary(cmap) & valid
    img = np.full(gt.shape + (3,), 250, np.uint8)
    img[E._dilate(gt_b, E.BOUNDARY_TOL)] = BAND
    img[gt_b] = GT_LINE
    img[pb & gt_b] = BOTH
    img[pb & ~gt_b] = PRED_LINE
    crop = (70, 40, 170, 110)
    bp, br, bf = fr["bf"]
    boundary = figure([panel(img, [], "whole frame"),
                       panel(img, [("text", 148, 60, "instrument: 3 px too wide", "lbl small")],
                             "enlarged", scale=6, crop=crop)],
                      f"Blue: the true boundary; light blue: within {E.BOUNDARY_TOL} px of it; orange: the predicted "
                      f"boundary; purple: both. The gallbladder's lower edge has no orange: in the class map the "
                      f"gallbladder became fat, so there is no edge there to predict. Along the top, the band with no "
                      f"region pushes the predicted edge out of the tolerance. Precision is the share of orange-or-purple inside the light band "
                      f"({bp:.2f}); recall is the share of blue-or-purple near a predicted boundary ({br:.2f}); "
                      f"boundary F = {bf:.2f}.")

    # Splitting and merging
    sm_marks = [("text", *nudged(gt == c, c), f"{C.CLASS_NAMES[c]}: {fr['overseg'].get(c, 0)} piece"
                 + ("s" if fr['overseg'].get(c, 0) != 1 else ""), "lbl") for c in C.CLASS_NAMES]
    splitmerge = figure([panel(paint_gt(gt), sm_marks, "overseg: regions covering ≥ 200 px and ≥ 5 % of each class")],
                        f"The liver is covered by two pieces, everything else by one. UE = {fr['underseg']:.2f}: mostly "
                        f"the fat's region crossing into the gallbladder. VI_split = {fr['vi_split']:.2f} bits (the liver "
                        f"cut, and the band with no region), VI_merge = {fr['vi_merge']:.2f} bits (the gallbladder merged "
                        f"into the fat).")

    # Time
    later = C.prediction(gt, tool_offset=8)
    swapped = C.prediction(gt, swap_liver=True)
    t_move = E.time_iou([lab, later], [valid] * 2)
    t_swap = E.time_iou([lab, swapped], [valid] * 2)
    rm = lambda L: [("text", *anchor(L == r), f"region {r}", "lbl") for r in range(5)]
    time = figure([panel(paint_regions(lab), rm(lab), "frame t"),
                   panel(paint_regions(later), rm(later), "frame t+1: the instrument moves 8 px"),
                   panel(paint_regions(swapped), rm(swapped), "frame t+1: the liver regions swap ids")],
                  f"time_IoU compares each region id with itself one frame later. The moving instrument keeps its id "
                  f"and overlaps most of itself: time_IoU = {t_move:.2f}. The id swap makes both liver regions miss "
                  f"themselves entirely: time_IoU = {t_swap:.2f}.")

    return dict(inputs=inputs, objects=objects, obj_scores=obj_scores, classmap=classmap,
                boundary=boundary, splitmerge=splitmerge, time=time)


# ── The worked examples ──────────────────────────────────────────────────────

def appendix():
    rows_obj, rows_rest = [], []
    for sc in [S.exact(), S.shifted(3), S.split(), S.merged(), S.gap(10), S.gap(10, painted=True),
               S.on_background(), S.sliver(299), S.sliver(300)]:
        fr = E.eval_frame(sc.lab, sc.gt, sc.valid)
        i = fr["inst"]
        rows_obj.append([html.escape(sc.title), fmt(i["f1@0.5"]), fmt(i["f1@0.75"]), fmt(i["sq"]),
                         fmt(i["pq"]), fmt(i["f1_avg"])])
        rows_rest.append([html.escape(sc.title), fmt(np.mean(list(fr["ious"].values()))), fmt(fr["bf"][2]),
                          fmt(fr["br_raw"][1]), fmt(fr["br_raw"][0]), fmt(fr["underseg"]),
                          fmt(fr["vi_split"]), fmt(fr["vi_merge"])])
    for sc in [S.spill(4), S.spill(5)]:
        fr = E.eval_frame(sc.lab, sc.gt, sc.valid)
        rows_rest.append([html.escape(sc.title) + f" (overseg of class 2: {fr['overseg'][2]})",
                          fmt(np.mean(list(fr["ious"].values()))), fmt(fr["bf"][2]), fmt(fr["br_raw"][1]),
                          fmt(fr["br_raw"][0]), fmt(fr["underseg"]), fmt(fr["vi_split"]), fmt(fr["vi_merge"])])
    pics = [panel(paint_gt(sc.gt), [], "GT") for sc in [S.split()]] + \
           [panel(paint_regions(sc.lab), [], sc.title) for sc in [S.split(), S.merged(), S.gap(10)]]
    return (figure(pics, "Each scene is 60 × 100 px: GT class 1 on the left, class 2 on the right.")
            + table(["scene", "F1@0.5", "F1@0.75", "SQ", "PQ", "F1_avg"], rows_obj)
            + table(["scene", "mIoU", "boundary F", "R_raw", "P_raw", "UE", "VI_split", "VI_merge"], rows_rest))


# ── Page ─────────────────────────────────────────────────────────────────────

CSS = """
:root { --bg:#fbfbf9; --fg:#1d1d1b; --muted:#66655f; --line:#e2e1db; --accent:#2b62c9; --v3:#8a5a00; --bad:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29b; --line:#34332f; --accent:#8db1ff; --v3:#e0b050; --bad:#ff7b6b; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 -apple-system, system-ui, sans-serif; }
main { max-width:1040px; margin:0 auto; padding:32px 16px 96px; }
h1 { font-size:28px; margin:0 0 8px; } h2 { font-size:21px; margin:44px 0 8px; padding-top:12px; border-top:1px solid var(--line); }
.lead, .note { color:var(--muted); } code { font-size:.92em; }
figure { margin:18px 0; } .panels { display:flex; flex-wrap:wrap; gap:14px; }
.panel { display:flex; flex-direction:column; gap:4px; flex:1 1 440px; max-width:100%; }
.panels.single .panel { flex:0 1 760px; }
.panels.n3 .panel { flex:1 1 300px; }
.panel svg { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
.panel span, figcaption { font-size:14px; color:var(--muted); }
svg text { font-family:-apple-system, system-ui, sans-serif; font-weight:600; fill:#161616; paint-order:stroke; stroke:#fff; text-anchor:middle; }
svg text.small { font-weight:500; }
svg text.gt { fill:#1e46a8; } svg text.reg { fill:#3a3a3a; } svg text.bad { fill:#c0392b; }
svg text.iou { fill:#1d6b2f; text-anchor:start; }
svg line.match { stroke:#1d6b2f; stroke-width:2.5; stroke-dasharray:5 3; }
table { border-collapse:collapse; margin:14px 0; font-size:14px; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--line); padding:5px 10px; text-align:left; white-space:nowrap; }
.v3 { color:var(--v3); }
"""


def build():
    c = cartoon_sections()
    body = f'''
<h1>What the v2 metrics measure</h1>
<p class="lead">v2 is the evaluator every score so far was measured with (<code>reference/eval_v2/</code>). This page
draws what each of its metrics looks at, on a cartoon of a laparoscopic frame: liver, gallbladder, fat and an
instrument, and a prediction that gets each of them wrong in one typical way. Every number is computed by v2 when
the page is built. The appendix repeats the metrics on tiny scenes whose numbers the tests check by hand.
Where <code>docs/eval_v3.md</code> changes a behaviour, the page says so.</p>

<h2>The frame</h2>
<p>v2 is given a <b>GT class map</b> (one class per pixel; 0 is unannotated background), the <b>predicted regions</b>
(one id per pixel from tracking, with no class; −1 means no region), and the <b>valid pixels</b>, those with a
finite, positive depth. Here every pixel is valid.</p>
{c["inputs"]}

<h2>1 · Did it find the objects, and how well do they fit? F1@t, SQ, PQ, F1_avg, inst_BF</h2>
<ol>
<li>GT objects are the connected components of each class with at least {E.INSTANCE_MIN_CC} px; predicted objects
are regions with at least {E.INSTANCE_MIN_CC} px. Background is never an object, and pixels with no region are
not a region. <span class="v3">v3: an object is a class's whole region in the frame, so the two liver pieces below
are one object.</span></li>
<li>Pairs are taken greedily, highest IoU first, each object at most once. A pair with IoU ≥ 0.5 is a hit (TP).
An unmatched region is a false positive (FP), and an unmatched GT object a miss (FN).</li>
</ol>
{c["objects"]}
{c["obj_scores"]}
<p class="note">F1@0.5 counts hits and nothing else: a pair at IoU 0.51 and one at 0.99 score the same. SQ says how
well the hits fit. PQ is exactly SQ × F1@0.5. F1_avg, the mean of F1@t over t = 0.50 … 0.95, is fixed to within
0.1·F1@0.5 by PQ and F1@0.5, so it adds nothing. <span class="v3">v3 keeps F1@0.5 and SQ, and drops PQ, F1@0.75 and
F1_avg.</span></p>

<h2>2 · Would the regions give the right class map? GT_mIoU, GT_mDice</h2>
{c["classmap"]}
<p class="note">Dice is 2·IoU / (1 + IoU) class by class, so GT_mDice repeats GT_mIoU. A tie in the vote goes to
the smaller class id. <span class="v3">v3 keeps mIoU only.</span></p>

<h2>3 · Are the edges in the right place? boundary_F, P, R and the raw variants</h2>
<p>A boundary pixel is one whose left, right, upper or lower neighbour has another label; both sides of an edge are
marked, so a boundary is 2 px wide. boundary_F compares the boundaries of the class map from section 2 with the GT's,
allowing {E.BOUNDARY_TOL} px. boundary_R_raw and P_raw do the same with the regions themselves, before any class is
assigned: an extra cut through an organ costs precision there, never recall. v2 also reports tolerances 1, 3 and 5,
points on the same curve. <span class="v3">v3 keeps boundary_F and boundary_R_raw at 2 px.</span></p>
{c["boundary"]}

<h2>4 · Is anything split or merged? overseg_mean, underseg_error, VI_split, VI_merge</h2>
{c["splitmerge"]}
<ul class="note">
<li><b>overseg</b> counts, per GT class, the regions that each cover at least 200 px <b>and</b> 5 % of it. v2's comment
says "or"; the code says "and". A region that merely spills into a neighbour is counted as one of its pieces.</li>
<li><b>underseg_error</b> sums, over every class S and region R touching it, the smaller of |R ∩ S| and |R ∖ S|,
divided by the GT area.</li>
<li><b>VI_split</b> = H(regions | GT) and <b>VI_merge</b> = H(GT | regions), in bits, over non-background pixels.
Pixels with no region count as a region of their own.</li>
</ul>
<p class="note"><span class="v3">v3 keeps VI_split and VI_merge, and drops overseg and UE.</span></p>

<h2>5 · Does a region keep its id over time? time_IoU</h2>
{c["time"]}
<p class="note">An id present in only one of the two frames is skipped, so a region that disappears, or a new one,
costs nothing.</p>

<h2>Behaviours to know</h2>
<ul>
<li>Thresholds are pixel counts at the evaluation resolution: {E.INSTANCE_MIN_CC} px for objects; 200 px and 5 % for overseg.</li>
<li>Pixels with no region are no false positive for objects, but they are background in the class map, which moves
its boundaries, and a region of their own in VI.</li>
<li>CholecSeg8k: v2's colour table gives Hepatic Vein a colour no mask uses, so the class is read as background.</li>
</ul>

<h2>Appendix · Worked examples</h2>
<p class="note">Tiny scenes, 60 × 100 px, where every number can be derived by hand;
<code>evalkit/tests/test_v2_metric_guide.py</code> checks each of them.</p>
{appendix()}
'''
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>v2 metric guide</title><style>{CSS}</style></head><body><main>{body}</main></body></html>')


if __name__ == "__main__":
    out = os.path.join(HERE, "index.html")
    with open(out, "w") as f:
        f.write(build())
    print(out, f"{os.path.getsize(out) / 1e3:.0f} kB")
