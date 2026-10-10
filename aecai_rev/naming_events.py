"""Item 6: read the detected events out as named English sentences.

The script takes a window's existing graph output (`temporal_graph__sam3d`
and `graph_frame_*__sam3d` under `pc_vis/`), detects the events with the
tagged `export_graph_figure.timeline_events` (the function the paper's
figure used), names every node by the CholecSeg8k class the overlay links it
to most often over the annotated frames (`hierarchy_frame_*.json`, the
figure's `hierarchy_links` with its threshold), and writes one sentence per
event, for example
`t=25: connective tissue (node 14) moves from in front of to behind the
L-hook electrocautery (node 15)`.

Usage:
    python3 -m aecai_rev.naming_events --clip VID12_s15_19900_crop [--focus 14] [--n 4]
"""
import argparse
import importlib
import json
import os
import re
import sys
from collections import Counter

from aecai_rev.config import load, write_meta
from aecai_rev.tagged import use

PHRASE = {"left": "left of", "right": "right of", "above": "above", "below": "below",
          "front": "in front of", "behind": "behind"}


def figure_module(cfg):
    """Import the tagged `scripts/export_graph_figure.py`."""
    use(cfg)
    scripts = os.path.join(cfg["paths"]["aecai_src"], "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("export_graph_figure")


def load_graph(cfg, clip):
    """Read the parts of a window's graph the event detection reads."""
    pc_vis = os.path.join(cfg["paths"]["cholec_gt"], clip, "pc_vis")
    with open(os.path.join(pc_vis, "temporal_graph__sam3d.json")) as f:
        temporal = json.load(f)
    graph = {}
    for fr in temporal["frames"]:
        with open(os.path.join(pc_vis, f"graph_frame_{fr:04d}__sam3d.json")) as f:
            graph[fr] = json.load(f)
    return dict(pc_vis=pc_vis, temporal=temporal, graph=graph, frames=temporal["frames"],
                seed=cfg["windows"]["seed_index"])


def node_names(eg, g, keep, names):
    """Name each node by the class the overlay links it to most often."""
    count = {n: Counter() for n in keep}
    index = os.path.join(g["pc_vis"], "hierarchy_index.json")
    with open(index) as f:
        frames = json.load(f)["frames"]
    for fr in frames:
        for link in eg.hierarchy_links(g, fr, keep, min_containment=0.3):
            count[link["region"]][link["gt"]] += 1
    out = {}
    for n, c in count.items():
        out[n] = names[c.most_common(1)[0][0]] if c else "unnamed region"
    return out, {n: dict(c) for n, c in count.items()}


def sentence(event, focus, partner, name):
    """Turn one detected event into an English sentence."""
    t = event["frame"]
    who = f"{name[focus]} (node {focus})"
    m = re.fullmatch(r"(\w+) → (\w+) of (\d+)", event["text"])
    if m:
        a, b = PHRASE.get(m.group(1), m.group(1)), PHRASE.get(m.group(2), m.group(2))
        other = int(m.group(3))
        return (f"t={t}: {who} moves from {a} to {b} the {name[other]} (node {other})")
    if event["text"].endswith("re-enters"):
        return f"t={t}: {who} re-enters the view"
    if event["text"].endswith("leaves the view"):
        return f"t={t}: {who} leaves the view"
    raise ValueError(f"unknown event text {event['text']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--focus", type=int, default=0, help="0: pick it as the figure does")
    ap.add_argument("--n", type=int, default=4, help="events to write")
    args = ap.parse_args()

    # Events: the tagged detection on the window's existing graph.
    cfg = load()
    eg = figure_module(cfg)
    g = load_graph(cfg, args.clip)
    keep = {n["id"] for n in g["temporal"]["nodes"]}
    focus = args.focus or eg.pick_focus(g, keep, g["seed"])
    partner = eg.focus_partner(g, focus, keep, g["seed"])
    events = eg.timeline_events(g, keep, focus=focus, max_events=args.n)

    # Names from the overlay, then one sentence per event.
    names = {int(k): v for k, v in cfg["classes"]["names"].items()}
    name, counts = node_names(eg, g, keep, names)
    lines = [sentence(e, focus, partner, name) for e in events]
    out = os.path.join(cfg["paths"]["results"], "06_naming")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "fig6_named_events.txt"), "w") as f:
        f.write(f"# window {args.clip}, focus node {focus}, partner node {partner}\n")
        f.write("\n".join(lines) + "\n")
    write_meta(os.path.join(out, "named_events_meta"), cfg, what="item 6: named event readout",
               clip=args.clip, focus=focus, partner=partner, node_names=name,
               link_counts=counts, events=events)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
