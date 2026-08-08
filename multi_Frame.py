"""
Multi-frame octopus skeleton pipeline

Usage
-----
    python R2.py frames_dir output_dir [--first-frame-arms 8]
                     [--stride 3] [--fps 30] [--video-fps 5]
                     [--no-edit] [--iterations 2]
                     [--min-arms 5] [--max-arms 8]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import Skeleton
from Skeleton import (Branch, cumulative_arc, interpolate_arc, curve_curvature,
                 fit_mask_constrained_spline, snap_points_to_mask,
                 graph_metrics, quality_score, validate_requirements,
                 export_all, save_graph_figure, save_overlay,
                 save_skeleton_png, load_binary, dense_iteration,
                 build_branches, construct_graph, print_metrics, clean_json)

LOG = logging.getLogger("multiframe")
logging.basicConfig(level=logging.INFO, format="%(message)s")

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# Single-frame processing (thin wrapper )


def process_frame(mask: np.ndarray, iterations: int, max_dimension: int,
                  min_arms: int, max_arms: int, exact_arms: Optional[int] = None
                  ) -> Tuple[List[Dict], List[Dict], Dict, List[Branch]]:
    """Run the Skeleton optimization loop on one mask. Unlike Skeleton.run, a frame
    whose best candidate violates soft constraints is still returned (with
    a warning) so that one bad frame never kills a whole sequence.

    When ``exact_arms`` is supplied the pipeline searches with the full
    [min_arms, max_arms] range so the skeletonizer is never forced into an
    impossible constraint, then **prefers** candidates that match the
    requested arm count.  If no candidate has exactly that many arms the
    closest one is accepted with a warning instead of raising an error."""
    settings = [(0.75, 0.90), (1.00, 0.65), (1.25, 0.45), (1.45, 0.30)]
    settings = settings[:max(1, iterations)]
    candidates = []
    for i, (morph_s, spline_s) in enumerate(settings, 1):
        try:
            # Always search with the flexible range so the skeletonizer
            # can find whatever arms the mask actually supports.
            dense = dense_iteration(mask, i, max_dimension, morph_s,
                                    min_arms, max_arms)
            branches = build_branches(dense, mask, spline_s)
            nodes, edges = construct_graph(branches, mask)
            metrics = graph_metrics(nodes, edges, mask, branches)
            score = quality_score(metrics, max_arms)
            # Bonus for matching the requested arm count (soft preference,
            # not a hard filter).
            if exact_arms is not None:
                score += 2000.0 - 2000.0 * abs(metrics["arm_count"] - exact_arms)
            candidates.append((score, nodes, edges, metrics, branches))
        except Exception as exc:
            LOG.warning(f"    iteration {i} failed: {exc}")
    if not candidates and min_arms > 3:
        # Relaxed retry: occluded / self-overlapping poses legitimately show
        # fewer distinct arms in a 2D silhouette. Continuity of the sequence
        # matters more than the per-frame arm floor.
        LOG.warning(f"    retrying frame with relaxed min-arms=3 (was {min_arms})")
        return process_frame(mask, iterations, max_dimension, 3, max_arms,
                             exact_arms)
    if not candidates:
        raise RuntimeError("all iterations failed for this frame")
    # Prefer the candidate closest to the requested arm count when exact_arms
    # is set, otherwise the highest raw quality score.
    if exact_arms is not None:
        score, nodes, edges, metrics, branches = max(
            candidates, key=lambda c: c[0] -
            1000.0 * abs(c[3]["arm_count"] - exact_arms))
    else:
        score, nodes, edges, metrics, branches = max(candidates, key=lambda c: c[0])
    if exact_arms is not None and metrics["arm_count"] != exact_arms:
        LOG.warning(f"    requested {exact_arms} arms but best candidate has "
                     f"{metrics['arm_count']}; accepting closest match")
    defects = validate_requirements(nodes, edges, metrics, min_arms, max_arms)
    if defects:
        LOG.warning("    accepted with defects: " + "; ".join(defects))
    return nodes, edges, metrics, branches


# ---------------------------------------------------------------------------
# Temporal arm identity (Hungarian matching against the previous frame)
# ---------------------------------------------------------------------------

def arm_signatures(nodes: List[Dict], edges: List[Dict]) -> Dict[int, Dict]:
    """Per-arm feature vector: base/mid/tip positions relative to the mantle
    center, launch angle, and total arm length."""
    center = next(n for n in nodes if n["is_center"])
    c = np.array([center["x"], center["y"]])
    sigs: Dict[int, Dict] = {}
    arm_ids = sorted({n["branch_id"] for n in nodes if n["branch_id"] > 0})
    for a in arm_ids:
        ns = {n["body_part"].split()[-1]: n for n in nodes if n["branch_id"] == a}
        # body_part endings: "Base", "1" (Mid 1), "Tip"
        base = ns.get("Base"); tip = ns.get("Tip")
        mid = next((n for n in nodes if n["branch_id"] == a
                    and "Mid" in n["body_part"]), None)
        if base is None or tip is None:
            continue
        b = np.array([base["x"], base["y"]]) - c
        t = np.array([tip["x"], tip["y"]]) - c
        m = (np.array([mid["x"], mid["y"]]) - c) if mid else 0.5 * (b + t)
        length = sum(e["length"] for e in edges if e["branch_id"] == a)
        sigs[a] = {"base": b, "mid": m, "tip": t,
                   "angle": math.atan2(b[1], b[0]), "length": length}
    return sigs


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def match_arms(prev_sig: Dict[int, Dict], cur_sig: Dict[int, Dict],
               diag: float) -> Dict[int, int]:
    """Return mapping {current_arm_id -> persistent_label} by minimizing a
    Hungarian cost over base/mid/tip displacement + launch-angle change."""
    prev_ids = sorted(prev_sig); cur_ids = sorted(cur_sig)
    if not prev_ids or not cur_ids:
        return {a: a for a in cur_ids}
    C = np.zeros((len(cur_ids), len(prev_ids)))
    for i, ca in enumerate(cur_ids):
        for j, pa in enumerate(prev_ids):
            s, p = cur_sig[ca], prev_sig[pa]
            d = (0.45 * np.linalg.norm(s["tip"] - p["tip"]) +
                 0.30 * np.linalg.norm(s["mid"] - p["mid"]) +
                 0.25 * np.linalg.norm(s["base"] - p["base"]))
            C[i, j] = d / diag + 0.35 * _ang_diff(s["angle"], p["angle"]) \
                + 0.15 * abs(s["length"] - p["length"]) / diag
    ri, cj = linear_sum_assignment(C)
    mapping: Dict[int, int] = {}
    used = set()
    for i, j in zip(ri, cj):
        mapping[cur_ids[i]] = prev_ids[j]
        used.add(prev_ids[j])
    # arms with no previous partner keep the lowest unused labels
    free = [k for k in range(1, 64) if k not in used]
    for a in cur_ids:
        if a not in mapping:
            mapping[a] = free.pop(0)
    return mapping


def relabel(nodes: List[Dict], edges: List[Dict],
            mapping: Dict[int, int]) -> None:
    """Rewrite branch ids / body-part strings in place with persistent labels."""
    def rename(txt: str, old: int, new: int) -> str:
        return txt.replace(f"Arm {old}", f"Arm {new}")
    for n in nodes:
        old = n["branch_id"]
        if old > 0 and old in mapping:
            n["branch_id"] = mapping[old]
            n["body_part"] = rename(n["body_part"], old, mapping[old])
    for e in edges:
        old = e["branch_id"]
        if old > 0 and old in mapping:
            e["branch_id"] = mapping[old]
            e["body_part"] = rename(e["body_part"], old, mapping[old])
            e["label"] = rename(e["label"], old, mapping[old])


# ---------------------------------------------------------------------------
# Mask-aware temporal fitting (previous graph -> current mask/current graph)
# ---------------------------------------------------------------------------

def mask_centroid(mask: np.ndarray) -> np.ndarray:
    """Return the foreground centroid in x/y order."""
    m = cv2.moments((mask > 0).astype(np.uint8))
    if m["m00"] > 0:
        return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    h, w = mask.shape
    return np.array([0.5 * w, 0.5 * h])


def _mask_tree(mask: np.ndarray) -> Tuple[cKDTree, np.ndarray]:
    yx = np.argwhere(mask > 0)
    if len(yx) == 0:
        raise ValueError("current mask has no foreground pixels")
    xy = np.column_stack([yx[:, 1], yx[:, 0]]).astype(float)
    return cKDTree(xy), xy


def _snap_with_distance(point: np.ndarray, tree: cKDTree,
                        xy: np.ndarray) -> Tuple[np.ndarray, float]:
    distance, index = tree.query(point)
    return xy[int(index)].copy(), float(distance)


def _edge_mask_coverage(edges: List[Dict], branch_id: int,
                        mask: np.ndarray) -> float:
    """Fraction of a rebuilt branch polyline that lies in the current mask."""
    pts = [np.asarray(e["polyline"], float) for e in edges
           if e["branch_id"] == branch_id and e.get("polyline")]
    if not pts:
        return 0.0
    p = np.vstack(pts)
    h, w = mask.shape
    x = np.clip(np.rint(p[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(p[:, 1]).astype(int), 0, h - 1)
    return float(np.mean(mask[y, x] > 0))


def temporal_fit(prev_nodes: List[Dict], prev_mask: np.ndarray,
                 detected_nodes: Optional[List[Dict]], mask: np.ndarray,
                 max_node_jump: float = 0.16, max_mask_gap: float = 0.08,
                 min_branch_coverage: float = 0.90
                 ) -> Tuple[List[Dict], List[Dict], Dict, List[Branch]]:
    """Fit the previous graph to ``mask``, using the new detection as evidence.

    The previous topology is the authority, which prevents arm flicker. A node
    with a plausible current match is blended with its translated previous
    position. A large mismatch falls back to the previous node snapped to the
    mask. If even that snap is implausibly far, only that node is omitted. An
    arm is removed only when fewer than two of its nodes survive or its rebuilt
    spline still has poor mask coverage.
    """
    diag = max(1.0, math.hypot(*mask.shape))
    soft_jump = max(4.0, max_node_jump * diag)
    hard_jump = 1.8 * soft_jump
    allowed_gap = max(3.0, max_mask_gap * diag)
    shift = mask_centroid(mask) - mask_centroid(prev_mask)
    tree, fg_xy = _mask_tree(mask)
    detected_by_key = ({node_key(n): n for n in detected_nodes}
                       if detected_nodes else {})

    fitted: List[Dict] = []
    omitted: List[str] = []
    for old in prev_nodes:
        n = copy.deepcopy(old)
        key = node_key(old)
        previous_guess = np.array([old["x"], old["y"]], float) + shift
        previous_snap, previous_gap = _snap_with_distance(
            previous_guess, tree, fg_xy)
        current = detected_by_key.get(key)

        # Center and head should always survive; for arm nodes, an excessive
        # mask snap means the local branch has disappeared/occluded.
        if current is None:
            if old["branch_id"] > 0 and previous_gap > allowed_gap:
                omitted.append(key)
                continue
            chosen = previous_snap
        else:
            current_xy = np.array([current["x"], current["y"]], float)
            jump = float(np.linalg.norm(current_xy - previous_guess))
            if jump <= soft_jump:
                chosen = 0.72 * current_xy + 0.28 * previous_guess
            elif jump <= hard_jump:
                chosen = 0.35 * current_xy + 0.65 * previous_guess
            elif previous_gap <= allowed_gap:
                # Do not let one bad detector node pull the whole arm away.
                chosen = previous_snap
                LOG.info(f"    repaired outlier node {key} from previous frame")
            elif old["branch_id"] > 0:
                omitted.append(key)
                continue
            else:
                chosen = current_xy
            chosen, _ = _snap_with_distance(chosen, tree, fg_xy)

        n["x"], n["y"] = float(chosen[0]), float(chosen[1])
        fitted.append(n)

    if omitted:
        LOG.info("    omitted incompatible nodes: " + ", ".join(omitted))

    # Keep a branch when at least one anatomical node remains. The graph
    # rebuilder can connect center directly to that node; dropping the whole
    # branch is reserved for zero surviving nodes or poor rebuilt coverage.
    arm_ids = sorted({n["branch_id"] for n in prev_nodes if n["branch_id"] > 0})
    dropped = []
    for arm_id in arm_ids:
        count = sum(n["branch_id"] == arm_id for n in fitted)
        if count == 0:
            dropped.append(arm_id)
    if dropped:
        fitted = [n for n in fitted if n["branch_id"] not in dropped]

    edges, branches = rebuild_graph_from_nodes(fitted, mask)
    weird = [a for a in arm_ids if a not in dropped and
             _edge_mask_coverage(edges, a, mask) < min_branch_coverage]
    if weird:
        fitted = [n for n in fitted if n["branch_id"] not in weird]
        dropped.extend(weird)
        edges, branches = rebuild_graph_from_nodes(fitted, mask)
    if dropped:
        LOG.info("    dropped mask-incompatible branches: " +
                 ", ".join(f"Arm {a}" for a in sorted(set(dropped))))

    metrics = graph_metrics(fitted, edges, mask, branches)
    return fitted, edges, metrics, branches


# ---------------------------------------------------------------------------
# Motion metadata: distance / speed / acceleration every `stride` frames
# ---------------------------------------------------------------------------

def node_key(n: Dict) -> str:
    """Stable anatomical identity of a node across frames."""
    if n["is_center"]:
        return "MantleCenter"
    if n.get("is_head"):
        return "Head"
    part = n["body_part"].replace(f"Arm {n['branch_id']} ", "")
    return f"Arm{n['branch_id']}_{part.replace(' ', '')}"


def compute_motion(frames: List[Dict], stride: int, fps: float) -> List[Dict]:
    """Distance / speed / acceleration using original source-frame indices.

    When an unreadable frame cannot be recovered, motion never pretends that
    adjacent processed records were adjacent source frames: the actual index
    gap determines dt and is exported unchanged.
    """
    rows: List[Dict] = []
    speeds_prev: Dict[str, Tuple[int, float, np.ndarray]] = {}
    for pos in range(1, len(frames)):
        cur = frames[pos]
        target = cur["index"] - stride
        eligible = [p for p in frames[:pos] if p["index"] <= target]
        if not eligible:
            continue
        prev = eligible[-1]
        frame_gap = cur["index"] - prev["index"]
        dt = frame_gap / fps
        pos_c = {node_key(n): np.array([n["x"], n["y"]]) for n in cur["nodes"]}
        pos_p = {node_key(n): np.array([n["x"], n["y"]]) for n in prev["nodes"]}
        for key in sorted(pos_c):
            if key not in pos_p:
                continue
            disp = pos_c[key] - pos_p[key]
            dist = float(np.linalg.norm(disp))
            speed = dist / dt
            vel = disp / dt
            accel = ""
            if key in speeds_prev:
                pidx, pspeed, pvel = speeds_prev[key]
                gap = (cur["index"] - pidx) / fps
                if gap > 0:
                    accel = float(np.linalg.norm(vel - pvel) / gap)
            speeds_prev[key] = (cur["index"], speed, vel)
            rows.append({
                "frame": cur["name"], "prev_frame": prev["name"],
                "frame_index": cur["index"], "node": key,
                "x": round(float(pos_c[key][0]), 2),
                "y": round(float(pos_c[key][1]), 2),
                "distance_px": round(dist, 3),
                "speed_px_per_s": round(speed, 3),
                "acceleration_px_per_s2": round(accel, 3) if accel != "" else "",
            })
    return rows


def export_motion(rows: List[Dict], out: Path) -> None:
    if not rows:
        LOG.warning("No motion rows (need more frames than the stride)")
        return
    fields = ["frame", "prev_frame", "frame_index", "node", "x", "y",
              "distance_px", "speed_px_per_s", "acceleration_px_per_s2"]
    with (out / "motion_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    with (out / "motion_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(clean_json(rows), f, indent=1)


# ---------------------------------------------------------------------------
# Rebuild edges from (possibly user-corrected) node positions
# ---------------------------------------------------------------------------

def rebuild_graph_from_nodes(nodes: List[Dict], mask: np.ndarray
                             ) -> Tuple[List[Dict], List[Branch]]:
    """Re-fit mask-constrained splines through the (edited) node chain of
    every arm and rebuild the edge list + Branch objects."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    h, w = mask.shape
    fg_yx = np.argwhere(mask > 0)
    fg_xy = np.column_stack([fg_yx[:, 1], fg_yx[:, 0]]).astype(float)
    fg_tree = cKDTree(fg_xy)

    # snap every node inside the mask and refresh its radius
    for n in nodes:
        p = snap_points_to_mask(np.array([[n["x"], n["y"]]], float),
                                mask, fg_tree, fg_xy)[0]
        n["x"], n["y"] = float(p[0]), float(p[1])
        xi = int(np.clip(round(n["x"]), 0, w - 1))
        yi = int(np.clip(round(n["y"]), 0, h - 1))
        n["radius"] = float(dt[yi, xi])

    center = next(n for n in nodes if n["is_center"])
    root = np.array([center["x"], center["y"]])
    # Node deletions leave harmless ID gaps, but several Skeleton exporters assume
    # IDs index the node array. Compact IDs and keep edge references coherent.
    for new_id, n in enumerate(nodes):
        n["node_id"] = new_id
    edges: List[Dict] = []
    branches: List[Branch] = []
    next_edge = 0
    arm_ids = sorted({n["branch_id"] for n in nodes if n["branch_id"] > 0})
    for a in arm_ids:
        chain = [center]
        arm_nodes = [n for n in nodes if n["branch_id"] == a]
        base = next((n for n in arm_nodes if "Base" in n["body_part"]), None)
        mids = sorted((n for n in arm_nodes if "Mid" in n["body_part"]),
                      key=lambda n: n["body_part"])
        tip = next((n for n in arm_nodes if "Tip" in n["body_part"]), None)
        chain.extend(([base] if base is not None else []) + mids +
                     ([tip] if tip is not None else []))
        if len(chain) < 2:
            continue
        pts = np.array([[n["x"], n["y"]] for n in chain], float)
        # densify control polyline before the mask-constrained fit
        seed = []
        for i in range(len(pts) - 1):
            k = max(2, int(np.linalg.norm(pts[i + 1] - pts[i]) / 4))
            seg = np.linspace(pts[i], pts[i + 1], k, endpoint=False)
            seed.append(seg)
        seed.append(pts[-1:])
        raw = np.vstack(seed)
        raw = snap_points_to_mask(raw, mask, fg_tree, fg_xy)
        curve, arc = fit_mask_constrained_spline(raw, mask, dt, 0.45)
        length = float(arc[-1])
        # arc positions of the (kept) interior nodes on the new curve
        tree = cKDTree(curve)
        node_arc = [0.0]
        for n in chain[1:]:
            _, j = tree.query([n["x"], n["y"]])
            node_arc.append(float(arc[j]))
        node_arc = np.maximum.accumulate(np.array(node_arc))
        node_arc[-1] = length
        node_xy = np.vstack([pts[0], pts[1:]])
        curv = curve_curvature(curve)
        branches.append(Branch(a, raw, curve, arc, length,
                               node_arc, node_xy, curv))
        ids = [n["node_id"] for n in chain]
        for j in range(len(ids) - 1):
            pl = Skeleton_best.split_curve(curve, arc, node_arc[j], node_arc[j + 1])
            pl_arc = cumulative_arc(pl)
            pc = curve_curvature(pl)
            sx = np.clip(np.rint(pl[:, 0]).astype(int), 0, w - 1)
            sy = np.clip(np.rint(pl[:, 1]).astype(int), 0, h - 1)
            edges.append({
                "edge_id": next_edge, "start_node": ids[j], "end_node": ids[j + 1],
                "branch_id": a, "body_part": f"Arm {a}", "label": f"Arm {a}",
                "length": float(pl_arc[-1]), "geodesic_distance": float(pl_arc[-1]),
                "average_radius": float(np.mean(dt[sy, sx])),
                "average_curvature": float(np.mean(pc)),
                "maximum_curvature": float(np.max(pc)),
                "polyline": [[float(x), float(y)] for x, y in pl]})
            next_edge += 1
    head = next((n for n in nodes if n.get("is_head")), None)
    if head is not None:
        hx, hy = head["x"], head["y"]
        n_s = max(6, int(math.hypot(hx - root[0], hy - root[1]) / 3))
        pl = np.linspace(root, [hx, hy], num=n_s)
        pl_arc = cumulative_arc(pl)
        sx = np.clip(np.rint(pl[:, 0]).astype(int), 0, w - 1)
        sy = np.clip(np.rint(pl[:, 1]).astype(int), 0, h - 1)
        edges.append({
            "edge_id": next_edge, "start_node": center["node_id"],
            "end_node": head["node_id"], "branch_id": 0, "body_part": "Head",
            "label": "Head", "length": float(pl_arc[-1]),
            "geodesic_distance": float(pl_arc[-1]),
            "average_radius": float(np.mean(dt[sy, sx])),
            "average_curvature": 0.0, "maximum_curvature": 0.0,
            "polyline": [[float(x), float(y)] for x, y in pl]})
    return edges, branches


# ---------------------------------------------------------------------------
# Interactive first-frame node editor (click node -> drag & drop -> rebuild)
# ---------------------------------------------------------------------------

def interactive_edit(nodes: List[Dict], edges: List[Dict],
                     mask: np.ndarray) -> bool:
    """Open a window showing the first processed graph over the silhouette.
    Click a node to select it (highlighted), drag & drop to move. Press
    'u' to undo the last move, Enter (or close the window) to accept.
    Returns True if any node was moved."""
    import matplotlib
    try:
        for backend in ("QtAgg", "TkAgg", "MacOSX"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                fig_test = plt.figure(); plt.close(fig_test)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("no interactive backend")
    except Exception:
        LOG.warning("No interactive display available - skipping node editing")
        return False
    import matplotlib.pyplot as plt

    moved = {"any": False}
    undo_stack: List[Tuple[int, float, float]] = []
    sel = {"idx": None}
    coords = np.array([[n["x"], n["y"]] for n in nodes])

    fig, ax = plt.subplots(figsize=(11, 11 * mask.shape[0] / mask.shape[1]))
    fig.canvas.manager.set_window_title(
        "Frame 1 - drag nodes to correct positions, press Enter when done")
    ax.imshow(mask, cmap="gray", alpha=0.35)
    edge_lines = []
    for e in edges:
        p = np.asarray(e["polyline"])
        ln, = ax.plot(p[:, 0], p[:, 1], lw=1.6, alpha=0.9,
                      color="#1cd679" if e["body_part"] == "Head" else
                      plt.get_cmap("tab10")((e["branch_id"] - 1) % 10))
        edge_lines.append(ln)
    colors = ["#ff3b30" if n["is_center"] else
              "#1cd679" if n.get("is_head") else
              "#ffd60a" if n["is_tip"] else "#53c8ff" for n in nodes]
    scat = ax.scatter(coords[:, 0], coords[:, 1], s=90, c=colors,
                      edgecolors="k", linewidths=1.0, zorder=5, picker=True)
    labels = [ax.annotate(n["body_part"], (n["x"], n["y"]), xytext=(6, -6),
                          textcoords="offset points", fontsize=7)
              for n in nodes]
    title = ax.set_title("Click a node to select, drag to move. "
                         "'u' = undo, Enter/close = accept", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(0, mask.shape[1])
    ax.set_ylim(mask.shape[0], 0)

    def redraw_sizes():
        sizes = np.full(len(nodes), 90.0)
        ec = np.array(["k"] * len(nodes), dtype=object)
        if sel["idx"] is not None:
            sizes[sel["idx"]] = 260.0
            ec[sel["idx"]] = "#ff2d95"
        scat.set_sizes(sizes); scat.set_edgecolors(list(ec))
        fig.canvas.draw_idle()

    def on_press(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        d = np.hypot(coords[:, 0] - ev.xdata, coords[:, 1] - ev.ydata)
        i = int(np.argmin(d))
        if d[i] < max(12.0, 0.02 * max(mask.shape)):
            sel["idx"] = i
            title.set_text(f"Selected: {nodes[i]['body_part']} - drag to new "
                           "position and release")
        else:
            sel["idx"] = None
            title.set_text("Click a node to select, drag to move. "
                           "'u' = undo, Enter/close = accept")
        redraw_sizes()

    def on_motion(ev):
        i = sel["idx"]
        if i is None or ev.inaxes != ax or ev.xdata is None or ev.button != 1:
            return
        coords[i] = [ev.xdata, ev.ydata]
        scat.set_offsets(coords)
        labels[i].xy = (ev.xdata, ev.ydata)
        fig.canvas.draw_idle()

    def on_release(ev):
        i = sel["idx"]
        if i is None or ev.xdata is None:
            return
        undo_stack.append((i, nodes[i]["x"], nodes[i]["y"]))
        nodes[i]["x"], nodes[i]["y"] = float(coords[i][0]), float(coords[i][1])
        moved["any"] = True
        title.set_text(f"Moved {nodes[i]['body_part']}. Select another node "
                       "or press Enter to accept")
        sel["idx"] = None
        redraw_sizes()

    def on_key(ev):
        if ev.key == "enter":
            plt.close(fig)
        elif ev.key == "u" and undo_stack:
            i, ox, oy = undo_stack.pop()
            nodes[i]["x"], nodes[i]["y"] = ox, oy
            coords[i] = [ox, oy]
            scat.set_offsets(coords); labels[i].xy = (ox, oy)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)
    return moved["any"]


# ---------------------------------------------------------------------------
# Video assembly from per-frame graph.png images
# ---------------------------------------------------------------------------

def build_video(png_paths: List[Path], out_path: Path, fps: float) -> None:
    imgs = [cv2.imread(str(p)) for p in png_paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        LOG.warning("No graph images to assemble into a video")
        return
    hmax = max(im.shape[0] for im in imgs)
    wmax = max(im.shape[1] for im in imgs)
    # even dimensions required by most codecs
    hmax += hmax % 2; wmax += wmax % 2
    for fourcc_name, suffix in (("mp4v", ".mp4"), ("XVID", ".avi")):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        path = out_path.with_suffix(suffix)
        vw = cv2.VideoWriter(str(path), fourcc, fps, (wmax, hmax))
        if not vw.isOpened():
            continue
        for im in imgs:
            canvas = np.full((hmax, wmax, 3), 17, np.uint8)
            y0 = (hmax - im.shape[0]) // 2
            x0 = (wmax - im.shape[1]) // 2
            canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            vw.write(canvas)
        vw.release()
        LOG.info(f"Video written: {path} ({len(imgs)} frames @ {fps} fps)")
        return
    LOG.warning("No usable video codec found")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_sequence(frames_dir: str, output_dir: str, stride: int = 3,
                 fps: float = 30.0, video_fps: float = 5.0,
                 iterations: int = 2, min_arms: int = 5, max_arms: int = 8,
                 max_dimension: int = 760, edit_first: bool = True,
                 first_frame_arms: Optional[int] = None) -> None:
    t0 = time.perf_counter()
    frames_path = Path(frames_dir)
    files = sorted(p for p in frames_path.iterdir()
                   if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No image frames found in {frames_dir}")
    if first_frame_arms is not None and not 1 <= first_frame_arms <= 32:
        raise SystemExit("--first-frame-arms must be between 1 and 32")
    if first_frame_arms is not None:
        min_arms = min(min_arms, first_frame_arms)
        max_arms = max(max_arms, first_frame_arms)
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)
    LOG.info(f"{len(files)} frames | stride={stride} | fps={fps}" +
             (f" | first-frame-arms={first_frame_arms}"
              if first_frame_arms is not None else ""))

    processed: List[Dict] = []
    prev_sig: Optional[Dict[int, Dict]] = None
    prev_mask: Optional[np.ndarray] = None
    graph_pngs: List[Path] = []

    for k, f in enumerate(files):
        LOG.info(f"\n[{k + 1}/{len(files)}] {f.name}")
        frame_out = out_root / f.stem
        frame_out.mkdir(exist_ok=True)
        try:
            mask = load_binary(str(f))
        except Exception as exc:
            LOG.warning(f"  unreadable mask, skipped: {exc}")
            continue

        detected_nodes: Optional[List[Dict]] = None
        detected_edges: Optional[List[Dict]] = None
        detection_error: Optional[Exception] = None
        try:
            exact = first_frame_arms if not processed else None
            detected_nodes, detected_edges, metrics, branches = process_frame(
                mask, iterations, max_dimension, min_arms, max_arms, exact)
        except Exception as exc:
            detection_error = exc

        if not processed:
            if (detection_error is not None or detected_nodes is None or
                    detected_edges is None):
                raise SystemExit(f"First frame failed: {detection_error}")
            nodes, edges = detected_nodes, detected_edges
        else:
            # Relabel the fresh detector output first so anatomical node keys
            # can be paired with persistent keys from the previous frame.
            if detected_nodes is not None and detected_edges is not None:
                raw_sig = arm_signatures(detected_nodes, detected_edges)
                mapping = match_arms(prev_sig or {}, raw_sig,
                                     math.hypot(*mask.shape))
                relabel(detected_nodes, detected_edges, mapping)
            else:
                LOG.warning(f"  detector failed ({detection_error}); "
                            "recovering every possible node from previous frame")
            try:
                nodes, edges, metrics, branches = temporal_fit(
                    processed[-1]["nodes"], prev_mask, detected_nodes, mask)
            except Exception as exc:
                # One last conservative recovery: translate and mask-snap the
                # previous graph without current detector evidence.
                LOG.warning(f"  temporal fusion failed ({exc}); using prior graph")
                try:
                    nodes, edges, metrics, branches = temporal_fit(
                        processed[-1]["nodes"], prev_mask, None, mask,
                        max_mask_gap=0.14, min_branch_coverage=0.82)
                except Exception as fallback_exc:
                    LOG.warning(f"  unrecoverable frame, skipped: {fallback_exc}")
                    continue

        # ---- interactive correction on the very first processed frame ----
        if k == 0 or not processed:
            first_png = frame_out / "graph.png"
            save_graph_figure(mask, nodes, edges, metrics, first_png)
            if edit_first:
                LOG.info("  opening editor window (drag nodes, Enter=done)...")
                if interactive_edit(nodes, edges, mask):
                    LOG.info("  rebuilding splines through corrected nodes...")
                    edges, branches = rebuild_graph_from_nodes(nodes, mask)
                    metrics = graph_metrics(nodes, edges, mask, branches)

        # ---- persistent identity after mask-aware fitting ----
        cur_sig = arm_signatures(nodes, edges)
        prev_sig = cur_sig
        prev_mask = mask.copy()

        # ---- export ----
        rec = {"iteration": 1, "settings": {}, "metrics": metrics,
               "score": quality_score(metrics, max_arms), "defects":
               validate_requirements(nodes, edges, metrics, min_arms, max_arms)}
        export_all(frame_out, nodes, edges, metrics, [rec])
        save_skeleton_png(mask, edges, frame_out / "skeleton.png")
        save_overlay(mask, nodes, edges, frame_out / "overlay.png")
        save_graph_figure(mask, nodes, edges, metrics, frame_out / "graph.png")
        graph_pngs.append(frame_out / "graph.png")
        processed.append({"name": f.name, "index": k, "nodes": nodes,
                          "edges": edges, "metrics": metrics})
        LOG.info(f"  ok: {metrics['arm_count']} arms, "
                 f"{metrics['total_nodes']} nodes")

    if not processed:
        raise SystemExit("Every frame failed - nothing to export")

    # ---- motion metadata (differences between every `stride`-th frame) ----
    rows = compute_motion(processed, stride, fps)
    export_motion(rows, out_root)
    LOG.info(f"\nmotion_metadata.csv: {len(rows)} rows "
             f"(stride={stride}, dt={stride / fps:.4f}s)")

    # ---- video of every frame's graph.png ----
    build_video(graph_pngs, out_root / "graphs_video.mp4", video_fps)
    LOG.info(f"Done: {len(processed)}/{len(files)} frames in "
             f"{time.perf_counter() - t0:.1f}s -> {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("frames", help="directory of binary-mask frames")
    ap.add_argument("output", help="output directory")
    ap.add_argument("--stride", type=int, default=3,
                    help="frame difference for distance/speed/accel (default 3)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="capture frame rate of the source video (default 30)")
    ap.add_argument("--video-fps", type=float, default=5.0,
                    help="playback fps of the generated graph video (default 5)")
    ap.add_argument("--iterations", type=int, default=2, choices=range(1, 5),
                    help="Skeleton refinement iterations per frame (default 2)")
    ap.add_argument("--first-frame-arms", type=int, metavar="N",
                    help="exact arm count required in frame 1; later frames "
                         "preserve these arm identities when the mask permits")
    ap.add_argument("--min-arms", type=int, default=5)
    ap.add_argument("--max-arms", type=int, default=8)
    ap.add_argument("--max-dimension", type=int, default=760)
    ap.add_argument("--no-edit", action="store_true",
                    help="skip the interactive first-frame node editor")
    args = ap.parse_args()
    run_sequence(args.frames, args.output, args.stride, args.fps,
                 args.video_fps, args.iterations, args.min_arms,
                 args.max_arms, args.max_dimension, not args.no_edit,
                 args.first_frame_arms)


if __name__ == "__main__":
    main()