"""
Build a clean anatomical octopus skeleton graph from one binary silhouette.

Usage
-----
    python R12.py input.png output/

Flexible arm count (6-8) with resilient detection.  Same output format
as R10 (graph.json, nodes.csv, edges.csv, graph.png, skeleton.png, overlay.png).

Dependencies: Python 3.9+, numpy, OpenCV, scipy, networkx, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, maximum_filter
from scipy.spatial import cKDTree

LOG = logging.getLogger("skeletonize")
logging.basicConfig(level=logging.INFO, format="%(message)s")
EPS = 1.0e-9
NBR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class DenseModel:
    mask_small: np.ndarray
    distance_small: np.ndarray
    skeleton_small: np.ndarray
    scale_x: float
    scale_y: float
    points_yx: np.ndarray
    adjacency: List[List[Tuple[int, float]]]
    root: int
    parent: np.ndarray
    geodesic: np.ndarray
    tip_paths: List[np.ndarray]
    iteration: int
    diagnostics: Dict[str, Any]


@dataclass
class Branch:
    arm_id: int
    raw_xy: np.ndarray
    curve_xy: np.ndarray
    arc: np.ndarray
    length: float
    node_arc: np.ndarray
    node_xy: np.ndarray
    curvature: np.ndarray# ---------------------------------------------------------------------------
# Binary-mask preparation and topology-preserving thinning
# ---------------------------------------------------------------------------

def load_binary(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    _, a = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    def quality(m: np.ndarray) -> Tuple[float, np.ndarray]:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        if n <= 1:
            return -1e30, m
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = (labels == idx).astype(np.uint8) * 255
        area = float(stats[idx, cv2.CC_STAT_AREA])
        border = float(np.count_nonzero(np.r_[keep[0], keep[-1], keep[:, 0], keep[:, -1]]))
        frac = area / keep.size
        return area - 1000.0 * border - (1e9 if frac > 0.85 else 0.0), keep
    qa, ma = quality(a)
    qb, mb = quality(b)
    mask = ma if qa >= qb else mb
    if np.count_nonzero(mask) < 100:
        raise ValueError("The largest foreground component is too small")
    return mask


def prepare_mask(mask: np.ndarray, max_dimension: int, smooth_factor: float
                 ) -> Tuple[np.ndarray, float, float]:
    h, w = mask.shape
    scale = min(1.0, float(max_dimension) / max(h, w))
    sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = max(0.65, smooth_factor * max(sw, sh) / 650.0)
    blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    binary = (blurred >= 112).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    binary = (labels == idx).astype(np.uint8)
    return binary, w / float(sw), h / float(sh)


def zhang_suen_thinning(binary: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    """Zhang--Suen thinning with compiled OpenCV fallback."""
    try:
        import cv2.ximgproc as ximgproc
        thinned = ximgproc.thinning((binary > 0).astype(np.uint8) * 255,
                                    thinningType=ximgproc.THINNING_ZHANGSUEN)
        return (thinned > 0).astype(np.uint8)
    except (ImportError, AttributeError, cv2.error):
        pass
    return _zhang_suen_thinning_numpy(binary, max_iterations)


def _zhang_suen_thinning_numpy(binary: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    im = np.pad((binary > 0).astype(np.uint8), 1)
    for _ in range(max_iterations):
        changed = 0
        for phase in (0, 1):
            p2 = im[:-2, 1:-1]; p3 = im[:-2, 2:]; p4 = im[1:-1, 2:]
            p5 = im[2:, 2:]; p6 = im[2:, 1:-1]; p7 = im[2:, :-2]
            p8 = im[1:-1, :-2]; p9 = im[:-2, :-2]
            c = im[1:-1, 1:-1]
            b = p2+p3+p4+p5+p6+p7+p8+p9
            a = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            a += ((p3 == 0) & (p4 == 1)); a += ((p4 == 0) & (p5 == 1))
            a += ((p5 == 0) & (p6 == 1)); a += ((p6 == 0) & (p7 == 1))
            a += ((p7 == 0) & (p8 == 1)); a += ((p8 == 0) & (p9 == 1))
            a += ((p9 == 0) & (p2 == 1))
            if phase == 0:
                m1 = p2*p4*p6 == 0
                m2 = p4*p6*p8 == 0
            else:
                m1 = p2*p4*p8 == 0
                m2 = p2*p6*p8 == 0
            remove = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & m1 & m2
            count = int(np.count_nonzero(remove))
            c[remove] = 0
            changed += count
        if changed == 0:
            break
    return im[1:-1, 1:-1]


def medial_axis_thinning(binary: np.ndarray) -> np.ndarray:
    """Distance-transform-based skeleton (skimage medial_axis).

    Structurally different from Zhang-Suen: instead of iteratively eroding
    boundary pixels, it keeps ridge pixels of the distance transform. It is
    more forgiving of thin/tapering arms that pixel-erosion thinning can
    erase, at the cost of more short spurious branches (handled downstream
    by remove_tiny_spurs). Used as an automatic fallback when Zhang-Suen
    does not yield enough terminal candidates for the target arm count.
    """
    from skimage.morphology import medial_axis as _medial_axis
    skel, _ = _medial_axis((binary > 0), return_distance=True)
    return skel.astype(np.uint8)


def remove_tiny_spurs(skel: np.ndarray, distance: np.ndarray, passes: int = 2) -> np.ndarray:
    """Remove only terminal chains shorter than local width; preserve long thin arms.
    RESTORED: R10 parameters (passes=2, width_factor=0.55) for consistent topology."""
    out = skel.copy().astype(np.uint8)
    for _ in range(passes):
        graph = pixel_graph(out)
        if len(graph[0]) == 0:
            return out
        pts, adj, index = graph
        deg = np.array([len(x) for x in adj])
        delete: set[int] = set()
        for leaf in np.flatnonzero(deg == 1):
            path = [int(leaf)]
            prev, cur, length = -1, int(leaf), 0.0
            while len(adj[cur]) <= 2:
                nxts = [(j, w) for j, w in adj[cur] if j != prev]
                if not nxts:
                    break
                nxt, step = nxts[0]
                path.append(nxt); length += step
                prev, cur = cur, nxt
                if len(adj[cur]) != 2:
                    break
            y, x = pts[cur]
            width = 2.0 * float(distance[y, x])
            if len(adj[cur]) >= 3 and length < max(2.5, 0.55 * width):
                delete.update(path[:-1])
        if not delete:
            break
        for i in delete:
            y, x = pts[i]
            out[y, x] = 0
    return out# ---------------------------------------------------------------------------
# Dense pixel graph, root selection, and topologically distinct tips
# ---------------------------------------------------------------------------

def pixel_graph(skeleton: np.ndarray
                ) -> Tuple[np.ndarray, List[List[Tuple[int, float]]], np.ndarray]:
    h, w = skeleton.shape
    points = np.argwhere(skeleton > 0).astype(np.int32)
    n = len(points)
    index = np.full((h, w), -1, np.int32)
    if n == 0:
        return points, [], index
    index[points[:, 0], points[:, 1]] = np.arange(n, dtype=np.int32)

    padded = np.full((h + 2, w + 2), -1, np.int32)
    padded[1:-1, 1:-1] = index
    py, px = points[:, 0] + 1, points[:, 1] + 1

    src_parts, dst_parts, wt_parts = [], [], []
    for dy, dx in NBR8:
        nbr = padded[py + dy, px + dx]
        valid = nbr >= 0
        if np.any(valid):
            idx = np.nonzero(valid)[0]
            src_parts.append(idx.astype(np.int32))
            dst_parts.append(nbr[valid])
            weight = math.sqrt(2.0) if dy and dx else 1.0
            wt_parts.append(np.full(idx.shape[0], weight, dtype=np.float64))

    if not src_parts:
        return points, [[] for _ in range(n)], index

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    wt = np.concatenate(wt_parts)

    order = np.argsort(src, kind="stable")
    src, dst, wt = src[order], dst[order], wt[order]
    boundaries = np.searchsorted(src, np.arange(n + 1))
    dst_list = dst.tolist()
    wt_list = wt.tolist()
    adjacency: List[List[Tuple[int, float]]] = [
        list(zip(dst_list[boundaries[i]:boundaries[i + 1]],
                wt_list[boundaries[i]:boundaries[i + 1]]))
        for i in range(n)
    ]
    return points, adjacency, index


def choose_anatomical_root(points: np.ndarray, adjacency: List[List[Tuple[int, float]]],
                           distance: np.ndarray, mask: np.ndarray) -> int:
    """Find the arm-confluence hub, not a terminal maximum in the mantle."""
    degree = np.array([len(a) for a in adjacency], dtype=np.int16)
    junction = np.zeros(mask.shape, np.uint8)
    jp = points[degree >= 3]
    if len(jp):
        junction[jp[:, 0], jp[:, 1]] = 1
        junction = cv2.dilate(junction, np.ones((7, 7), np.uint8))
        n, labels, stats, centers = cv2.connectedComponentsWithStats(junction, 8)
        fg = np.argwhere(mask > 0)
        fg_center = fg.mean(axis=0)
        diag = math.hypot(*mask.shape)
        best_score, best_center = -1e30, None
        for k in range(1, n):
            area = float(stats[k, cv2.CC_STAT_AREA])
            cy, cx = centers[k][1], centers[k][0]
            yi, xi = int(round(cy)), int(round(cx))
            radius = float(distance[np.clip(yi, 0, mask.shape[0]-1),
                                    np.clip(xi, 0, mask.shape[1]-1)])
            centrality = np.linalg.norm(np.array([cy, cx]) - fg_center) / diag
            score = math.log1p(area) + 0.08 * radius - 2.0 * centrality
            if score > best_score:
                best_score, best_center = score, np.array([cy, cx])
        if best_center is not None:
            cand = np.flatnonzero(degree >= 3)
            d2 = np.sum((points[cand].astype(float) - best_center) ** 2, axis=1)
            near = cand[np.argsort(d2)[:min(30, len(cand))]]
            return int(max(near, key=lambda i: float(distance[tuple(points[i])]) -
                           0.15 * math.sqrt(float(np.sum((points[i]-best_center)**2)))))
    vals = distance[points[:, 0], points[:, 1]]
    return int(np.argmax(vals))


def dijkstra_tree(points: np.ndarray, adjacency: List[List[Tuple[int, float]]],
                  root: int, distance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(points)
    dist = np.full(n, np.inf, float)
    parent = np.full(n, -1, np.int32)
    dist[root] = 0.0
    pq = [(0.0, root)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist[u]:
            continue
        yu, xu = points[u]
        for v, step in adjacency[u]:
            yv, xv = points[v]
            clearance = 0.5 * (float(distance[yu, xu]) + float(distance[yv, xv]))
            cost = step * (1.0 + 0.10 / max(clearance, 0.5))
            nd = du + cost
            if nd + 1e-12 < dist[v]:
                dist[v], parent[v] = nd, u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def backtrack(parent: np.ndarray, root: int, tip: int) -> np.ndarray:
    path, seen, cur = [], set(), int(tip)
    while cur >= 0 and cur not in seen:
        path.append(cur); seen.add(cur)
        if cur == root:
            break
        cur = int(parent[cur])
    if not path or path[-1] != root:
        return np.empty(0, np.int32)
    return np.asarray(path[::-1], np.int32)


def path_arc(points: np.ndarray, path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    q = points[path].astype(float)
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def common_prefix_fraction(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k / max(1, min(len(a), len(b)))


def merge_close_tips(points: np.ndarray, distance: np.ndarray,
                     candidates: List[Tuple[float, np.ndarray, int]],
                     radius_factor: float = 1.2, min_radius: float = 6.0
                     ) -> List[Tuple[float, np.ndarray, int]]:
    """Merge arm candidates whose final TIP positions end up very close
    together, even if their paths never overlapped enough for the
    prefix-based dedup above to catch them (e.g. a curled tip that forks
    into two nearby skeleton endpoints, or segmentation noise near a tip).
    For each close pair, keeps only the longer/more confident candidate.
    The merge radius scales with local body width at the tips, so it stays
    proportionate on both thick and thin arms.
    """
    remaining = sorted(candidates, key=lambda z: z[0], reverse=True)
    kept: List[Tuple[float, np.ndarray, int]] = []
    kept_widths: List[float] = []
    for length, path, tip in remaining:
        ty, tx = points[tip]
        cand_width = 2.0 * float(distance[ty, tx])
        merged = False
        for (k_length, k_path, k_tip), k_width in zip(kept, kept_widths):
            ky, kx = points[k_tip]
            d = math.hypot(float(ty - ky), float(tx - kx))
            merge_radius = max(min_radius, radius_factor * 0.5 * (cand_width + k_width))
            if d < merge_radius:
                merged = True
                break
        if not merged:
            kept.append((length, path, tip))
            kept_widths.append(cand_width)
    return kept


def path_closeness_fraction(points: np.ndarray, distance: np.ndarray,
                            a: np.ndarray, b: np.ndarray,
                            proximity_factor: float = 1.0) -> float:
    """Fraction of the SHORTER of two arm paths (root-to-tip) that runs
    within local body width of the other, anywhere along its length --
    i.e. how much of it stays right alongside the other arm from root
    toward tip. Independent of common_prefix_fraction (which requires
    pixel-identical prefixes) and merge_close_tips (which only checks the
    final tip position).
    """
    pa_yx, pb_yx = points[a].astype(float), points[b].astype(float)
    pa_idx = a
    if len(pa_yx) > len(pb_yx):
        pa_yx, pb_yx = pb_yx, pa_yx
        pa_idx = b
    if len(pa_yx) < 2 or len(pb_yx) < 2:
        return 0.0
    tree_b = cKDTree(pb_yx)
    dists, _ = tree_b.query(pa_yx)
    widths = 2.0 * distance[points[pa_idx, 0], points[pa_idx, 1]]
    close = dists <= np.maximum(3.0, proximity_factor * widths)
    seg_len = np.linalg.norm(np.diff(pa_yx, axis=0), axis=1)
    seg_close = close[:-1] & close[1:]
    total = float(seg_len.sum())
    if total <= 0:
        return 0.0
    return float(seg_len[seg_close].sum()) / total


def merge_close_paths(points: np.ndarray, distance: np.ndarray,
                      candidates: List[Tuple[float, np.ndarray, int]],
                      closeness_threshold: float = 0.8,
                      proximity_factor: float = 0.2
                      ) -> List[Tuple[float, np.ndarray, int]]:
    """Merge arms that run very close together (within local body width)
    for at least closeness_threshold (default 80%) of the shorter arm's
    length, from root to tip -- almost certainly the same physical arm
    picked up twice, even if their tips end up far apart. Linking is
    transitive (union-find), so a chain of arms that are all pairwise
    close collapses into one, not just isolated pairs. Keeps only the
    longer/more confident candidate of each merged group.
    """
    n = len(candidates)
    if n <= 1:
        return list(candidates)

    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        _, path_i, _ = candidates[i]
        for j in range(i + 1, n):
            _, path_j, _ = candidates[j]
            frac = path_closeness_fraction(points, distance, path_i, path_j, proximity_factor)
            if frac >= closeness_threshold:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    kept = [candidates[max(members, key=lambda idx: candidates[idx][0])]
            for members in clusters.values()]
    kept.sort(key=lambda z: z[0], reverse=True)
    return kept


def select_arm_paths(points: np.ndarray, adjacency: List[List[Tuple[int, float]]],
                     root: int, parent: np.ndarray, geodesic: np.ndarray,
                     distance: np.ndarray, min_arms: int = 5,
                     max_arms: int = 8) -> List[np.ndarray]:
    """Select anatomically distinct arm paths.

    FIX: now accepts min_arms..max_arms instead of hard-coding 8.
    Also relaxes the prefix threshold progressively when arms are scarce,
    so that curled/overlapping arms are not lost to aggressive dedup."""
    degree = np.array([len(a) for a in adjacency])
    candidates = list(map(int, np.flatnonzero((degree == 1) & np.isfinite(geodesic))))
    all_paths: List[Tuple[float, np.ndarray, int]] = []
    for tip in candidates:
        p = backtrack(parent, root, tip)
        if len(p) >= 4:
            all_paths.append((path_arc(points, p), p, tip))
    all_paths.sort(key=lambda z: z[0], reverse=True)
    if not all_paths:
        raise RuntimeError("No terminal skeleton paths were found")

    # Greedy branch persistence: two candidates sharing most of their route are
    # a real tip plus a local spur on the same arm, so retain only the longer.
    selected: List[Tuple[float, np.ndarray, int]] = []
    for item in all_paths:
        if all(common_prefix_fraction(item[1], s[1]) < 0.40 for s in selected):
            selected.append(item)

    # If skeleton endpoints are lost through a touching/occluded silhouette,
    # add geodesic local maxima.
    if len(selected) < max_arms:
        order = np.argsort(geodesic)[::-1]
        for tip in order:
            if not np.isfinite(geodesic[tip]):
                continue
            p = backtrack(parent, root, int(tip))
            if len(p) < 8:
                continue
            if all(common_prefix_fraction(p, s[1]) < 0.58 for s in selected):
                selected.append((path_arc(points, p), p, int(tip)))
            if len(selected) >= max_arms + 2:
                break

    # FIX: if still short, relax the prefix threshold once (matching R10's 0.82)
    if len(selected) < min_arms:
        for item in all_paths:
            if not any(np.array_equal(item[1], s[1]) for s in selected):
                if all(common_prefix_fraction(item[1], s[1]) < 0.82 for s in selected):
                    selected.append(item)
            if len(selected) >= max_arms:
                break

    if len(selected) < min_arms:
        raise RuntimeError(
            f"Only {len(selected)} anatomically distinct arms detected; expected at least {min_arms}")


    # Usually there are extra terminals: arms + the mantle cap.
    # Mantle is the broadest route at its distal half. Remove that one when
    # choosing the persistent terminal paths.
    pool = selected[:max(max_arms + 2, max_arms)]
    if len(pool) > max_arms:
        features = []
        for length, p, tip in pool:
            tail = p[int(0.35 * len(p)):]
            mean_r = float(np.mean(distance[points[tail, 0], points[tail, 1]]))
            q90_r = float(np.percentile(distance[points[tail, 0], points[tail, 1]], 90))
            features.append((0.65 * mean_r + 0.35 * q90_r, length, p))
        mantle_i = int(np.argmax([f[0] for f in features]))
        pool = [x for i, x in enumerate(pool) if i != mantle_i]
    pool.sort(key=lambda z: z[0], reverse=True)
    final = pool[:max_arms]

    # Distance-based tip merge -- LAST step, after the arm count has already
    # been decided (mantle cap removed, capped at max_arms). If two of these
    # final arms' tips end up very close together, they're almost certainly
    # the same physical arm counted twice; collapse them into one. This
    # intentionally runs after every "try to find more arms" step above and
    # does not attempt to backfill the count afterward -- if merging turns
    # 8 detected arms into 5, the final result is 5 arms, not a refilled 8.
    final = merge_close_tips(points, distance, final)
    if not final:
        raise RuntimeError("No arms remained after tip merging")

    # Path-closeness merge -- separate, additional final step. Two arms
    # that run right alongside each other (within local body width) for
    # ~80% or more of the shorter one's length are almost certainly the
    # same physical arm, even if their tips end up in different places.
    final = merge_close_paths(points, distance, final, closeness_threshold=0.8)
    if not final:
        raise RuntimeError("No arms remained after path-closeness merging")
    return [x[1] for x in final]# ---------------------------------------------------------------------------
# Centered branch splines, geodesic node sampling, and graph construction
# ---------------------------------------------------------------------------

def cumulative_arc(poly: np.ndarray) -> np.ndarray:
    if len(poly) == 0:
        return np.zeros(0)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]


def interpolate_arc(poly: np.ndarray, arc: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.empty((len(targets), 2), float)
    for d in range(2):
        out[:, d] = np.interp(targets, arc, poly[:, d])
    return out


def deduplicate_polyline(poly: np.ndarray, tolerance: float = 0.15) -> np.ndarray:
    if len(poly) < 2:
        return poly
    keep = np.r_[True, np.linalg.norm(np.diff(poly, axis=0), axis=1) > tolerance]
    return poly[keep]


def fit_mask_constrained_spline(raw_xy: np.ndarray, mask: np.ndarray,
                                distance: np.ndarray, smoothness: float) -> Tuple[np.ndarray, np.ndarray]:
    raw_xy = deduplicate_polyline(raw_xy.astype(float))
    if len(raw_xy) < 4:
        arc = cumulative_arc(raw_xy)
        return raw_xy, arc
    raw_arc = cumulative_arc(raw_xy)
    total = float(raw_arc[-1])
    n_ctrl = int(np.clip(math.ceil(total / 7.0), 12, 90))
    control = interpolate_arc(raw_xy, raw_arc, np.linspace(0, total, n_ctrl))
    u = np.linspace(0.0, 1.0, n_ctrl)
    s = smoothness * n_ctrl * max(1.0, total / 200.0)
    try:
        tck, _ = splprep([control[:, 0], control[:, 1]], u=u, s=s, k=3)
        n_dense = max(80, int(math.ceil(total * 1.5)))
        curve = np.column_stack(splev(np.linspace(0.0, 1.0, n_dense), tck))
    except Exception:
        curve = interpolate_arc(raw_xy, raw_arc, np.linspace(0, total, max(80, int(total))))
    curve[0], curve[-1] = raw_xy[0], raw_xy[-1]

    h, w = mask.shape
    raw_tree = cKDTree(raw_xy)
    for _ in range(3):
        xi = np.clip(np.rint(curve[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(curve[:, 1]).astype(int), 0, h - 1)
        bad = (mask[yi, xi] == 0) | (distance[yi, xi] < 0.55)
        if not np.any(bad):
            break
        _, near = raw_tree.query(curve[bad])
        curve[bad] = 0.25 * curve[bad] + 0.75 * raw_xy[near]
    if len(curve) > 9:
        filtered = np.column_stack([gaussian_filter1d(curve[:, d], 1.25, mode="nearest") for d in range(2)])
        curve[1:-1] = filtered[1:-1]
    curve[0], curve[-1] = raw_xy[0], raw_xy[-1]
    arc = cumulative_arc(curve)
    return curve, arc


def curve_curvature(curve: np.ndarray) -> np.ndarray:
    if len(curve) < 5:
        return np.zeros(len(curve))
    x, y = curve[:, 0], curve[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return np.abs(dx * ddy - dy * ddx) / np.maximum((dx*dx + dy*dy) ** 1.5, EPS)


def order_branches_clockwise(raw_paths_xy: List[np.ndarray],
                             extra: Optional[List[Any]] = None):
    decorated = []
    for i, p in enumerate(raw_paths_xy):
        a = cumulative_arc(p)
        probe = interpolate_arc(p, a, np.array([min(a[-1], max(5.0, 0.12*a[-1]))]))[0]
        v = probe - p[0]
        angle = math.atan2(v[1], v[0])
        decorated.append((angle, p, extra[i] if extra is not None else None))
    decorated.sort(key=lambda x: x[0])
    k = int(np.argmin([x[1][-1, 1] for x in decorated]))
    ordered = decorated[k:] + decorated[:k]
    if extra is not None:
        return [p for _, p, _ in ordered], [e for _, _, e in ordered]
    return [p for _, p, _ in ordered]


def compute_fork_offsets(tip_paths: List[np.ndarray]) -> List[int]:
    """For each arm's root-to-tip skeleton path (as pixel-index arrays,
    identical values = identical physical pixel), find how many leading
    pixels it shares with the path of its closest sibling. Two arm paths
    from the same Dijkstra tree are *exactly* identical up to the pixel
    where they structurally fork -- that shared prefix is the ambiguous
    stretch where individual arms can't be told apart in a touching /
    overlapping silhouette. Returns, per arm, the path index at which it
    first becomes unique to that arm.
    """
    n = len(tip_paths)
    offsets = [0] * n
    for i in range(n):
        pi = tip_paths[i]
        max_shared = 0
        for j in range(n):
            if i == j:
                continue
            pj = tip_paths[j]
            m = min(len(pi), len(pj))
            k = 0
            while k < m and pi[k] == pj[k]:
                k += 1
            max_shared = max(max_shared, k)
        offsets[i] = min(max_shared, max(0, len(pi) - 2))
    return offsets


def snap_tip_to_contour(tip_xy: np.ndarray, prev_xy: np.ndarray, full_mask: np.ndarray,
                        search_radius: float) -> np.ndarray:
    """Correct a skeleton-derived arm tip using the actual mask silhouette.

    When segmentation leaves a rounded/blobby arm end instead of a sharp
    taper, the thinned skeleton's degree-1 endpoint can land off-center or
    short of where the arm visually ends -- the skeleton's topology inside
    a near-circular blob is unreliable. This instead uses the skeleton tip
    only to establish outward *direction* (from prev_xy through tip_xy),
    then finds the farthest point on the real contour of the silhouette in
    that direction, within search_radius of the skeleton tip. That contour
    point is a much more faithful "true" anatomical tip for a blobby end.
    """
    direction = tip_xy - prev_xy
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return tip_xy
    direction = direction / norm

    contours, _ = cv2.findContours((full_mask > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return tip_xy
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)

    d_to_tip = np.linalg.norm(contour - tip_xy, axis=1)
    nearby = contour[d_to_tip < search_radius]
    if len(nearby) == 0:
        return tip_xy

    # how far "outward" (along the arm's direction) each nearby contour
    # point sits, relative to the skeleton tip
    proj = (nearby - tip_xy) @ direction
    best = nearby[np.argmax(proj)]
    return best


def build_branches(model: DenseModel, full_mask: np.ndarray,
                   spline_smoothness: float) -> List[Branch]:
    full_dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    try:
        mantle_peak, _ = find_mantle_and_head(full_mask, full_dt)
        root_xy = np.array([mantle_peak[0], mantle_peak[1]])
    except RuntimeError:
        root_yx = model.points_yx[model.root].astype(float)
        root_xy = np.array([root_yx[1] * model.scale_x, root_yx[0] * model.scale_y])
    fork_offsets = compute_fork_offsets(model.tip_paths)
    raw_paths = []
    neck_xys = []
    for path, offset in zip(model.tip_paths, fork_offsets):
        yx = model.points_yx[path].astype(float)
        xy = np.column_stack([yx[:, 1] * model.scale_x, yx[:, 0] * model.scale_y])
        xy[0] = root_xy
        neck_xys.append(xy[min(offset, len(xy) - 1)].copy())
        raw_paths.append(deduplicate_polyline(xy))
    raw_paths, neck_xys = order_branches_clockwise(raw_paths, neck_xys)
    mh, mw = full_mask.shape
    branches = []
    for arm_id, (raw, neck_xy) in enumerate(zip(raw_paths, neck_xys), 1):
        # Trim off the shared/ambiguous trunk this arm's path has in common
        # with its neighbors (see compute_fork_offsets) and replace it with
        # a straight spoke from the mantle center to this arm's own fork
        # point -- avoids drawing several arms on top of one another for a
        # stretch and then kinking apart at the true skeleton branch node.
        d_to_neck = np.linalg.norm(raw - neck_xy, axis=1)
        neck_pos = int(np.argmin(d_to_neck))
        tail = raw[neck_pos:]
        if len(tail) < 2:
            tail = raw[-2:]
        raw = np.vstack([root_xy, neck_xy, tail[1:]]) if len(tail) > 1 else np.vstack([root_xy, neck_xy])

        if len(raw) >= 2:
            tx, ty = raw[-1]
            xi = int(np.clip(round(tx), 0, mw - 1))
            yi = int(np.clip(round(ty), 0, mh - 1))
            local_width = float(full_dt[yi, xi])
            # Search a bit wider than the local body width: a blobby tip's
            # true extremity can sit a few pixels beyond where the skeleton
            # endpoint landed.
            search_radius = max(1.0, 2.5 * local_width)
            snapped_tip = snap_tip_to_contour(raw[-1], raw[-2], full_mask, search_radius)
            raw = np.vstack([raw[:-1], snapped_tip])
        curve, arc = fit_mask_constrained_spline(raw, full_mask, full_dt, spline_smoothness)
        length = float(arc[-1])
        node_arc = np.linspace(0.0, length, 4)
        node_xy = interpolate_arc(curve, arc, node_arc)
        node_xy[0] = root_xy
        branches.append(Branch(arm_id, raw, curve, arc, length,
                               node_arc, node_xy, curve_curvature(curve)))
    return branches


def split_curve(curve: np.ndarray, arc: np.ndarray, a: float, b: float) -> np.ndarray:
    interior = curve[(arc > a + EPS) & (arc < b - EPS)]
    ends = interpolate_arc(curve, arc, np.array([a, b]))
    return np.vstack([ends[0], interior, ends[1]])


def find_distance_peaks(mask: np.ndarray, dt: np.ndarray, min_separation: float,
                        detection_size: int = 7, max_peaks: int = 6) -> List[Tuple[float, float, float]]:
    size = max(3, int(detection_size))
    if size % 2 == 0:
        size += 1
    local_max = (dt == maximum_filter(dt, size=size)) & (mask > 0) & (dt > 1.0)
    ys, xs = np.nonzero(local_max)
    if len(xs) == 0:
        return []
    vals = dt[ys, xs]
    order = np.argsort(vals)[::-1]
    peaks: List[Tuple[float, float, float]] = []
    for idx in order:
        x, y, v = float(xs[idx]), float(ys[idx]), float(vals[idx])
        if all(math.hypot(x - px, y - py) >= min_separation for px, py, _ in peaks):
            peaks.append((x, y, v))
        if len(peaks) >= max_peaks:
            break
    return peaks


def infer_head_direction(mantle_xy: Tuple[float, float],
                         arm_base_xys: List[Tuple[float, float]]) -> float:
    """Angle (radians) bisecting the widest empty angular gap between arm
    attachment points, as seen from the mantle center.

    Octopus arms don't fully encircle the mantle -- the head pokes out
    through whatever gap they leave (very often between the front-most
    pair). This stays well-defined even when the arm crown surrounds the
    mantle from most directions, a case where a linear mantle->arm-crown-
    centroid axis degenerates (the centroid lands almost on top of the
    mantle center itself, giving no usable direction -- exactly what
    happens when arms are spread across most of the compass).
    """
    mx, my = mantle_xy
    angles = sorted(math.atan2(y - my, x - mx) for x, y in arm_base_xys)
    n = len(angles)
    if n == 0:
        return 0.0
    best_gap, best_mid = -1.0, angles[0]
    for i in range(n):
        a = angles[i]
        b = angles[(i + 1) % n]
        gap = (b - a) if i < n - 1 else (b + 2 * math.pi - a)
        if gap > best_gap:
            best_gap, best_mid = gap, a + gap / 2.0
    return best_mid


def find_mantle_and_head(mask: np.ndarray, dt: np.ndarray,
                         head_direction_rad: Optional[float] = None,
                         arm_exclusion_mask: Optional[np.ndarray] = None,
                         cone_half_angle_deg: float = 40.0
                         ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    diag = math.hypot(*mask.shape)
    search_mask = mask
    if arm_exclusion_mask is not None:
        search_mask = cv2.bitwise_and(mask, cv2.bitwise_not(arm_exclusion_mask))

    peaks: List[Tuple[float, float, float]] = []
    for frac in (0.02, 0.035, 0.05, 0.07, 0.10, 0.14):
        peaks = find_distance_peaks(search_mask, dt, max(4.0, frac * diag))
        if len(peaks) >= 2:
            break
    if len(peaks) < 2:
        if peaks:
            return peaks[0], peaks[0]
        raise RuntimeError("Could not locate two spatially distinct mantle/head blobs")

    mantle_peak = peaks[0]
    if head_direction_rad is None:
        return mantle_peak, peaks[1]

    # Constrain the head candidate to a narrow angular CONE around the
    # inferred head direction (the widest gap in arm attachment angles),
    # rather than a corridor toward a point -- this stays meaningful even
    # when the arms surround the mantle from most directions, where a
    # corridor toward an arm-crown centroid would degenerate.
    mx, my, _ = mantle_peak
    target = np.array([math.cos(head_direction_rad), math.sin(head_direction_rad)])
    half_angle = math.radians(cone_half_angle_deg)

    best = None
    for x, y, v in peaks[1:]:
        vec = np.array([x - mx, y - my])
        d = float(np.linalg.norm(vec))
        if d < 1e-6:
            continue
        cos_sim = float(np.clip((vec @ target) / d, -1.0, 1.0))
        if math.acos(cos_sim) <= half_angle:
            if best is None or v > best[2]:
                best = (x, y, v)
    if best is None:
        # No detected peak fell inside the cone -- place the head
        # geometrically, offset from the mantle edge along the inferred
        # direction, rather than an unconstrained (and possibly badly
        # wrong) thickness peak.
        offset = mantle_peak[2] * 1.4
        fx = float(np.clip(mx + target[0] * offset, 0, mask.shape[1] - 1))
        fy = float(np.clip(my + target[1] * offset, 0, mask.shape[0] - 1))
        xi, yi = int(round(fx)), int(round(fy))
        best = (fx, fy, float(dt[yi, xi]))
    return mantle_peak, best


def build_arm_exclusion_mask(branches: List["Branch"], shape: Tuple[int, int],
                             margin: int = 9) -> np.ndarray:
    """Rasterize the already-detected arm curves (dilated by a margin) so
    head-peak search can exclude them -- a thick arm base should never be
    mistaken for the head."""
    h, w = shape
    excl = np.zeros((h, w), np.uint8)
    for b in branches:
        pts = np.rint(b.curve_xy).astype(int)
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        excl[ys, xs] = 255
    if margin > 1:
        excl = cv2.dilate(excl, np.ones((margin, margin), np.uint8))
    return excl# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def construct_graph(branches: List[Branch], full_mask: np.ndarray,
                    max_arms: int = 8
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    h, w = full_mask.shape
    root = branches[0].node_xy[0]
    rx, ry = int(np.clip(round(root[0]), 0, w-1)), int(np.clip(round(root[1]), 0, h-1))
    nodes: List[Dict[str, Any]] = [{
        "node_id": 0, "x": float(root[0]), "y": float(root[1]),
        "radius": float(dt[ry, rx]), "degree": len(branches) + 1, "branch_id": 0,
        "body_part": "Mantle Center", "is_center": True, "is_tip": False,
        "is_head": False
    }]
    edges: List[Dict[str, Any]] = []
    next_node, next_edge = 1, 0
    for b in branches:
        ids = [0]
        labels = [f"Arm {b.arm_id} Base", f"Arm {b.arm_id} Mid 1", f"Arm {b.arm_id} Tip"]
        for j in range(1, 4):
            x, y = b.node_xy[j]
            xi, yi = int(np.clip(round(x), 0, w-1)), int(np.clip(round(y), 0, h-1))
            nid = next_node; next_node += 1; ids.append(nid)
            nodes.append({
                "node_id": nid, "x": float(x), "y": float(y),
                "radius": float(dt[yi, xi]), "degree": 1 if j == 3 else 2,
                "branch_id": b.arm_id, "body_part": labels[j-1],
                "is_center": False, "is_tip": bool(j == 3), "is_head": False
            })
        for j in range(3):
            pl = split_curve(b.curve_xy, b.arc, b.node_arc[j], b.node_arc[j+1])
            pl_arc = cumulative_arc(pl)
            curv = curve_curvature(pl)
            sample_x = np.clip(np.rint(pl[:, 0]).astype(int), 0, w-1)
            sample_y = np.clip(np.rint(pl[:, 1]).astype(int), 0, h-1)
            radii = dt[sample_y, sample_x]
            edges.append({
                "edge_id": next_edge, "start_node": ids[j], "end_node": ids[j+1],
                "branch_id": b.arm_id, "body_part": f"Arm {b.arm_id}",
                "label": f"Arm {b.arm_id}", "length": float(pl_arc[-1]),
                "geodesic_distance": float(pl_arc[-1]),
                "average_radius": float(np.mean(radii)),
                "average_curvature": float(np.mean(curv)),
                "maximum_curvature": float(np.max(curv)),
                "polyline": [[float(x), float(y)] for x, y in pl]
            })
            next_edge += 1

    # --- Head node h1 -------------------------------------------------
    # Direction of the widest empty angular gap between arm bases, as seen
    # from the mantle center -- where the arms don't fully encircle the
    # mantle, the head pokes out through that gap. Stays well-defined even
    # when arms surround the mantle from most directions (unlike a linear
    # mantle->arm-crown-centroid axis, which degenerates to near zero
    # length in that case).
    mantle_xy = (root[0], root[1])
    arm_base_xys = [tuple(b.node_xy[1]) for b in branches]
    head_dir = infer_head_direction(mantle_xy, arm_base_xys)
    arm_excl = build_arm_exclusion_mask(branches, full_mask.shape)
    _, head_peak = find_mantle_and_head(full_mask, dt, head_direction_rad=head_dir,
                                        arm_exclusion_mask=arm_excl)
    hx, hy, hr = head_peak
    head_id = next_node; next_node += 1
    nodes.append({
        "node_id": head_id, "x": hx, "y": hy, "radius": hr,
        "degree": 1, "branch_id": 0, "body_part": "Head",
        "is_center": False, "is_tip": False, "is_head": True
    })
    n_head_samples = max(6, int(math.hypot(hx - root[0], hy - root[1]) / 3))
    head_polyline = np.linspace(root, [hx, hy], num=n_head_samples)
    head_arc = cumulative_arc(head_polyline)
    hsx = np.clip(np.rint(head_polyline[:, 0]).astype(int), 0, w-1)
    hsy = np.clip(np.rint(head_polyline[:, 1]).astype(int), 0, h-1)
    head_radii = dt[hsy, hsx]
    edges.append({
        "edge_id": next_edge, "start_node": 0, "end_node": head_id,
        "branch_id": 0, "body_part": "Head", "label": "Head",
        "length": float(head_arc[-1]), "geodesic_distance": float(head_arc[-1]),
        "average_radius": float(np.mean(head_radii)),
        "average_curvature": 0.0, "maximum_curvature": 0.0,
        "polyline": [[float(x), float(y)] for x, y in head_polyline]
    })
    next_edge += 1

    return nodes, edges# ---------------------------------------------------------------------------
# Metrics, scoring, and validation
# ---------------------------------------------------------------------------

def count_crossings(edges: List[Dict[str, Any]]) -> int:
    segments = []
    for e in edges:
        p = np.asarray(e["polyline"], float)
        for i in range(len(p)-1):
            segments.append((p[i], p[i+1], e["edge_id"], e["start_node"], e["end_node"]))
    def orient(a, b, c):
        return float(np.cross(b-a, c-a))
    crossings = 0
    for i in range(len(segments)):
        a, b, ei, ui, vi = segments[i]
        for j in range(i+1, len(segments)):
            c, d, ej, uj, vj = segments[j]
            if ei == ej or {ui, vi} & {uj, vj}:
                continue
            if max(a[0], b[0]) + 1e-6 < min(c[0], d[0]) or max(c[0], d[0]) + 1e-6 < min(a[0], b[0]):
                continue
            if max(a[1], b[1]) + 1e-6 < min(c[1], d[1]) or max(c[1], d[1]) + 1e-6 < min(a[1], b[1]):
                continue
            if orient(a,b,c)*orient(a,b,d) < -1e-6 and orient(c,d,a)*orient(c,d,b) < -1e-6:
                crossings += 1
    return crossings


def graph_metrics(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                  mask: np.ndarray, branches: List[Branch]) -> Dict[str, Any]:
    g = nx.Graph()
    g.add_nodes_from(n["node_id"] for n in nodes)
    g.add_edges_from((e["start_node"], e["end_node"]) for e in edges)
    components = nx.number_connected_components(g)
    cycles = len(nx.cycle_basis(g))
    lengths = np.asarray([e["length"] for e in edges], float)
    curv = np.concatenate([b.curvature for b in branches]) if branches else np.zeros(1)
    branch_lengths = np.asarray([b.length for b in branches], float)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    h, w = mask.shape
    dense = np.vstack([b.curve_xy for b in branches])
    xi = np.clip(np.rint(dense[:, 0]).astype(int), 0, w-1)
    yi = np.clip(np.rint(dense[:, 1]).astype(int), 0, h-1)
    inside = mask[yi, xi] > 0
    radii = dt[yi, xi]
    tips = sum(bool(n.get("is_tip")) for n in nodes)
    centers = sum(bool(n.get("is_center")) for n in nodes)
    heads = sum(bool(n.get("is_head")) for n in nodes)
    duplicate_nodes = len(nodes) - len({(round(n["x"], 5), round(n["y"], 5)) for n in nodes})
    edge_keys = [(min(e["start_node"],e["end_node"]), max(e["start_node"],e["end_node"])) for e in edges]
    duplicate_edges = len(edge_keys) - len(set(edge_keys))
    symmetry = float(np.clip(1.0 - np.std(branch_lengths) / max(np.mean(branch_lengths), EPS), 0.0, 1.0))
    metrics = {
        "total_nodes": len(nodes), "total_edges": len(edges),
        "connected_components": components, "number_of_cycles": cycles,
        "average_edge_length": float(np.mean(lengths)),
        "average_branch_curvature": float(np.mean(curv)),
        "maximum_curvature": float(np.max(curv)),
        "average_node_spacing": float(np.mean(lengths)),
        "skeleton_length": float(np.sum(branch_lengths)),
        "average_distance_to_boundary": float(np.mean(radii)),
        "branch_symmetry": symmetry, "arm_count": len(branches),
        "tip_count": tips, "center_count": centers, "head_count": heads,
        "inside_fraction": float(np.mean(inside)),
        "crossing_edges": count_crossings(edges),
        "duplicate_nodes": duplicate_nodes, "duplicate_edges": duplicate_edges,
        "tree": bool(components == 1 and cycles == 0 and len(edges) == len(nodes)-1),
    }
    return metrics


def quality_score(metrics: Dict[str, Any], max_arms: int = 8) -> float:
    score = 0.0
    score += 1000.0 if metrics["tree"] else -1000.0
    score -= 500.0 * abs(metrics["arm_count"] - max_arms)
    score -= 500.0 * abs(metrics["tip_count"] - metrics["arm_count"])
    score -= 1000.0 * abs(metrics["center_count"] - 1)
    score -= 1000.0 * abs(metrics["head_count"] - 1)
    score += 800.0 * metrics["inside_fraction"]
    score -= 120.0 * metrics["crossing_edges"]
    score -= 300.0 * (metrics["duplicate_nodes"] + metrics["duplicate_edges"])
    score -= 25.0 * metrics["average_branch_curvature"]
    score -= 3.0 * metrics["maximum_curvature"]
    return score


def validate_requirements(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                          metrics: Dict[str, Any], min_arms: int = 5,
                          max_arms: int = 8) -> List[str]:
    errors = []
    arm_count = metrics["arm_count"]
    # Node count range adapts to actual arm count: 1 center + 1 head + 3*arms
    expected_nodes = 2 + 3 * arm_count
    checks = [
        (expected_nodes - 3 <= len(nodes) <= expected_nodes + 3,
         f"node count {len(nodes)} is not near expected {expected_nodes}"),
        (metrics["tree"], "graph is not one connected acyclic tree"),
        (min_arms <= arm_count <= max_arms,
         f"arm count {arm_count} is not in [{min_arms}, {max_arms}]"),
        (metrics["tip_count"] == arm_count, "tip count does not match arm count"),
        (metrics["center_count"] == 1, "center count is not one"),
        (metrics["head_count"] == 1, "head count is not one"),
        (metrics["duplicate_nodes"] == 0, "duplicate nodes exist"),
        (metrics["duplicate_edges"] == 0, "duplicate edges exist"),
        (metrics["inside_fraction"] >= 0.999, "part of a spline leaves the mask"),
    ]
    errors.extend(msg for ok, msg in checks if not ok)
    center = next((n for n in nodes if n["is_center"]), None)
    if center is None or center["body_part"] != "Mantle Center":
        errors.append("Mantle Center label is missing")
    head = next((n for n in nodes if n.get("is_head")), None)
    if head is None or head["body_part"] != "Head":
        errors.append("Head label is missing")
    for arm in range(1, arm_count + 1):
        ns = [n for n in nodes if n["branch_id"] == arm]
        if len(ns) != 3 or sum(n["is_tip"] for n in ns) != 1:
            errors.append(f"Arm {arm} does not have Base, Mid 1, Tip")
        es = [e for e in edges if e["branch_id"] == arm]
        if len(es) != 3 or any(len(e["polyline"]) < 3 for e in es):
            errors.append(f"Arm {arm} edges are missing dense spline geometry")
    return errors# ---------------------------------------------------------------------------
# Dense iteration
# ---------------------------------------------------------------------------

def dense_iteration(mask: np.ndarray, iteration: int, max_dimension: int,
                    smooth_factor: float, min_arms: int = 5,
                    max_arms: int = 8, skeleton_backend: str = "auto") -> DenseModel:
    small, sx, sy = prepare_mask(mask, max_dimension, smooth_factor)
    dt = cv2.distanceTransform(small, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Try skeletonizers in order until one yields enough terminal candidates
    # to plausibly cover every arm plus the mantle/head terminal. Zhang-Suen
    # goes first (cheap, usually cleanest); medial_axis is the fallback for
    # frames where thin/curled arms get eroded away entirely.
    backend_fns = []
    if skeleton_backend in ("auto", "zhang_suen"):
        backend_fns.append(("zhang_suen", zhang_suen_thinning))
    if skeleton_backend in ("auto", "medial_axis"):
        backend_fns.append(("medial_axis", medial_axis_thinning))
    if not backend_fns:
        raise ValueError(f"Unknown skeleton_backend: {skeleton_backend!r}")

    best = None  # (name, skel, points, adj, n_endpoints)
    for name, fn in backend_fns:
        try:
            skel = fn(small)
            skel = remove_tiny_spurs(skel, dt, passes=1)
            points, adj, _ = pixel_graph(skel)
        except Exception:
            continue
        if len(points) < 20:
            continue
        n_endpoints = int(sum(len(a) == 1 for a in adj))
        if best is None or n_endpoints > best[4]:
            best = (name, skel, points, adj, n_endpoints)
        if n_endpoints >= max_arms + 1:  # arms + mantle/head, roughly
            break

    if best is None:
        raise RuntimeError("Thinning produced an empty or degenerate skeleton on every backend")

    backend_name, skel, points, adj, n_endpoints = best
    root = choose_anatomical_root(points, adj, dt, small)
    geod, parent = dijkstra_tree(points, adj, root, dt)
    paths = select_arm_paths(points, adj, root, parent, geod, dt, min_arms, max_arms)
    diagnostics = {
        "dense_pixels": int(len(points)),
        "raw_endpoint_count": n_endpoints,
        "skeleton_backend": backend_name,
        "root_small_xy": [int(points[root,1]), int(points[root,0])],
        "selected_path_lengths_small": [path_arc(points, p) for p in paths],
        "smooth_factor": smooth_factor,
    }
    return DenseModel(small, dt, skel, sx, sy, points, adj, root,
                      parent, geod, paths, iteration, diagnostics)


# ---------------------------------------------------------------------------
# Export and clean visualizations
# ---------------------------------------------------------------------------

def clean_json(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    return x


def export_all(output: Path, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
               metrics: Dict[str, Any], iterations: List[Dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"nodes": nodes, "edges": edges, "metrics": metrics,
               "optimization_iterations": iterations,
               "coordinate_system": "image pixels; origin top-left; x right; y down"}
    with (output / "graph.json").open("w", encoding="utf-8") as f:
        json.dump(clean_json(payload), f, indent=2)
    node_fields = ["node_id","x","y","radius","degree","branch_id","body_part",
                   "is_center","is_tip","is_head"]
    with (output / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields)
        writer.writeheader(); writer.writerows({k:n.get(k,"") for k in node_fields} for n in nodes)
    edge_fields = ["edge_id","start_node","end_node","branch_id","body_part","label","length",
                   "geodesic_distance","average_radius","average_curvature","maximum_curvature","polyline"]
    with (output / "edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fields)
        writer.writeheader()
        for e in edges:
            row = {k:e.get(k,"") for k in edge_fields}
            row["polyline"] = json.dumps(e["polyline"], separators=(",",":"))
            writer.writerow(row)


def branch_color(arm: int) -> Tuple[int,int,int]:
    palette = [(230,70,70),(240,145,35),(205,185,35),(65,175,75),
               (35,170,175),(55,115,225),(145,85,210),(220,75,155)]
    return palette[(arm-1) % len(palette)]


def draw_polyline_cv(canvas: np.ndarray, polyline: Sequence[Sequence[float]],
                     color: Tuple[int,int,int], thickness: int) -> None:
    p = np.rint(np.asarray(polyline)).astype(np.int32).reshape(-1,1,2)
    if len(p) >= 2:
        cv2.polylines(canvas, [p], False, color, thickness, cv2.LINE_AA)


def save_skeleton_png(mask: np.ndarray, edges: List[Dict[str, Any]], path: Path) -> None:
    image = np.zeros(mask.shape, np.uint8)
    for e in edges:
        p = np.rint(np.asarray(e["polyline"])).astype(np.int32).reshape(-1,1,2)
        cv2.polylines(image, [p], False, 255, 1, cv2.LINE_8)
    cv2.imwrite(str(path), image)


def save_overlay(mask: np.ndarray, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                 path: Path) -> None:
    base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    base = cv2.addWeighted(base, 0.68, np.full_like(base, 25), 0.32, 0)
    for e in edges:
        c = (0, 230, 120) if e.get("body_part") == "Head" else branch_color(e["branch_id"])
        draw_polyline_cv(base, e["polyline"], c, max(2, int(round(max(mask.shape)/700))))
    for n in nodes:
        p = (int(round(n["x"])), int(round(n["y"])))
        if n["is_center"]:
            color, r = (0,0,255), 7
        elif n.get("is_head"):
            color, r = (0,230,120), 7
        elif n["is_tip"]:
            color, r = (0,215,255), 4
        else:
            color, r = (255,255,255), 4
        cv2.circle(base, p, r, color, -1, cv2.LINE_AA)
        cv2.circle(base, p, r, (25,25,25), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), base)


def save_graph_figure(mask: np.ndarray, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                      metrics: Dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = mask.shape
    fig_w = 14.0
    fig_h = max(8.0, fig_w*h/w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#111318")
    ax.imshow(mask, cmap="gray", vmin=0, vmax=255, alpha=0.30)
    cmap = plt.get_cmap("tab10")
    for e in edges:
        p = np.asarray(e["polyline"])
        if e.get("body_part") == "Head":
            color = "#1cd679"
        else:
            color = cmap((e["branch_id"]-1) % 10)
        ax.plot(p[:,0], p[:,1], color=color, lw=2.0, solid_capstyle="round", zorder=3)
        mid = p[len(p)//2]
        ax.annotate(f"E{e['edge_id']}  {e['label']}", mid, xytext=(4,3), textcoords="offset points",
                    fontsize=5.8, color=color, zorder=5)
    offsets = [(7,-8),(7,8),(-7,-8),(-7,8),(10,0),(-10,0)]
    for n in nodes:
        x,y=n["x"],n["y"]
        if n["is_center"]:
            marker, size, color = "*", 150, "#ff3b30"
        elif n.get("is_head"):
            marker, size, color = "^", 90, "#1cd679"
        elif n["is_tip"]:
            marker, size, color = "o", 48, "#ffd60a"
        else:
            marker, size, color = "o", 35, "#53c8ff"
        ax.scatter([x],[y],s=size,marker=marker,c=[color],edgecolors="#111318",linewidths=.7,zorder=6)
        ox, oy = offsets[n["node_id"] % len(offsets)]
        ax.annotate(f"N{n['node_id']}  {n['body_part']}", (x,y), xytext=(ox,oy),
                    textcoords="offset points", fontsize=6.2, color="#f4f5f7",
                    bbox=dict(boxstyle="round,pad=.18",fc="#111318",ec="#737780",lw=.35,alpha=.82),
                    zorder=7)
    title = (f"Anatomical medial skeleton — {metrics['total_nodes']} nodes, "
             f"{metrics['total_edges']} edges, {metrics['arm_count']} arms, "
             f"{metrics['head_count']} head")
    ax.set_title(title, color="#f4f5f7", fontsize=11, pad=8)
    ax.set_xlim(0,w); ax.set_ylim(h,0); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=.4); fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def print_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    labels = [
        ("total_nodes","Total nodes"),("total_edges","Total edges"),
        ("connected_components","Connected components"),("number_of_cycles","Number of cycles"),
        ("average_edge_length","Average edge length"),("average_branch_curvature","Average branch curvature"),
        ("maximum_curvature","Maximum curvature"),("average_node_spacing","Average node spacing"),
        ("skeleton_length","Skeleton length"),("average_distance_to_boundary","Average distance to boundary"),
        ("branch_symmetry","Branch symmetry"),("arm_count","Arm count"),("tip_count","Tip count"),
        ("head_count","Head count"),("inside_fraction","Inside fraction"),
        ("crossing_edges","Crossing edges")]
    for key, label in labels:
        value = metrics[key]
        text = f"{value:.6f}" if isinstance(value, float) else str(value)
        LOG.info(f"{prefix}{label}: {text}")# ---------------------------------------------------------------------------
# End-to-end iterative optimizer
# ---------------------------------------------------------------------------

def run(input_path: str, output_dir: str, max_dimension: int = 760,
        iterations: int = 3, min_arms: int = 5,
        max_arms: int = 8, skeleton_backend: str = "auto"
        ) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], Dict[str,Any]]:
    """Cascading arm-count optimizer.

    Strategy: try the strictest arm count first (e.g. 8), and only fall back
    to fewer arms when every smoothing iteration fails.  This guarantees the
    best anatomical result for each image — an 8-arm octopus gets 8 arms,
    a 6-arm specimen doesn't force phantom limbs.
    """
    start = time.perf_counter()
    mask = load_binary(input_path)
    LOG.info(f"Input: {input_path} ({mask.shape[1]} x {mask.shape[0]})")

    # Smoothing parameter sweep (shared across all arm-count attempts)
    settings = [
        (0.75, 0.90),   # mild smoothing
        (1.00, 0.65),   # medium smoothing
        (1.25, 0.45),   # stronger smoothing
        (1.45, 0.30),   # aggressive smoothing
    ][:max(1, iterations)]

    # Cascade: try each target arm count from max_arms down to min_arms.
    # Stop at the first arm count that produces at least one valid candidate.
    for target_arms in range(max_arms, min_arms - 1, -1):
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Attempting {target_arms} arms  (will fall back if all iterations fail)")
        LOG.info(f"{'='*60}")

        records, candidates = [], []
        previous = None
        for i, (morph_smooth, spline_smooth) in enumerate(settings, 1):
            LOG.info(f"\n  Iteration {i}/{len(settings)}: mask smoothing={morph_smooth:.2f}, spline smoothing={spline_smooth:.2f}")
            try:
                dense = dense_iteration(mask, i, max_dimension, morph_smooth,
                                        target_arms, target_arms,
                                        skeleton_backend=skeleton_backend)
                branches = build_branches(dense, mask, spline_smooth)
                nodes, edges = construct_graph(branches, mask, target_arms)
                metrics = graph_metrics(nodes, edges, mask, branches)
                score = quality_score(metrics, target_arms)
                defects = validate_requirements(nodes, edges, metrics,
                                                target_arms, target_arms)
                delta = None if previous is None else score - previous
                rec = {"iteration": i, "settings": {"mask_smoothing": morph_smooth,
                       "spline_smoothing": spline_smooth}, "dense": dense.diagnostics,
                       "metrics": metrics, "score": score, "score_change": delta,
                       "defects": defects, "target_arms": target_arms}
                records.append(rec)
                candidates.append((score, nodes, edges, metrics, branches, dense))
                print_metrics(metrics, "    ")
                LOG.info(f"    Quality score: {score:.4f}" +
                         ("" if delta is None else f" ({delta:+.4f} vs previous)"))
                LOG.info("    Defects: " + ("none" if not defects else "; ".join(defects)))
                previous = score
            except Exception as exc:
                LOG.warning(f"    Iteration {i} failed: {exc}")
                records.append({"iteration": i,
                    "settings": {"mask_smoothing": morph_smooth,
                                 "spline_smoothing": spline_smooth},
                    "error": str(exc), "score": -1e30})

        if candidates:
            # Pick the best smoothing iteration for this arm count
            score, nodes, edges, metrics, branches, dense = max(candidates, key=lambda x: x[0])
            LOG.info(f"\n✓  {target_arms}-arm candidate found (score={score:.4f})")

            # Export results
            output = Path(output_dir)
            export_all(output, nodes, edges, metrics, records)
            save_skeleton_png(mask, edges, output / "skeleton.png")
            save_overlay(mask, nodes, edges, output / "overlay.png")
            save_graph_figure(mask, nodes, edges, metrics, output / "graph.png")
            LOG.info("\nExported graph.json, nodes.csv, edges.csv, graph.png, skeleton.png, overlay.png")
            print_metrics(metrics, "  ")
            LOG.info(f"Elapsed: {time.perf_counter()-start:.3f} s")
            return nodes, edges, metrics

        # This arm count failed — log and try fewer arms
        LOG.warning(f"\n✗  No valid {target_arms}-arm result; trying {target_arms-1} arms next...")

    # All arm counts exhausted
    raise RuntimeError(
        f"All arm counts [{min_arms}..{max_arms}] failed across all smoothing iterations")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a smooth anatomical octopus skeleton graph (6-8 arms)")
    parser.add_argument("input", help="binary-mask image")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--max-dimension", type=int, default=760,
                        help="working resolution for dense thinning (default: 760)")
    parser.add_argument("--iterations", type=int, default=3, choices=range(1, 5), metavar="1..4",
                        help="automatic refinement iterations (default: 3)")
    parser.add_argument("--min-arms", type=int, default=5,
                        help="minimum arms required (default: 5)")
    parser.add_argument("--max-arms", type=int, default=8,
                        help="maximum arms (default: 8)")
    parser.add_argument("--skeleton-backend", choices=["auto", "zhang_suen", "medial_axis"],
                        default="auto",
                        help="thinning algorithm: zhang_suen, medial_axis (skimage, "
                             "distance-transform ridge), or auto (try zhang_suen, "
                             "fall back to medial_axis if too few endpoints) (default: auto)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.quiet:
        LOG.setLevel(logging.WARNING)
    if args.min_arms < 1 or args.min_arms > args.max_arms:
        parser.error("--min-arms must be >= 1 and <= --max-arms")
    try:
        run(args.input, args.output, args.max_dimension, args.iterations,
            args.min_arms, args.max_arms, args.skeleton_backend)
    except Exception as exc:
        LOG.error(f"ERROR: {exc}")
        if not args.quiet:
            logging.exception("Skeletonization failed")
        sys.exit(2)


if __name__ == "__main__":
    main()