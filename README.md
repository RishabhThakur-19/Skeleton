# Skeleton.py  — Octopus Anatomical Skeleton Graph Extractor

Build a clean, smooth, anatomical skeleton graph from a single binary silhouette image of an octopus.

The implementation deliberately separates two representations:

1. **Dense pixel skeleton** — a topology-preserving, one-pixel-wide medial axis extracted via Zhang–Suen thinning. Used only as a geometric guide for routing.
2. **Anatomical tree** — a 21–26 node graph whose edges are mask-constrained cubic B-splines. This is the final output.

---

## Output Graph Structure

```
Mantle Center (1 node) ──┬── Arm 1: Base → Mid 1 → Tip
                         ├── Arm 2: Base → Mid 1 → Tip
                         ├── Arm 3: Base → Mid 1 → Tip
                         ├── Arm 4: Base → Mid 1 → Tip
                         ├── Arm 5: Base → Mid 1 → Tip
                         ├── Arm 6: Base → Mid 1 → Tip
                         ├── Arm 7: Base → Mid 1 → Tip
                         ├── Arm 8: Base → Mid 1 → Tip
                         └── Head (1 node)

Total: 1 center + 1 head + 8 arms × 3 nodes = 26 nodes, 25 edges
```

- **Mantle Center** — the arm-confluence hub (degree 8), found at the junction cluster where most arms meet.
- **Head** — the second-highest distance-transform peak that forms its own distinct circular blob, separate from the Mantle Center.
- **Arm nodes** — each arm has exactly 3 nodes: Base (near mantle), Mid 1 (intermediate), Tip (distal endpoint).

---

## Quick Start

```bash
# Basic usage
python R12_best.py input.png output/

# With custom parameters
python R12_best.py input.png output/ --iterations 4 --min-arms 5 --max-arms 8 --max-dimension 1024

# Quiet mode (warnings only)
python R12_best.py input.png output/ --quiet
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `input` | positional | *(required)* | Path to binary-mask image (PNG, JPEG, etc.) |
| `output` | positional | *(required)* | Output directory for results |
| `--max-dimension` | int | `760` | Working resolution for dense thinning — the image is resized so its longest side is at most this many pixels |
| `--iterations` | int | `3` | Number of automatic refinement iterations (1–4). Each uses different smoothing parameters; the best-scoring result is selected |
| `--min-arms` | int | `5` | Minimum number of distinct arms required to accept a result. Fewer than this triggers a hard failure |
| `--max-arms` | int | `8` | Biological maximum/ideal arm count. The algorithm aims for this many but accepts fewer if the silhouette is occluded |
| `--quiet` | flag | `off` | Suppress INFO-level logging; only warnings and errors are shown |

---

## Dependencies

- Python 3.9+
- `numpy`
- `opencv-python` (or `opencv-contrib-python` for faster thinning via `cv2.ximgproc`)
- `scipy`
- `networkx`
- `matplotlib`

Install with:

```bash
pip install numpy opencv-python scipy networkx matplotlib
```

No scikit-image dependency is required.

---

## Pipeline Stages

### 1. Binary Mask Preparation (`load_binary`, `prepare_mask`)

#### Polarity-Agnostic Loading (`load_binary`)

The mask loader does not assume white-on-black or black-on-white. It runs Otsu thresholding in **both directions**:

1. `cv2.THRESH_BINARY + cv2.THRESH_OTSU` — foreground is brighter than background
2. `cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU` — foreground is darker than background

Each polarity is scored by a `quality()` function that evaluates the largest connected component:

- **Area bonus**: Larger foreground blob is preferred (`area` pixels)
- **Border penalty**: Components touching the image border are penalized (`-1000.0 × border_pixels`) — this rejects cases where the background leaks to the edge
- **Coverage penalty**: If the largest component covers > 85% of the image area, it is heavily penalized (`-1e9`) — this rejects the wrong polarity where the entire image becomes foreground

The polarity with the higher quality score is selected. If the winning component has fewer than 100 pixels, a `ValueError` is raised.

#### Mask Preprocessing (`prepare_mask`)

1. **Downsampling**: The mask is resized so its longest side is at most `--max-dimension` pixels (minimum 32 px on any side). Area interpolation (`cv2.INTER_AREA`) preserves binary structure.

2. **Scale-adaptive Gaussian smoothing**: A Gaussian blur suppresses suction-cup crenellation while preserving thin arms. The sigma scales with image size:
   ```
   sigma = max(0.65, smooth_factor × max(sw, sh) / 650.0)
   ```
   The `smooth_factor` varies per iteration (0.90, 0.65, 0.45, 0.30 — see Stage 8). After blurring, pixels ≥ 112 are kept as foreground. Gaussian thresholding is less destructive than morphological opening.

3. **Pinhole filling**: A single 3×3 morphological close (`cv2.MORPH_CLOSE`, 1 iteration) fills only tiny holes (1–2 pixel gaps). Anatomical holes inside curled arms are preserved because they are larger than the kernel.

4. **Largest component**: Connected components are computed (8-connectivity) and only the largest foreground blob is kept. This removes stray noise and disconnected artifacts.

The function returns the processed binary mask and the scale factors (`scale_x`, `scale_y`) for converting back to original image coordinates.

---

### 2. Topology-Preserving Thinning (`zhang_suen_thinning`, `remove_tiny_spurs`)

#### Zhang–Suen Thinning

Reduces the binary mask to a one-pixel-wide, 8-connected medial axis skeleton. The algorithm runs iteratively, removing **removable boundary pixels** in two alternating sub-iterations until convergence.

**Compiled path (preferred)**: If `opencv-contrib-python` is installed, the function uses `cv2.ximgproc.thinning()` with `THINNING_ZHANGSUEN`. This is a compiled C++ implementation that is substantially faster.

**Vectorized NumPy fallback**: If `cv2.ximgproc` is unavailable, the pure-NumPy implementation (`_zhang_suen_thinning_numpy`) runs. It pads the binary image with a 1-pixel zero border, then iterates:

For each sub-iteration (phase 0 and phase 1), every foreground pixel `c` is evaluated against its 8 neighbors (p2–p9 in clockwise order starting from top-left):

- `b` = sum of 8 neighbors (number of non-zero neighbors)
- `a` = number of 0→1 transitions around the 8-neighbor ring (equivalent to the number of connected components in the 3×3 neighborhood excluding center)
- A pixel is **removable** if all conditions hold:
  - `c == 1` (is foreground)
  - `2 ≤ b ≤ 6` (not too isolated, not too surrounded)
  - `a == 1` (exactly one transition — preserves topology)
  - **Phase 0**: `p2×p4×p6 == 0` AND `p4×p6×p8 == 0` (opposite-pair constraints for even phase)
  - **Phase 1**: `p2×p4×p8 == 0` AND `p2×p6×p8 == 0` (opposite-pair constraints for odd phase)

The two phases alternate, each removing different boundary pixels. The loop terminates when a full pair of phases removes zero pixels (convergence). The padded border is stripped before returning.

All neighbor accesses use **sliced array views** (`im[:-2, 1:-1]`, etc.) for vectorized evaluation — no per-pixel Python loops.

#### Tiny Spur Removal (`remove_tiny_spurs`)

Removes only terminal chains shorter than local width, preserving long thin arms. Runs up to `passes` (default 2) iterative passes:

1. Build the current pixel graph to find degrees
2. For each **leaf node** (degree-1 pixel):
   - Walk forward along the chain while degree ≤ 2 (following the terminal branch)
   - Accumulate the chain length (sum of Euclidean edge weights)
   - When a junction (degree ≥ 3) is reached, check the length against the local width at the junction
3. The local width at the junction is `2.0 × distance_transform_radius` — the Euclidean distance to the nearest mask boundary
4. A chain is deleted if its length is less than `max(2.5, 0.55 × local_width)` — the 0.55× threshold means only spurs shorter than roughly half the local arm width are removed
5. If no chains are deleted in a pass, the algorithm terminates early

Long thin arms are preserved because their length far exceeds the local width threshold. Only mask bumps and junction artifacts are removed.

---

### 3. Pixel Graph & Root Selection (`pixel_graph`, `choose_anatomical_root`)

#### Pixel Graph Construction

Every skeleton pixel becomes a node; 8-neighborhood connectivity forms edges with Euclidean weights. The construction uses a **single padded-array pass** for speed:

1. All skeleton pixel coordinates are collected as `(y, x)` integer pairs
2. A 2D index map `index[h, w]` assigns each skeleton pixel a flat integer ID (0 to N-1); non-skeleton pixels are -1
3. The index map is padded with a 1-pixel zero border: `padded[h+2, w+2]`
4. For each of the 8 neighbor offsets `(dy, dx)`, neighbor indices are looked up via **vectorized array indexing**: `padded[py + dy, px + dx]` — no per-pixel bounds checks needed because of the padding
5. Valid neighbors (index ≥ 0) produce edges with weight `√2` for diagonal neighbors, `1.0` for orthogonal
6. All edges are sorted by source index, then converted to an adjacency list using `searchsorted` for O(N log N) boundary detection

The result is `(points, adjacency, index)` where `adjacency[i]` is a list of `(neighbor_id, weight)` tuples.

#### Anatomical Root Selection

The root is the **arm-confluence hub** — NOT the global distance-transform maximum (that would be the mantle cap). The algorithm:

1. **Find junction pixels**: All skeleton pixels with degree ≥ 3 in the pixel graph
2. **Dilate and cluster**: Junction pixels are placed on a binary image, dilated with a 7×7 kernel, then clustered via 8-connected components. This groups nearby junctions into connected junction regions
3. **Score each cluster**:
   ```
   score = log(1 + area) + 0.08 × local_radius − 2.0 × centrality
   ```
   - `area` = number of pixels in the dilated junction cluster (larger clusters = more arms meeting)
   - `local_radius` = distance-transform value at the cluster centroid (prevents suction-cup artifacts from winning)
   - `centrality` = normalized distance from the cluster centroid to the foreground centroid (rejects curled tip clusters at the periphery)
4. **Select root pixel**: From the best cluster, the 30 nearest junction pixels are considered. Among them, the one with the highest `distance − 0.15 × distance_to_centroid` is chosen — this biases toward high-clearance ridge points near the cluster center

If no junctions are found (degenerate case), the skeleton pixel with the highest distance-transform value is used as fallback.

---

### 4. Geodesic Tree & Arm Path Selection (`dijkstra_tree`, `select_arm_paths`)

#### Dijkstra Geodesic Tree

Geodesic distances are computed from the root through the skeleton graph using Dijkstra's algorithm with a priority queue. The edge cost is slightly adjusted to prefer high-clearance ridge pixels:

```
cost = step × (1.0 + 0.10 / max(clearance, 0.5))
```

where `clearance = 0.5 × (distance[yu, xu] + distance[yv, xv])` — the average distance-transform value of the two connected pixels. When a skeleton loop offers two alternative paths, this bias slightly prefers routes through the center of the arm (high clearance) over paths along the inside corner of a curl.

The function returns `(geodesic_distances, parent_array)` — the parent array enables backtracking from any pixel to the root along the geodesic tree.

#### Arm Path Selection

The algorithm finds arm paths through a multi-stage process:

**Stage A — Endpoint tracing**:
1. All skeleton endpoints (degree-1 pixels with finite geodesic distance) are identified
2. Each endpoint is traced back to the root via `backtrack()` following the parent array
3. Paths shorter than 4 pixels are discarded
4. All valid paths are sorted by arc length (descending)

**Stage B — Greedy deduplication**:
Candidates sharing > 58% of their prefix with an already-selected path are rejected. Two candidates sharing most of their route represent a real tip plus a local spur on the same arm — only the longer one is retained.

**Stage C — Length floor enforcement**:
A real arm reaches well clear of the arm-confluence hub. The minimum arm length is:
```
length_floor = max(4.0 × root_radius, 0.4 × median(genuine_lengths))
```
where `root_radius` is the distance-transform value at the root pixel. Short stubs that only just clear the hub's girth are mask bumps, not appendages.

**Stage D — Geodesic maxima recovery** (if fewer than `max_arms` found):
When skeleton endpoints are lost through touching/occluded silhouettes, the algorithm falls back to geodesic local maxima — pixels on the medial tree that are far from the root but not at skeleton endpoints. Pixels are considered in order of decreasing geodesic distance. Each candidate must:
- Have a backtrack path of at least 8 pixels
- Clear the length floor
- Share < 58% prefix with already-selected paths
- The search stops once `max_arms + 2` candidates are found

**Stage E — Relaxed recovery** (final fallback for strongly overlapping poses):
If still fewer than `max_arms`, the prefix threshold is relaxed from 0.58 to 0.82. This allows more similar paths to coexist, recovering arms that share long common routes due to severe overlap. The length floor is still enforced.

**Stage F — Mantle cap removal** (if more candidates than `max_arms`):
If there are more candidates than `max_arms`, the broadest distal-half route (mantle cap) is identified and removed. The cap is the path whose distal half has the largest average distance-transform radius — it routes across the broad mantle body rather than down a narrow arm.

**Stage G — Final selection**:
The longest distinct arms are selected, up to `max_arms`. If fewer than `min_arms` survive all stages, a `RuntimeError` is raised.

---

### 5. Branch Splines (`build_branches`)

#### Clockwise Ordering

Arms are ordered by launch angle at 12% geodesic distance from the root. For each arm path, the point at 12% of its total arc length is sampled, and the angle from the root to this point is computed. Arms are sorted clockwise starting from the uppermost arm (smallest angle in image coordinates, where Y increases downward). This produces stable anatomical numbering across frames.

#### Mask-Constrained B-Spline Fitting (`fit_mask_constrained_spline`)

Each raw polyline (pixel path from root to tip) is smoothed into a cubic B-spline constrained to stay inside the mask:

1. **Uniform control points**: Control points are placed at equal arc-length intervals along the raw polyline. The number of control points is `max(5, len(raw) // 15)`.

2. **Cubic B-spline fitting**: `scipy.interpolate.splprep` (k=3) fits a parametric cubic B-spline. The smoothness parameter `s` controls the trade-off between fidelity to the raw path and smoothness:
   ```
   s = spline_smooth × sqrt(len(raw_points))
   ```
   where `spline_smooth` varies per iteration (0.90, 0.65, 0.45, 0.30 — see Stage 8).

3. **Dense evaluation**: The spline is evaluated at 200 uniformly-spaced parameter values using `splev`.

4. **Silhouette enforcement**: Each spline point is checked against the mask:
   - Points inside the mask are kept as-is
   - Points outside the mask are projected back to the nearest point on the **raw centerline** neighborhood (not arbitrary nearest foreground pixel). This preserves the correct routing through self-crossing curls — the spline is snapped to where the raw path went, not to the nearest mask edge which might be a different arm

5. **Gaussian coordinate smoothing**: `scipy.ndimage.gaussian_filter1d` is applied to each coordinate axis (sigma = 1.5) to remove pixel-scale wobble. Endpoints are preserved exactly by zero-padding the boundaries before filtering.

6. **Final mask check**: After Gaussian smoothing, any point that was nudged outside the mask is hard-snapped to the nearest foreground pixel.

7. **Resampling**: The final curve is resampled to 200 points for uniform density.

#### Node Sampling

Each arm is divided into exactly 3 equi-arc-length nodes:
- **Base** — at 0% arc length (root end)
- **Mid 1** — at 50% arc length
- **Tip** — at 100% arc length (distal endpoint)

Arc lengths are computed from the dense spline curve using cumulative Euclidean distances.

#### Curvature Computation

Per-point curvature is computed from the dense spline using the standard formula for parametric curves:
```
κ = |x'·y'' − y'·x''| / (x'² + y'²)^(3/2)
```
where derivatives are computed via central differences on the spline coordinates. Curvature is used for quality scoring and branch statistics.

---

### 6. Graph Construction (`construct_graph`)

#### Head Placement

The head is placed at the second-highest distance-transform peak, non-maximum-suppressed from the Mantle Center:

1. The distance transform is computed on the binary mask
2. A non-maximum suppression pass finds local maxima using `scipy.ndimage.maximum_filter` with increasing window sizes (5, 7, 9, ..., up to 31)
3. For each window size, the top-N peaks are found (sorted by distance-transform value)
4. The algorithm searches for a peak that is:
   - At least `min_distance` pixels from the Mantle Center (starts at `0.12 × image_diagonal`, increases by 20% per attempt)
   - A local maximum within its window
   - Forms its own distinct circular blob
5. If no suitable peak is found at any separation distance, the head is placed at the farthest point on the mask boundary from the root

#### Edge Polylines

Dense spline segments are generated between consecutive nodes on each branch. Per-segment statistics are computed:
- **Curvature**: Mean and maximum curvature along the segment
- **Radius**: Mean distance-transform value (local clearance)
- **Polyline**: The actual pixel coordinates of the spline segment

#### Node and Edge Data Structures

Each node is a dictionary with:
- `node_id`: integer ID
- `x`, `y`: coordinates in original image space (scaled from working resolution)
- `body_part`: string label ("Mantle Center", "Head", "Arm N Base", "Arm N Mid 1", "Arm N Tip")
- `is_center`, `is_head`, `is_tip`: boolean flags
- `arm_id`: integer arm assignment (0 for center and head)
- `branch_id`: integer branch assignment

Each edge is a dictionary with:
- `edge_id`: integer ID
- `source`, `target`: node IDs
- `label`: string label
- `body_part`: string label
- `branch_id`: integer branch assignment
- `length`: Euclidean length
- `curvature_mean`, `curvature_max`: curvature statistics
- `radius_mean`: mean local clearance
- `polyline`: list of `(x, y)` coordinates

---

### 7. Quality Scoring & Validation (`quality_score`, `validate_requirements`)

#### Quality Score (Higher Is Better)

The quality score is a weighted sum of metrics:

| Component | Weight | Description |
|---|---|---|
| Tree structure | +1000 / -1000 | +1000 if the graph is a connected acyclic tree, -1000 otherwise |
| Arm count deviation | -500 × \|arm_count − max_arms\| | Penalizes missing arms |
| Tip count deviation | -500 × \|tip_count − arm_count\| | Penalizes mismatched tips |
| Center count | -1000 × \|center_count − 1\| | Must have exactly one center |
| Head count | -1000 × \|head_count − 1\| | Must have exactly one head |
| Inside fraction | +800 × inside_fraction | Fraction of spline pixels inside the mask (0–1) |
| Crossing edges | -120 × crossing_edges | Number of edge crossings |
| Duplicate nodes | -300 × duplicate_nodes | Penalizes overlapping nodes |
| Duplicate edges | -300 × duplicate_edges | Penalizes overlapping edges |
| Average curvature | -25 × average_branch_curvature | Penalizes excessive curvature |
| Maximum curvature | -3 × maximum_curvature | Penalizes sharp bends |

#### Validation Checks

**Soft warnings** (logged but not fatal):
- Node count in valid range (21–26 for 8 arms)
- Graph is one connected acyclic tree
- Arm count in [min_arms, max_arms]
- Tip count matches arm count
- Exactly one center and one head
- No duplicate nodes or edges
- ≥ 99.9% of spline pixels inside the mask
- Each arm has Base, Mid 1, Tip nodes with dense spline geometry

**Hard constraints** (fatal):
- Arm count < min_arms — the result is rejected entirely

#### Crossing Detection

The `count_crossings` function uses a fully vectorized broad-phase/narrow-phase approach:
1. **Broad phase**: KD-tree radius joins (`scipy.spatial.cKDTree.query_radius`) find edge pairs whose bounding boxes are close enough to potentially cross
2. **Narrow phase**: Segment-segment intersection tests on the candidate pairs using vectorized cross-product computations
3. Adjacent edges (sharing a node) are excluded from crossing checks

This replaces the original O(n²) Python loop and reduces crossing detection from ~85% of total runtime to a small fraction.

---

### 8. Iterative Optimization (`run`)

The end-to-end optimizer runs multiple smoothing iterations with different parameters. Each iteration uses a different `(morph_smooth, spline_smooth)` pair:

| Iteration | Mask Smoothing Factor | Spline Smoothness |
|---|---|---|
| 1 | 0.75 | 0.90 |
| 2 | 1.00 | 0.65 |
| 3 | 1.25 | 0.45 |
| 4 | 1.45 | 0.30 |

The `--iterations` flag controls how many of these are run (1–4). Each iteration:

1. Calls `dense_iteration()` which runs the full pipeline from mask preparation through arm path selection
2. Calls `build_branches()` to fit splines and sample nodes
3. Calls `construct_graph()` to build the final graph
4. Computes `graph_metrics()` and `quality_score()`
5. Runs `validate_requirements()` to check constraints

The best-scoring candidate across all iterations is selected. If the best candidate violates hard constraints (arm count < min_arms), a `RuntimeError` is raised.

---

## Output Files

All files are written to the output directory:

| File | Format | Description |
|---|---|---|
| `graph.json` | JSON | Complete graph structure with nodes, edges, metrics, and iteration records |
| `nodes.csv` | CSV | Node table: node_id, x, y, body_part, arm_id, branch_id, is_center, is_head, is_tip |
| `edges.csv` | CSV | Edge table: edge_id, source, target, label, body_part, branch_id, length, curvature_mean, curvature_max, radius_mean |
| `graph.png` | PNG | Annotated graph figure (matplotlib) with color-coded arms, node markers, and labels |
| `skeleton.png` | PNG | Binary skeleton overlay on the mask (white polylines on dark background) |
| `overlay.png` | PNG | Color-coded arm overlay on the original mask with node markers |

### Node Color Legend (graph.png)

| Node Type | Marker | Color |
|---|---|---|
| Mantle Center | Star (★) | Red (#ff3b30) |
| Head | Triangle (▲) | Green (#1cd679) |
| Tips | Circle (●) | Yellow (#ffd60a) |
| Base / Mid nodes | Circle (●) | Blue (#53c8ff) |

### Arm Color Palette

Arms use the matplotlib `tab10` colormap, cycling through 10 distinct colors. Arm N uses color `(N-1) % 10` from the palette.

---

## Coordinate System

- **Origin**: Top-left corner of the image
- **X axis**: Rightward (columns)
- **Y axis**: Downward (rows)
- **Units**: Image pixels (original input resolution, not the downsampled working resolution)

All output coordinates are scaled back to the original image dimensions using the `scale_x` and `scale_y` factors computed during mask preparation.

---

## Metrics Summary

The `graph_metrics` function computes these metrics:

| Metric | Description |
|---|---|
| `total_nodes` | Total number of nodes in the graph |
| `total_edges` | Total number of edges |
| `connected_components` | Number of connected components (should be 1) |
| `number_of_cycles` | Number of cycles in the graph (should be 0 for a tree) |
| `average_edge_length` | Mean Euclidean length of all edges |
| `average_branch_curvature` | Mean curvature across all spline segments |
| `maximum_curvature` | Maximum curvature across all spline segments |
| `average_node_spacing` | Mean distance between consecutive nodes on branches |
| `skeleton_length` | Total arc length of all splines |
| `average_distance_to_boundary` | Mean distance-transform value along all splines |
| `branch_symmetry` | Standard deviation of arm lengths divided by mean (lower = more symmetric) |
| `arm_count` | Number of detected arms |
| `tip_count` | Number of tip nodes |
| `head_count` | Number of head nodes (should be 1) |
| `inside_fraction` | Fraction of spline pixels inside the mask (should be ≥ 0.999) |
| `crossing_edges` | Number of edge crossings |

---

## Key Design Decisions

### Why Two Representations?

The dense pixel skeleton is topology-preserving but noisy and jagged. The anatomical tree is smooth and semantically meaningful but needs the dense skeleton as a geometric guide to route arms correctly through complex poses (curls, overlaps, self-contacts). The two representations serve different purposes: the dense skeleton provides the correct topology and routing, while the anatomical tree provides a clean, smooth, semantically labeled output.

### Why Mask-Constrained Splines?

A free spline can leave the silhouette. The mask constraint ensures every point on every arm spline stays inside the octopus body — even for tight curls that pass close to themselves. Projection is to the nearest point on the *raw centerline* (not arbitrary nearest foreground), which preserves the correct routing through self-crossing regions. If a spline point leaves the mask, snapping it to the nearest mask edge might route it through a different arm; snapping to the raw centerline keeps it on the correct path.

### Why Not scikit-image Skeletonize?

Zhang–Suen thinning produces a deterministic, one-pixel-wide skeleton with no branching artifacts. The `cv2.ximgproc` implementation is substantially faster than scikit-image's approach. The NumPy fallback provides full vectorized performance without the scikit-image dependency.

### How Does It Handle Occluded Arms?

When skeleton endpoints are lost through touching/occluded silhouettes, the algorithm falls back to geodesic local maxima — points on the medial tree that are far from the root but not at skeleton endpoints. This is deterministic and follows the medial tree without drawing Euclidean rays through the mask. The relaxed recovery stage (Stage E) further handles strongly overlapping poses by allowing more similar paths to coexist.

### What About the Head?

The head is placed at the second-highest distance-transform peak that forms its own distinct circular blob. Non-maximum suppression with increasing separation distances ensures the head and mantle are found as separate entities, even when the body is roughly radially symmetric. The minimum separation starts at 12% of the image diagonal and increases by 20% per attempt until two distinct peaks are found.

### Why Iterative Optimization?

Different smoothing parameters produce different results on different images. A single fixed parameter set cannot handle the full range of octopus poses and image qualities. By running multiple iterations with varying mask smoothing and spline smoothness, the algorithm explores a range of solutions and selects the best one. This is more robust than a single-pass approach and avoids manual parameter tuning.

---

## Performance Notes

- **Crossing detection**: The `count_crossings` function uses a fully vectorized broad-phase/narrow-phase approach with KD-tree radius joins, replacing an O(n²) Python loop. This reduced crossing detection from ~85% of total runtime to a small fraction.
- **Pixel graph construction**: `pixel_graph` builds adjacency with a single padded-array pass — no per-pixel Python loops. Neighbor lookups use vectorized array indexing with an 8-neighbor offset loop.
- **Thinning**: Prefers `cv2.ximgproc.thinning` (compiled C++) over the NumPy fallback. The NumPy implementation uses sliced array views for full vectorization.
- **Dijkstra**: Uses `heapq` for the priority queue with early termination on stale entries. The clearance-adjusted cost function adds minimal overhead.
- **Working resolution**: The `--max-dimension` parameter controls the working resolution. Lower values are faster but may lose fine detail in thin arms. The default of 760 pixels provides a good balance for most inputs.

---
