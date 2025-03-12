import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from scipy.spatial import distance_matrix
import pickle
import random


def generate_random_nodes(domain, total_nodes=15, seed=2):
    np.random.seed(seed)
    xmin, xmax, ymin, ymax = domain
    nodes = np.column_stack(
        (
            np.random.uniform(xmin, xmax, total_nodes),
            np.random.uniform(ymin, ymax, total_nodes),
        )
    )
    return nodes


def build_and_solve_vrp(
    nodes, num_vehicles=1, max_distance=1000, seed=42, max_runtime=10
):
    m = Model()
    m.add_vehicle_type(num_vehicles, max_distance=max_distance)
    depot = m.add_depot(0, 0)
    clients = []
    for x, y in nodes:
        c = m.add_client(x=x, y=y)
        clients.append(c)
    all_locs = [depot] + clients
    for i, frm in enumerate(all_locs):
        for j, to in enumerate(all_locs):
            if i != j:
                dist = int(np.hypot(frm.x - to.x, frm.y - to.y))
                m.add_edge(frm, to, distance=dist)
    result = m.solve(stop=MaxRuntime(max_runtime), seed=seed, display=False)
    return m, result


def dist_point_to_segment(px, py, seg):
    (x1, y1), (x2, y2) = seg
    seg_vec = np.array([x2 - x1, y2 - y1])
    pt_vec = np.array([px - x1, py - y1])
    seg_len_sq = seg_vec.dot(seg_vec)
    if seg_len_sq < 1e-9:
        return np.hypot(px - x1, py - y1)
    t = pt_vec.dot(seg_vec) / seg_len_sq
    if t < 0:
        return np.hypot(px - x1, py - y1)
    elif t > 1:
        return np.hypot(px - x2, py - y2)
    else:
        proj = np.array([x1, y1]) + t * seg_vec
        return np.hypot(px - proj[0], py - proj[1])


def min_dist_to_edges(px, py, edges):
    if not edges:
        return 0.0
    return min(dist_point_to_segment(px, py, seg) for seg in edges)


def centroidal_voronoi_tessellation(known, domain, num_virtual, iterations=50):
    xmin, xmax, ymin, ymax = domain
    virtual = np.column_stack(
        (
            np.random.uniform(xmin, xmax, num_virtual),
            np.random.uniform(ymin, ymax, num_virtual),
        )
    )
    for _ in range(iterations):
        generators = np.vstack([known, virtual])
        num_generators = generators.shape[0]
        assignments = {i: [] for i in range(num_generators)}
        sample_points = np.column_stack(
            (np.random.uniform(xmin, xmax, 10000), np.random.uniform(ymin, ymax, 10000))
        )
        for pt in sample_points:
            dists = np.linalg.norm(generators - pt, axis=1)
            nearest = np.argmin(dists)
            assignments[nearest].append(pt)
        new_virtual = []
        for i in range(num_virtual):
            idx = len(known) + i
            region_points = np.array(assignments[idx])
            if region_points.size > 0:
                centroid = region_points.mean(axis=0)
            else:
                centroid = virtual[i]
            new_virtual.append(centroid)
        new_virtual = np.array(new_virtual)
        if np.linalg.norm(new_virtual - virtual) < 1e-3:
            virtual = new_virtual
            break
        virtual = new_virtual
    return virtual


def weighted_cvt_route_far(
    known_nodes,
    edges,
    domain,
    num_virtual=10,
    alpha=0.1,
    beta=1.0,
    n_iter=30,
    n_samples=20000,
    seed=0,
):
    np.random.seed(seed)
    xmin, xmax, ymin, ymax = domain
    virtual_nodes = np.column_stack(
        (
            np.random.uniform(xmin, xmax, num_virtual),
            np.random.uniform(ymin, ymax, num_virtual),
        )
    )
    for it in range(n_iter):
        sample_pts = np.column_stack(
            (
                np.random.uniform(xmin, xmax, n_samples),
                np.random.uniform(ymin, ymax, n_samples),
            )
        )
        generators = np.vstack([known_nodes, virtual_nodes])
        diff = sample_pts[:, None, :] - generators[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        nearest_idx = np.argmin(dist_sq, axis=1)
        new_virtual = []
        offset = known_nodes.shape[0]
        for i in range(num_virtual):
            gen_i = offset + i
            pts_i = sample_pts[nearest_idx == gen_i]
            if len(pts_i) == 0:
                new_virtual.append(virtual_nodes[i])
                continue
            w_i = []
            for sx, sy in pts_i:
                d_route = min_dist_to_edges(sx, sy, edges)
                w_val = alpha + beta * d_route
                w_i.append(w_val)
            w_i = np.array(w_i)
            numer = np.sum(pts_i * w_i.reshape(-1, 1), axis=0)
            denom = np.sum(w_i)
            centroid = numer / denom
            new_virtual.append(centroid)
        new_virtual = np.array(new_virtual)
        if np.linalg.norm(new_virtual - virtual_nodes) < 1e-5:
            virtual_nodes = new_virtual
            break
        virtual_nodes = new_virtual
    return virtual_nodes


def extract_edges_from_solution(result, nodes):
    edges = []
    for route in result.best.routes():
        if len(route) < 2:
            continue
        coords = []
        for rid in route:
            if rid == 0:
                coords.append((0, 0))
            else:
                coords.append(tuple(nodes[rid - 1]))
        for i in range(len(coords) - 1):
            edges.append((coords[i], coords[i + 1]))
    return edges


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def do_line_segments_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
            1
        ] <= max(p[1], r[1]):
            return True
        return False

    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False


def line_segment_intersects_rect(edge, rect):
    (xmin, ymin, xmax, ymax) = rect
    (p1, p2) = edge

    def point_in_rect(pt, rect):
        (x, y) = pt
        (xmin, ymin, xmax, ymax) = rect
        return xmin <= x <= xmax and ymin <= y <= ymax

    if point_in_rect(p1, rect) or point_in_rect(p2, rect):
        return True

    rect_edges = [
        ((xmin, ymin), (xmax, ymin)),  # bottom
        ((xmax, ymin), (xmax, ymax)),  # right
        ((xmax, ymax), (xmin, ymax)),  # top
        ((xmin, ymax), (xmin, ymin)),  # left
    ]
    for r_edge in rect_edges:
        if do_line_segments_intersect(p1, p2, r_edge[0], r_edge[1]):
            return True
    return False


def evaluate_edge_metrics(edges, domain, grid_resolution=10):
    xmin, xmax, ymin, ymax = domain
    cell_width = (xmax - xmin) / grid_resolution
    cell_height = (ymax - ymin) / grid_resolution
    total_cells = grid_resolution * grid_resolution

    cell_edge_counts = []
    cell_nearest_dists = []

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            cell_xmin = xmin + i * cell_width
            cell_ymin = ymin + j * cell_height
            cell_xmax = cell_xmin + cell_width
            cell_ymax = cell_ymin + cell_height
            cell_rect = (cell_xmin, cell_ymin, cell_xmax, cell_ymax)
            center = ((cell_xmin + cell_xmax) / 2, (cell_ymin + cell_ymax) / 2)

            count = 0
            for edge in edges:
                if line_segment_intersects_rect(edge, cell_rect):
                    count += 1
            cell_edge_counts.append(count)

            if edges:
                min_dist = min(
                    dist_point_to_segment(center[0], center[1], edge) for edge in edges
                )
            else:
                min_dist = math.hypot(cell_width, cell_height)
            cell_nearest_dists.append(min_dist)

    covered_cells = sum(1 for count in cell_edge_counts if count > 0)
    ECR = covered_cells / total_cells
    avg_edge_count = sum(cell_edge_counts) / total_cells
    EDV = sum((count - avg_edge_count) ** 2 for count in cell_edge_counts) / total_cells
    dmax = max(cell_nearest_dists)
    ANE = sum(cell_nearest_dists) / total_cells

    return {"ECR": ECR, "EDV": EDV, "dmax": dmax, "ANE": ANE}


domain = (0, 100, 0, 100)
num_vehicles = 1
max_distance = 1000

results_original = {"ECR": [], "EDV": [], "dmax": [], "ANE": []}
results_node_cvt = {"ECR": [], "EDV": [], "dmax": [], "ANE": []}
results_edge_cvt = {"ECR": [], "EDV": [], "dmax": [], "ANE": []}
results_initial = {"num_original": [], "num_virtual": []}

num_cases = 100

for seed in range(num_cases):
    num_original = random.randint(5, 15)  # Randomly select between 5 and 15
    num_virtual = random.randint(1, 5)
    results_initial["num_original"].append(num_original)
    results_initial["num_virtual"].append(num_virtual)
    original_nodes = generate_random_nodes(domain, total_nodes=num_original, seed=seed)
    m_orig, result_orig = build_and_solve_vrp(
        original_nodes, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    )
    edges = extract_edges_from_solution(result_orig, original_nodes)
    original_metrics = evaluate_edge_metrics(edges, domain, grid_resolution=30)
    for key in results_original:
        results_original[key].append(original_metrics[key])

    cvt_nodes = centroidal_voronoi_tessellation(original_nodes, domain, num_virtual)
    nodes_cvt = np.vstack([original_nodes, cvt_nodes])
    m_cvt, result_cvt = build_and_solve_vrp(
        nodes_cvt, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    )
    node_cvt_edges = extract_edges_from_solution(result_cvt, nodes_cvt)
    node_cvt_metrics = evaluate_edge_metrics(node_cvt_edges, domain, grid_resolution=30)
    for key in results_node_cvt:
        results_node_cvt[key].append(node_cvt_metrics[key])

    pseudo_nodes_edge_cvt = weighted_cvt_route_far(
        original_nodes,
        edges,
        domain,
        num_virtual=num_virtual,
        alpha=0.0,
        beta=1.0,
        n_iter=30,
        n_samples=20000,
        seed=0,
    )
    nodes_edge_cvt = np.vstack([original_nodes, pseudo_nodes_edge_cvt])
    m_edge_cvt, result_edge_cvt = build_and_solve_vrp(
        nodes_edge_cvt, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    )
    edge_cvt_edges = extract_edges_from_solution(result_edge_cvt, nodes_edge_cvt)
    edge_cvt_metrics = evaluate_edge_metrics(edge_cvt_edges, domain, grid_resolution=20)
    for key in results_edge_cvt:
        results_edge_cvt[key].append(edge_cvt_metrics[key])

with open("results_original.pkl", "wb") as f:
    pickle.dump(results_original, f)
with open("results_node_cvt.pkl", "wb") as f:
    pickle.dump(results_node_cvt, f)
with open("results_edge_cvt.pkl", "wb") as f:
    pickle.dump(results_edge_cvt, f)
with open("results_initial.pkl", "wb") as f:
    pickle.dump(results_initial, f)

import matplotlib.pyplot as plt

metrics_names = ["ECR", "EDV", "dmax", "ANE"]
fig, axs = plt.subplots(2, 2, figsize=(6, 6))
axs = axs.flatten()

for idx, metric in enumerate(metrics_names):
    data = [
        results_original[metric],
        results_node_cvt[metric],
        results_edge_cvt[metric],
    ]
    axs[idx].boxplot(data, labels=["Original VRP", "Node-based CVT", "Edge-based CVT"])
    axs[idx].set_title(metric)
    axs[idx].set_ylabel("Value")

plt.tight_layout()
plt.show()
