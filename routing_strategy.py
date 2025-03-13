import numpy as np
import matplotlib.pyplot as plt
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from scipy.spatial import distance_matrix
import jax
import jax.numpy as jnp
import time


def generate_random_nodes(domain, total_nodes=15, seed=2):
    """domain: (xmin, xmax, ymin, ymax)"""
    np.random.seed(seed)
    xmin, xmax, ymin, ymax = domain
    nodes = np.column_stack(
        (
            np.random.uniform(xmin, xmax, total_nodes),
            np.random.uniform(ymin, ymax, total_nodes),
        )
    )
    return nodes


def farthest_point_sampling(known, domain, num_virtual, candidate_count=1000):
    xmin, xmax, ymin, ymax = domain
    candidates = np.column_stack(
        (
            np.random.uniform(xmin, xmax, candidate_count),
            np.random.uniform(ymin, ymax, candidate_count),
        )
    )
    selected = known.copy()
    new_points = []
    for _ in range(num_virtual):
        dists = distance_matrix(candidates, selected)
        min_dists = dists.min(axis=1)
        idx = np.argmax(min_dists)
        best = candidates[idx]
        new_points.append(best)
        selected = np.vstack([selected, best])
    return np.array(new_points)


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


def halton_sequence(size, dim):
    def van_der_corput(n_sample, base=2):
        sequence = []
        for i in range(n_sample):
            n_th_number = 0
            f = 1.0 / base
            i_copy = i + 1
            while i_copy > 0:
                n_th_number += f * (i_copy % base)
                i_copy //= base
                f /= base
            sequence.append(n_th_number)
        return np.array(sequence)

    seq = []
    bases = [2, 3]
    for d in range(dim):
        seq.append(van_der_corput(size, base=bases[d]))
    return np.column_stack(seq)


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


def plot_vrp_solution(ax, nodes, num_original, result, title):
    if num_original > 0:
        ax.scatter(
            nodes[:num_original, 0],
            nodes[:num_original, 1],
            color="blue",
            label="Original Nodes",
        )
    if nodes.shape[0] > num_original:
        ax.scatter(
            nodes[num_original:, 0],
            nodes[num_original:, 1],
            color="red",
            marker="s",
            label="Pseudo Nodes",
        )
    ax.scatter(0, 0, color="green", marker="D", s=100, label="Depot")
    for route in result.best.routes():
        route = list(route)
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1] != 0:
            route.append(0)
        route_coords = []
        for v in route:
            if v == 0:
                route_coords.append([0, 0])
            else:
                route_coords.append(nodes[v - 1])
        route_coords = np.array(route_coords)
        ax.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            color="black",
            linewidth=2,
            label="Route",
        )
    distance = result.best.distance()
    ax.set_title(f"{title}\nDistance: {distance:.1f}")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize="small")


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


####Grant updated dist_point_to_segment to vectorize so distance to all points can be found at same time
def dist_points_to_segment(points, p1, p2):
    # Vector from p1 to p2
    diff = p2 - p1

    # Squared norm of the segment vector (diff)
    norm = jnp.dot(diff, diff)

    # Projection of points onto the segment, normalized
    u = jnp.dot(points - p1, diff) / norm

    # Clip u to ensure it falls within the segment [0, 1]
    u = jnp.clip(u, 0, 1)

    # Calculate the closest point on the segment
    p = p1 + u[..., None] * diff  # Broadcasting to handle multiple points

    # Distance from each point to the closest point on the segment
    dp = p - points
    dist = jnp.linalg.norm(dp, axis=1)

    return dist


##### now vectorize with respect to segments so distance from all segments to all points can be found
dist_points_to_segments = jax.vmap(dist_points_to_segment, in_axes=(None, 0, 0))


def min_dist_to_edges(px, py, edges):
    if not edges:
        return 0.0
    return min(dist_point_to_segment(px, py, seg) for seg in edges)


def extract_edges_from_solution(result, nodes):
    edges = []
    # add depot to nodes
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


def fps_route_far(
    known_nodes,
    edges,
    domain,
    num_virtual=10,
    candidate_count=2000,
    alpha=1.0,
    beta=1.0,
    seed=0,
):
    np.random.seed(seed)
    xmin, xmax, ymin, ymax = domain
    candidates = np.column_stack(
        (
            np.random.uniform(xmin, xmax, candidate_count),
            np.random.uniform(ymin, ymax, candidate_count),
        )
    )
    selected = known_nodes.copy()
    pseudo_nodes = []
    for _ in range(num_virtual):
        best_score = -1
        best_idx = None
        for i, c in enumerate(candidates):
            cx, cy = c
            d_node = np.min(np.sqrt(np.sum((selected - c) ** 2, axis=1)))
            d_route = min_dist_to_edges(cx, cy, edges)
            score = alpha * d_node + beta * d_route
            if score > best_score:
                best_score = score
                best_idx = i
        chosen = candidates[best_idx]
        pseudo_nodes.append(chosen)
        selected = np.vstack([selected, chosen])
        candidates = np.delete(candidates, best_idx, axis=0)
        if candidates.shape[0] == 0:
            break
    return np.array(pseudo_nodes)


def approximate_node_voronoi(nodes, domain, grid_size=100):
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size
    assignment_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            cx = xmin + (i + 0.5) * dx
            cy = ymin + (j + 0.5) * dy
            dists = np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy)
            nearest = np.argmin(dists)
            assignment_matrix[j, i] = nearest
    return assignment_matrix


####### vectorized version of approximate_edge_voronoi
def approximate_edge_voronoi(edges, domain, grid_size=100):
    xPositions = np.linspace(domain[0], domain[1], grid_size, endpoint=False)
    yPositions = np.linspace(domain[2], domain[3], grid_size, endpoint=False)
    [cx, cy] = np.meshgrid(xPositions, yPositions)
    points = np.column_stack((cx.flatten(), cy.flatten()))
    edges = np.array(edges)
    p1 = edges[:, 0]
    p2 = edges[:, 1]
    dist = dist_points_to_segments(points, p1, p2)
    assignment_matrix = np.argmin(dist, axis=0)
    return assignment_matrix, points


# def approximate_edge_voronoi(edges, domain, grid_size=100):
#     xmin, xmax, ymin, ymax = domain
#     dx = (xmax - xmin) / grid_size
#     dy = (ymax - ymin) / grid_size
#     assignment_matrix = np.zeros((grid_size, grid_size), dtype=int)
#     for i in range(grid_size):
#         for j in range(grid_size):
#             cx = xmin + (i + 0.5) * dx
#             cy = ymin + (j + 0.5) * dy
#             best_dist = 1e12
#             best_idx = 0
#             for e_idx, seg in enumerate(edges):
#                 d = dist_point_to_segment(cx, cy, seg)
#                 if d < best_dist:
#                     best_dist = d
#                     best_idx = e_idx
#             assignment_matrix[j, i] = best_idx
#     return assignment_matrix


def calculate_partition_areas(assignment_matrix, domain, grid_size):
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size
    cell_area = dx * dy

    unique_labels, counts = np.unique(assignment_matrix, return_counts=True)

    areas = {label: count * cell_area for label, count in zip(unique_labels, counts)}
    return areas


def approximate_generalized_voronoi(nodes, edges, domain, grid_size=100):
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size
    all_sites = [(i, "node") for i in range(len(nodes))] + [
        (j, "edge") for j in range(len(edges))
    ]
    assignment_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            cx = xmin + (i + 0.5) * dx
            cy = ymin + (j + 0.5) * dy
            best_dist = 1e12
            best_site_idx = 0
            for site_idx, site_type in all_sites:
                if site_type == "node":
                    d = np.hypot(nodes[site_idx, 0] - cx, nodes[site_idx, 1] - cy)
                else:
                    d = dist_point_to_segment(cx, cy, edges[site_idx])
                if d < best_dist:
                    best_dist = d
                    if site_type == "node":
                        best_site_idx = site_idx
                    else:
                        best_site_idx = len(nodes) + site_idx
            assignment_matrix[j, i] = best_site_idx
    return assignment_matrix


def plot_voronoi(ax, assignment_matrix, domain, title="Voronoi"):
    xmin, xmax, ymin, ymax = domain
    im = ax.imshow(
        assignment_matrix,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        alpha=0.4,
        cmap="tab20",
    )
    ax.set_title(title)
    ax.set_aspect("equal")


def plot_voronoi_with_area(
    ax, assignment_matrix, partition_areas, domain, grid_size, title="Voronoi"
):
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size

    im = ax.imshow(
        assignment_matrix,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        alpha=0.4,
        cmap="tab20",
    )

    unique_labels = np.unique(assignment_matrix)
    print("unique labels", unique_labels)
    for label in unique_labels:
        indices = np.where(assignment_matrix == label)
        if indices[0].size > 0:
            cx = xmin + (np.mean(indices[1]) + 0.5) * dx
            cy = ymin + (np.mean(indices[0]) + 0.5) * dy
            print("label", label, "cx", cx, "cy", cy)
            area = partition_areas.get(label, 0)
            ax.text(
                cx,
                cy,
                f"{area:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

    ax.set_title(title)
    ax.set_aspect("equal")


def find_edges_budget_and_cells(knownNodes, domain, max_distance):
    m_orig, result_orig = build_and_solve_vrp(
        knownNodes, num_vehicles=1, max_distance=max_distance, seed=42
    )
    edges = extract_edges_from_solution(result_orig, knownNodes)
    edges.insert(0, ((0, 0), edges[0][0]))
    edges.append((edges[-1][1], (0, 0)))
    grid_size = 100
    edge_vor, cellPoints = approximate_edge_voronoi(edges, domain, grid_size=grid_size)
    cellXYList = [cellPoints[edge_vor == i] for i in range(len(edges))]
    edge_vor = edge_vor.reshape(grid_size, grid_size)

    partition_areas = calculate_partition_areas(edge_vor, domain, grid_size=grid_size)
    total_area = (domain[1] - domain[0]) * (domain[3] - domain[2])
    path_budget = np.array(
        [
            max_distance * partition_area / total_area
            for partition_area in partition_areas.values()
        ]
    )

    plot = False
    if plot:
        fig, ax = plt.subplots()
        plot_voronoi_with_area(
            ax,
            edge_vor,
            partition_areas,
            domain,
            grid_size=200,
            title="Edge-based Voronoi",
        )
        plot_vrp_solution(ax, knownNodes, 15, result_orig, "Node-based CVT + Voronoi")
        plt.show()
    return np.array(edges), path_budget, cellXYList


def main():
    domain = (0, 100, 0, 100)
    num_original = 15
    num_virtual = 5
    num_vehicles = 3
    max_distance = 350

    original_nodes = generate_random_nodes(domain, total_nodes=num_original, seed=2)

    m_orig, result_orig = build_and_solve_vrp(
        original_nodes, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    )

    # nodes_fps = np.vstack(
    #     [original_nodes, farthest_point_sampling(original_nodes, domain, num_virtual)]
    # )
    # m_fps, result_fps = build_and_solve_vrp(
    #     nodes_fps, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    # )

    # nodes_cvt = np.vstack(
    #     [
    #         original_nodes,
    #         centroidal_voronoi_tessellation(original_nodes, domain, num_virtual),
    #     ]
    # )
    # m_cvt, result_cvt = build_and_solve_vrp(
    #     nodes_cvt, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    # )

    # halton_rel = halton_sequence(num_virtual, 2)
    # xmin, xmax, ymin, ymax = domain
    # halton_nodes = np.column_stack(
    #     (
    #         halton_rel[:, 0] * (xmax - xmin) + xmin,
    #         halton_rel[:, 1] * (ymax - ymin) + ymin,
    #     )
    # )
    # nodes_halton = np.vstack([original_nodes, halton_nodes])
    # m_halton, result_halton = build_and_solve_vrp(
    #     nodes_halton, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    # )

    edges = extract_edges_from_solution(result_orig, original_nodes)
    edges.insert(0, ((0, 0), edges[0][0]))
    edges.append((edges[-1][1], (0, 0)))
    print("edges", edges)

    # pseudo_nodes_edge_cvt = weighted_cvt_route_far(
    #     original_nodes,
    #     edges,
    #     domain,
    #     num_virtual=num_virtual,
    #     alpha=0.0,
    #     beta=1.0,
    #     n_iter=30,
    #     n_samples=20000,
    #     seed=0,
    # )
    # nodes_edge_cvt = np.vstack([original_nodes, pseudo_nodes_edge_cvt])
    # m_edge_cvt, result_edge_cvt = build_and_solve_vrp(
    #     nodes_edge_cvt, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    # )

    # pseudo_nodes_edge_fps = fps_route_far(
    #     original_nodes,
    #     edges,
    #     domain,
    #     num_virtual=num_virtual,
    #     candidate_count=2000,
    #     alpha=1.0,
    #     beta=2.0,
    #     seed=0,
    # )
    # nodes_edge_fps = np.vstack([original_nodes, pseudo_nodes_edge_fps])
    # m_edge_fps, result_edge_fps = build_and_solve_vrp(
    #     nodes_edge_fps, num_vehicles=num_vehicles, max_distance=max_distance, seed=42
    # )

    # fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # plot_vrp_solution(axs[0, 0], original_nodes, num_original, result_orig, "Original")
    # plot_vrp_solution(axs[0, 1], nodes_fps, num_original, result_fps, "Node-based FPS")
    # plot_vrp_solution(axs[0, 2], nodes_cvt, num_original, result_cvt, "Node-based CVT")
    # plot_vrp_solution(
    #     axs[1, 0], nodes_halton, num_original, result_halton, "Node-based Halton"
    # )
    # plot_vrp_solution(
    #     axs[1, 1], nodes_edge_cvt, num_original, result_edge_cvt, "Edge-based CVT"
    # )
    # plot_vrp_solution(
    #     axs[1, 2], nodes_edge_fps, num_original, result_edge_fps, "Edge-based FPS"
    # )

    # plt.tight_layout()
    # plt.show()

    fig2, axs2 = plt.subplots(1, 3, figsize=(12, 4))

    # axs2[0].scatter(
    #     original_nodes[:, 0], original_nodes[:, 1], c="blue", label="Original Nodes"
    # )
    # for seg in edges:
    #     (x1, y1), (x2, y2) = seg
    #     axs2[0].plot([x1, x2], [y1, y2], color="green", linewidth=2)
    # axs2[0].set_title("Original (Nodes + Edges)")
    # axs2[0].set_aspect("equal")
    # axs2[0].legend()
    #
    # node_vor = approximate_node_voronoi(original_nodes, domain, grid_size=200)
    # plot_voronoi(axs2[1], node_vor, domain, title="Node-based Voronoi")
    # plot_vrp_solution(
    #     axs2[1], original_nodes, num_original, result_orig, "Node-based CVT + Voronoi"
    # )

    start = time.time()
    edge_vor = approximate_edge_voronoi(edges, domain, grid_size=1000)
    print("time to compute edge voronoi", time.time() - start)

    partition_areas = calculate_partition_areas(edge_vor, domain, grid_size=200)
    plot_voronoi_with_area(
        axs2[2],
        edge_vor,
        partition_areas,
        domain,
        grid_size=200,
        title="Edge-based Voronoi",
    )
    plot_vrp_solution(
        axs2[2], original_nodes, num_original, result_orig, "Node-based CVT + Voronoi"
    )

    # gen_vor = approximate_generalized_voronoi(
    #     original_nodes, edges, domain, grid_size=200
    # )
    # plot_voronoi(axs2[3], gen_vor, domain, title="Generalized Voronoi")
    # plot_vrp_solution(
    #     axs2[3], original_nodes, num_original, result_orig, "Node-based CVT + Voronoi"
    # )

    plt.tight_layout()
    plt.show()

    # fig, ax = plt.subplots(figsize=(8, 7))

    # node_vor = approximate_node_voronoi(nodes_fps, domain, grid_size=200)

    # im = ax.imshow(
    #     node_vor,
    #     origin="lower",
    #     extent=[domain[0], domain[1], domain[2], domain[3]],
    #     alpha=0.4,
    #     cmap="tab20",
    # )

    # plot_vrp_solution(
    #     ax, nodes_cvt, num_original, result_cvt, "Node-based CVT + Voronoi"
    # )

    # plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(figsize=(8, 7))

    # edge_vor = approximate_edge_voronoi(edges, domain, grid_size=200)

    # im = ax.imshow(
    #     edge_vor,
    #     origin="lower",
    #     extent=[domain[0], domain[1], domain[2], domain[3]],
    #     alpha=0.4,
    #     cmap="tab20",
    # )

    # im = ax.imshow(
    #     node_vor,
    #     origin="lower",
    #     extent=[domain[0], domain[1], domain[2], domain[3]],
    #     alpha=0.4,
    #     cmap="tab20",
    # )

    # plot_vrp_solution(
    #     ax, nodes_edge_cvt, num_original, result_edge_cvt, "Edge-based CVT + Voronoi"
    # )

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
