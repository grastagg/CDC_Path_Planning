from jax._src.interpreters.batching import batch
import numpy as np
from pyoptsparse import Optimization, OPT, IPOPT
from scipy.interpolate import BSpline
import time
from jax import jacfwd
from jax import jit
from functools import partial
import jax.numpy as jnp
import getpass
import matplotlib.pyplot as plt
import matplotlib
import jax
import time
import scipy

np.set_printoptions(precision=15)


jax.config.update("jax_enable_x64", True)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from bspline.matrix_evaluation import (
    matrix_bspline_evaluation_for_dataset,
    matrix_bspline_derivative_evaluation_for_dataset,
)

# jax.config.update("jax_disable_jit", True)

numSamplesPerInterval = 5
numSamplesPerIntervalObj = 30


# def plot_spline(spline, pursuerPosition, pursuerRange, pursuerCaptureRange,pez_limit,useProbabalistic):
def plot_spline(
    spline,
    knownHazards,
    gridPoints,
    fig,
    ax,
):
    t0 = spline.t[spline.k]
    tf = spline.t[-1 - spline.k]
    t = np.linspace(t0, tf, 1000, endpoint=True)
    ax.set_xlabel("X", fontsize=26)
    ax.set_ylabel("Y", fontsize=26)
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)

    pos = spline(t)

    pos = spline(t)
    ax.plot(pos[:, 0], pos[:, 1], "r", label="Path")

    posObjFunc = evaluate_spline(spline.c, spline.t, numSamplesPerIntervalObj)
    Z = batch_hazard_probs(gridPoints, posObjFunc, knownHazards)

    z_size = int(np.sqrt(Z.shape[0]))
    gridPointsX = gridPoints[:, 0].reshape((z_size, z_size))
    gridPointsY = gridPoints[:, 1].reshape((z_size, z_size))
    Z = Z.reshape((z_size, z_size))
    c = ax.pcolormesh(gridPointsX, gridPointsY, Z, vmin=0, vmax=1)
    cbar = fig.colorbar(c, ax=ax, shrink=0.7)
    cbar.set_label("Hazard Probability", fontsize=26)
    cbar.ax.tick_params(labelsize=22)

    ax.scatter(
        knownHazards[:, 0], knownHazards[:, 1], c="r", marker="x", label="Hazard"
    )
    ax.scatter(pos[0, 0], pos[0, 1], c="g", marker="o", label="Start Node")
    ax.scatter(pos[-1, 0], pos[-1, 1], c="b", marker="o", label="End Node")

    ax.set_aspect(1)
    # c = plt.Circle(pursuerPosition, pursuerRange + pursuerCaptureRange, fill=False)
    # ax.add_artist(c)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=20)
    ax.set_title("Optimized Path", fontsize=26)
    fig.set_size_inches(20, 10)


def plot_hazard_prob(knownHazards, gridPoints, fig, ax):
    Z = batch_hazard_probs(gridPoints, np.array([[-100, -100]]), knownHazards)

    z_size = int(np.sqrt(Z.shape[0]))
    gridPointsX = gridPoints[:, 0].reshape((z_size, z_size))
    gridPointsY = gridPoints[:, 1].reshape((z_size, z_size))
    Z = Z.reshape((z_size, z_size))
    c = ax.pcolormesh(gridPointsX, gridPointsY, Z, vmin=0, vmax=1)
    cbar = fig.colorbar(c, ax=ax, shrink=0.7)
    cbar.set_label("Hazard Probability", fontsize=26)
    cbar.ax.tick_params(labelsize=22)


@jax.jit
def safe_norm(x, y, eps=1e-6):
    return jnp.sqrt(jnp.sum((x - y) ** 2) + eps)  # Avoids sqrt(0)


safe_norm_vectorized = jax.vmap(safe_norm, in_axes=(None, 0))


@jax.jit
def prior_hazard_distirbution(x, hazardLocations):
    # If there is prior knowledge of where hazards are located include it as a probability distribution here
    baseProir = 0.1
    maxProir = 0.9
    decayRate = 0.5
    # dists = jnp.linalg.norm(hazardLocations - x, axis=1)
    dists = safe_norm_vectorized(x, hazardLocations)

    singleHazardProbs = jnp.exp(-decayRate * dists)
    combinedHazardProbs = 1 - jnp.prod(1 - singleHazardProbs)

    # scale appropriately
    # prior = baseProir + (maxProir - baseProir) * combinedHazardProbs

    return combinedHazardProbs
    # return prior


@jax.jit
def hazard_at_x_given_searched_at_points(x, points, steepness):
    dists = safe_norm_vectorized(x, points)
    return jnp.exp(-steepness * dists)


@jax.jit
def hazard_posterior_prob(x, searched_points, knownHazards):
    """
    Compute posterior probability of hazard existence at point x given search history.

    Args:
        x: Query point (2D array)
        searched_points: Array of searched locations (N x 2)
        radius: Detection radius of sensor

    Returns:
        Posterior probability (0-1) of hazard existing at x
    """

    steepness = 5.0
    false_alarm = 0.0
    prior = prior_hazard_distirbution(x, knownHazards)

    p_hazard_at_x_givin_points = hazard_at_x_given_searched_at_points(
        x, searched_points, steepness
    )

    # Compute log probabilities safely
    log_p_no_detect_given_hazard = jnp.sum(jnp.log1p(-p_hazard_at_x_givin_points))
    log_p_no_detect_given_no_hazard = searched_points.shape[0] * jnp.log1p(-false_alarm)

    # Bayesian update in log space
    log_numerator = jnp.log(prior) + log_p_no_detect_given_hazard
    log_denominator = jnp.logaddexp(
        log_numerator, jnp.log1p(-prior) + log_p_no_detect_given_no_hazard
    )

    result = jnp.exp(log_numerator - log_denominator)

    return result


# Batch version for multiple points
batch_hazard_probs = jit(jax.vmap(hazard_posterior_prob, in_axes=(0, None, None)))


@partial(jit, static_argnums=(2,))
def evaluate_spline(controlPoints, knotPoints, numSamplesPerInterval):
    knotPoints = knotPoints.reshape((-1,))
    return matrix_bspline_evaluation_for_dataset(
        controlPoints.T, knotPoints, numSamplesPerInterval
    )


@partial(jit, static_argnums=(2, 3))
def evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, derivativeOrder):
    scaleFactor = knotPoints[-splineOrder - 1] / (len(knotPoints) - 2 * splineOrder - 1)
    return matrix_bspline_derivative_evaluation_for_dataset(
        derivativeOrder, scaleFactor, controlPoints.T, knotPoints, numSamplesPerInterval
    )


@partial(jit, static_argnums=(2, 3))
def create_unclamped_knot_points(t0, tf, numControlPoints, splineOrder):
    internalKnots = jnp.linspace(t0, tf, numControlPoints - 2, endpoint=True)
    h = internalKnots[1] - internalKnots[0]
    knots = jnp.concatenate(
        (
            jnp.linspace(t0 - splineOrder * h, t0 - h, splineOrder),
            internalKnots,
            jnp.linspace(tf + h, tf + splineOrder * h, splineOrder),
        )
    )

    return knots


@partial(jit, static_argnums=(2,))
def objective_funtion(controlPoints, tf, splineOrder, knownHazards, gridPoints):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    # dt = knotPoints[3] - knotPoints[0]
    # splineSampledt = 0.5
    # numSamplesPerIntervalObj = (dt / splineSampledt).astype(int)
    pos = evaluate_spline(controlPoints, knotPoints, numSamplesPerIntervalObj)
    return jnp.mean(batch_hazard_probs(gridPoints, pos, knownHazards))


@partial(jit, static_argnums=(2,))
def get_spline_velocity(controlPoints, tf, splineOrder):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    return jnp.linalg.norm(out_d1, axis=1)


@partial(jit, static_argnums=(2,))
def get_spline_path_length(controlPoints, tf, splineOrder):
    numControlPoints = int(len(controlPoints) / 2)
    # controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    # pos = evaluate_spline(controlPoints, knotPoints)
    velocities = get_spline_velocity(controlPoints, tf, splineOrder)
    tSamples = jnp.linspace(
        knotPoints[splineOrder], knotPoints[-splineOrder - 1], len(velocities)
    )
    dt = tSamples[1] - tSamples[0]
    pathLength = jnp.sum((velocities[:-1] + velocities[1:]) * 0.5 * dt)

    return pathLength


@partial(jit, static_argnums=(2,))
def get_spline_turn_rate(controlPoints, tf, splineOrder):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    out_d2 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 2)
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u


@partial(jit, static_argnums=(2,))
def get_spline_curvature(controlPoints, tf, splineOrder):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    out_d2 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 2)
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u / v


@partial(jit, static_argnums=(2,))
def get_spline_heading(controlPoints, tf, splineOrder):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, splineOrder, 1)
    return jnp.arctan2(out_d1[:, 1], out_d1[:, 0])


def get_start_constraint(controlPoints):
    cp1 = controlPoints[0:2]
    cp2 = controlPoints[2:4]
    cp3 = controlPoints[4:6]
    return np.array((1 / 6) * cp1 + (2 / 3) * cp2 + (1 / 6) * cp3)


def initial_velocity_constraint(controlPoints, tf):
    numControlPoints = len(controlPoints) // 2
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, 3)
    # velocityControlPoint1 = (
    #     (3) / (knotPoints[3] - knotPoints[0]) * (controlPoints[1] - controlPoints[0])
    # )
    # velocityControlPoint2 = (
    #     (3) / (knotPoints[3] - knotPoints[0]) * (controlPoints[2] - controlPoints[1])
    # )
    # velocity = 1 / 2 * (velocityControlPoint1 + velocityControlPoint2)
    velocity = evaluate_spline_derivative(controlPoints, knotPoints, 3, 1)
    return velocity[0]


def final_velocity_constraint(controlPoints, tf):
    numControlPoints = len(controlPoints) // 2
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, 3)
    # velocityControlPointNMinus2 = (
    #     (3) / (knotPoints[3] - knotPoints[0]) * (controlPoints[-2] - controlPoints[-3])
    # )
    # velocityControlPointNMinus1 = (
    #     (3) / (knotPoints[3] - knotPoints[0]) * (controlPoints[-1] - controlPoints[-2])
    # )
    # velocity = 1 / 2 * (velocityControlPointNMinus2 + velocityControlPointNMinus1)
    velocity = evaluate_spline_derivative(controlPoints, knotPoints, 3, 1)
    return velocity[-1]


############ Can speed thses up with simple anylitic solution ############
dInitialVelocityDControlPoints = jacfwd(initial_velocity_constraint, argnums=0)
dInitialVelocityDtf = jacfwd(initial_velocity_constraint, argnums=1)
dFinalVelocityDControlPoints = jacfwd(final_velocity_constraint, argnums=0)
dFinalVelocityDtf = jacfwd(final_velocity_constraint, argnums=1)


def get_end_constraint(controlPoints):
    cpnMinus2 = controlPoints[-6:-4]
    cpnMinus1 = controlPoints[-4:-2]
    cpn = controlPoints[-2:]
    return (1 / 6) * cpnMinus2 + (2 / 3) * cpnMinus1 + (1 / 6) * cpn


def get_start_constraint_jacobian(controlPoints):
    numControlPoints = int(controlPoints.shape[0] / 2)
    jac = np.zeros((2, 2 * numControlPoints))
    jac[0, 0] = 1 / 6
    jac[0, 2] = 2 / 3
    jac[0, 4] = 1 / 6
    jac[1, 1] = 1 / 6
    jac[1, 3] = 2 / 3
    jac[1, 5] = 1 / 6
    return jac


def get_end_constraint_jacobian(controlPoints):
    numControlPoints = int(controlPoints.shape[0] / 2)
    jac = np.zeros((2, 2 * numControlPoints))
    jac[0, -6] = 1 / 6
    jac[0, -4] = 2 / 3
    jac[0, -2] = 1 / 6
    jac[1, -5] = 1 / 6
    jac[1, -3] = 2 / 3
    jac[1, -1] = 1 / 6
    return jac


def get_turn_rate_velocity_and_headings(controlPoints, knotPoints):
    out_d1 = evaluate_spline_derivative(controlPoints, knotPoints, 3, 1)
    out_d2 = evaluate_spline_derivative(controlPoints, knotPoints, 3, 2)
    v = np.linalg.norm(out_d1, axis=1)
    u = np.cross(out_d1, out_d2) / (v**2)
    heading = np.arctan2(out_d1[:, 1], out_d1[:, 0])
    return u, v, heading


#
# def vectorized_point_in_convex_polygon(points, polygon_vertices):
#     """
#     Args:
#         points: Shape (N, 2), array of query points.
#         polygon_vertices: Shape (M, 2), ordered vertices of the convex polygon (CCW or CW).
#
#     Returns:
#         Boolean array of shape (N,), where True = inside/on-edge.
#     """
#     polygon_vertices = jnp.asarray(polygon_vertices)
#     points = jnp.asarray(points)
#
#     # Generate edges (V_i to V_{i+1})
#     edges = jnp.roll(polygon_vertices, shift=-1, axis=0) - polygon_vertices  # (M, 2)
#
#     # Compute vectors from polygon vertices to query points
#     vectors = points[jnp.newaxis, :, :] - polygon_vertices[:, jnp.newaxis, :]  # (M, N, 2)
#
#     # Cross products: edge_x * vector_y - edge_y * vector_x
#     cross_products = jnp.cross(edges[:, jnp.newaxis, :], vectors)  # (M, N)
#
#     # Check sign consistency across edges for all points
#     all_non_negative = jnp.all(cross_products >= -1e-8, axis=0)  # Allow small negatives for FP tolerance
#     all_non_positive = jnp.all(cross_products <= 1e-8, axis=0)
#     inside = all_non_negative | all_non_positive  # (N,)
#
#     return inside
#
# dControlPointsDPointsInPolygon = jacfwd(vectorized_point_in_convex_polygon, argnums=0)


def compute_spline_constraints(
    controlPoints,
    knotPoints,
    splineOrder,
):
    pos = evaluate_spline(controlPoints, knotPoints, numSamplesPerInterval)

    turn_rate, velocity, agentHeadings = get_turn_rate_velocity_and_headings(
        controlPoints, knotPoints
    )
    t0 = knotPoints[splineOrder]
    tf = knotPoints[-splineOrder - 1]
    dt = (tf - t0) / numSamplesPerInterval

    curvature = turn_rate / velocity

    tSamples = jnp.linspace(t0, tf, len(velocity))
    dt = tSamples[1] - tSamples[0]
    pathLength = jnp.sum((velocity[:-1] + velocity[1:]) * 0.5 * dt)

    return velocity, turn_rate, curvature, pos, pathLength


def create_spline(knotPoints, controlPoints, order):
    spline = BSpline(knotPoints, controlPoints, order)
    return spline


dVelocityDControlPoints = jacfwd(get_spline_velocity)
sdVelocityDtf = jacfwd(get_spline_velocity, argnums=1)

dTurnRateDControlPoints = jacfwd(get_spline_turn_rate)
dTurnRateTf = jacfwd(get_spline_turn_rate, argnums=1)

dCurvatureDControlPoints = jacfwd(get_spline_curvature)
dCurvatureDtf = jacfwd(get_spline_curvature, argnums=1)

dObjectiveFunctionDControlPoints = jacfwd(objective_funtion, argnums=0)
dObjectiveFunctionDtf = jacfwd(objective_funtion, argnums=1)

dPathLengthDControlPoints = jacfwd(get_spline_path_length, argnums=0)
dPathLengthDtf = jacfwd(get_spline_path_length, argnums=1)


def assure_velocity_constraint(controlPoints, velocityBounds):
    splineOrder = 3
    tf = np.linalg.norm(controlPoints[0] - controlPoints[-1]) / velocityBounds[1]
    v = get_spline_velocity(controlPoints, tf, splineOrder)
    while np.max(v) > velocityBounds[1]:
        tf += 0.01
        v = get_spline_velocity(controlPoints, tf, splineOrder)
    return tf


def move_first_control_point_so_spline_passes_through_start(
    controlPoints, knotPoints, start, startVelocity
):
    num_control_points = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((num_control_points, 2))
    dt = knotPoints[3] - knotPoints[0]
    A = np.array(
        [
            [1 / 6, 0, 2 / 3, 0],
            [0, 1 / 6, 0, 2 / 3],
            [-3 / (2 * dt), 0, 0, 0],
            [0, -3 / (2 * dt), 0, 0],
        ]
    )
    c3x = controlPoints[2, 0]
    c3y = controlPoints[2, 1]

    b = np.array(
        [
            [start[0] - (1 / 6) * c3x],
            [start[1] - (1 / 6) * c3y],
            [startVelocity[0] - 3 / (2 * dt) * c3x],
            [startVelocity[1] - 3 / (2 * dt) * c3y],
        ]
    )

    x = np.linalg.solve(A, b)
    controlPoints[0:2, 0:2] = x.reshape((2, 2))

    return controlPoints


def move_last_control_point_so_spline_passes_through_end(
    controlPoints, knotPoints, end, endVelocity
):
    num_control_points = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((num_control_points, 2))
    dt = knotPoints[3] - knotPoints[0]
    A = np.array(
        [
            [2 / 3, 0, 1 / 6, 0],
            [0, 2 / 3, 0, 1 / 6],
            [0, 0, 3 / (2 * dt), 0],
            [0, 0, 0, 3 / (2 * dt)],
        ]
    )
    cn_minus_2_x = controlPoints[-3, 0]
    cn_minus_2_y = controlPoints[-3, 1]

    b = np.array(
        [
            [end[0] - (1 / 6) * cn_minus_2_x],
            [end[1] - (1 / 6) * cn_minus_2_y],
            [endVelocity[0] + 3 / (2 * dt) * cn_minus_2_x],
            [endVelocity[1] + 3 / (2 * dt) * cn_minus_2_y],
        ]
    )

    x = np.linalg.solve(A, b)
    controlPoints[-2:, -2:] = x.reshape((2, 2))

    return controlPoints


def generate_lawnmower_path(start, end, path_budget, rung_spacing):
    start = np.array(start)
    end = np.array(end)

    # Compute the straight-line distance and unit direction vector
    direction = end - start
    straight_length = np.linalg.norm(direction)
    unit_direction = direction / straight_length

    # Compute the perpendicular direction for rungs
    perp_direction = np.array([-unit_direction[1], unit_direction[0]])

    # Compute number of rungs and spacing
    num_rungs = int(straight_length // rung_spacing)
    rung_positions = np.linspace(0, straight_length, num_rungs + 1)

    # Compute remaining path budget after straight path
    remaining_budget = path_budget - straight_length
    if remaining_budget <= 0:
        print("Path budget is too small for additional rungs.")
        return []

    # Compute the maximum rung length
    max_rung_length = remaining_budget / (num_rungs) if num_rungs > 0 else 0

    # Generate the path as a series of rectangles centered on the straight-line path
    path = []
    flip = 1  # Alternates rung direction
    for i in range(num_rungs + 1):
        base_point = start + rung_positions[i] * unit_direction
        rung_end1 = base_point + (max_rung_length / 2) * perp_direction
        rung_end2 = base_point - (max_rung_length / 2) * perp_direction

        if i == 0:
            path.append(start)
            path.append(
                rung_end2 if flip == 1 else rung_end1
            )  # Ensure correct first rung direction
        elif i == num_rungs:
            path.append(rung_end1 if flip == 1 else rung_end2)
            path.append(end)
        else:
            if flip == 1:
                path.append(rung_end1)
                path.append(rung_end2)
            else:
                path.append(rung_end2)
                path.append(rung_end1)

        flip *= -1  # Flip direction for next rung

    return np.array(path)


def sample_path(path, interval):
    sampled_points = [path[0]]
    total_distance = 0
    for i in range(1, len(path)):
        segment_start = path[i - 1]
        segment_end = path[i]
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)

        while total_distance + segment_length >= len(sampled_points) * interval:
            remaining_distance = (len(sampled_points) * interval) - total_distance
            new_point = (
                segment_start + (remaining_distance / segment_length) * segment_vector
            )
            sampled_points.append(new_point)

        total_distance += segment_length

    return np.array(sampled_points)


def fit_spline_to_path(path, num_control_points, splineOrder, startPoint, endPoint):
    tf = 1
    t = np.linspace(0, tf, len(path))

    # num_control_points = params.numControlPoints
    n_interior_knots = num_control_points - splineOrder - 1
    qs = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
    knots = np.quantile(t, qs)

    s = 0
    tck_x = scipy.interpolate.splrep(t, path[:, 0], k=splineOrder, t=knots, s=s)
    control_points_x = tck_x[1]
    control_points_x = control_points_x[control_points_x != 0]
    control_points_x[0] = startPoint[0]
    control_points_x[-1] = endPoint[0]

    tck_y = scipy.interpolate.splrep(t, path[:, 1], k=splineOrder, t=knots, s=s)
    control_points_y = tck_y[1]
    control_points_y = control_points_y[control_points_y != 0]
    control_points_y[0] = startPoint[1]
    control_points_y[-1] = endPoint[1]
    combined_control_points = np.hstack(
        (
            control_points_x.reshape((len(control_points_x), 1)),
            control_points_y.reshape((len(control_points_y), 1)),
        )
    )

    combined_knot_points = tck_x[0]
    return combined_control_points, combined_knot_points


def create_initial_lawnmower_path(
    startingLocation, endingLocation, numControlPoints, pathBudget, sensingRadius
):
    path = generate_lawnmower_path(
        startingLocation, endingLocation, pathBudget, 2 * sensingRadius
    )
    path = sample_path(path, 0.05)
    splineControlPoints, splineKnotPoints = fit_spline_to_path(
        path, numControlPoints, 3, startingLocation, endingLocation
    )

    plot = False
    if plot:
        fig, ax = plt.subplots()
        ax.plot(path[:, 0], path[:, 1], "r")
        spline = create_spline(splineKnotPoints, splineControlPoints, 3)
        t = np.linspace(splineKnotPoints[3], splineKnotPoints[-4], 1000)
        points = spline(t)
        ax.plot(points[:, 0], points[:, 1], "b")
        plt.show()

    return splineControlPoints


def create_initial_spline(
    startingLocation,
    endingLocation,
    numControlPoints,
    splineOrder,
    initialVelocity,
    finalVelocity,
    velocityConstraints,
    pathBudget,
):
    tf = 1.0
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)

    x0 = create_initial_lawnmower_path(
        startingLocation, endingLocation, numControlPoints, pathBudget, 1.0
    ).flatten()
    # x0 = np.linspace(startingLocation, endingLocation, numControlPoints).flatten()

    tf = assure_velocity_constraint(x0, velocityConstraints)
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    x0 = move_first_control_point_so_spline_passes_through_start(
        x0, knotPoints, startingLocation, initialVelocity
    )
    x0 = x0.flatten()
    x0 = move_last_control_point_so_spline_passes_through_end(
        x0, knotPoints, endingLocation, finalVelocity
    )
    x0 = x0.flatten()
    tf = assure_velocity_constraint(x0, velocityConstraints)
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    x0 = move_first_control_point_so_spline_passes_through_start(
        x0, knotPoints, startingLocation, initialVelocity
    )
    x0 = x0.flatten()
    x0 = move_last_control_point_so_spline_passes_through_end(
        x0, knotPoints, endingLocation, finalVelocity
    )
    x0 = x0.flatten()
    return x0, tf


def optimize_spline_path(
    startingLocation,
    endingLocation,
    initialVelocity,
    finalVelocity,
    numControlPoints,
    splineOrder,
    velocityConstraints,
    turnrateConstraints,
    curvatureConstraints,
    pathLengthConstraint,
    knownHazards,
    gridPoints,
):
    def objfunc(xDict):
        tf = xDict["tf"]
        knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, 3)
        controlPoints = xDict["control_points"]
        initialVelocity = initial_velocity_constraint(controlPoints, tf)
        finalVelocity = final_velocity_constraint(controlPoints, tf)
        funcs = {}
        funcs["start"] = get_start_constraint(controlPoints)
        funcs["end"] = get_end_constraint(controlPoints)
        controlPoints = controlPoints.reshape((numControlPoints, 2))

        velocity, turn_rate, curvature, pos, pathLength = compute_spline_constraints(
            controlPoints, knotPoints, splineOrder
        )
        obj = objective_funtion(
            controlPoints.flatten(), tf, splineOrder, knownHazards, gridPoints
        )

        funcs["obj"] = obj
        funcs["turn_rate"] = turn_rate
        funcs["velocity"] = velocity
        funcs["curvature"] = curvature
        funcs["path_length"] = pathLength
        funcs["final_velocity"] = finalVelocity
        funcs["initial_velocity"] = initialVelocity
        funcs["pos"] = pos
        return funcs, False

    def sens(xDict, funcs):
        funcsSens = {}
        controlPoints = jnp.array(xDict["control_points"])
        tf = xDict["tf"]

        dStartDControlPointsVal = get_start_constraint_jacobian(controlPoints)
        dEndDControlPointsVal = get_end_constraint_jacobian(controlPoints)

        dVelocityDControlPointsVal = dVelocityDControlPoints(
            controlPoints, tf, splineOrder
        )
        dVelocityDtfVal = sdVelocityDtf(controlPoints, tf, splineOrder)
        dTurnRateDControlPointsVal = dTurnRateDControlPoints(
            controlPoints, tf, splineOrder
        )
        dTurnRateDtfVal = dTurnRateTf(controlPoints, tf, splineOrder)
        dCurvatureDControlPointsVal = dCurvatureDControlPoints(
            controlPoints, tf, splineOrder
        )
        dCurvatureDtfVal = dCurvatureDtf(controlPoints, tf, splineOrder)

        dObjectiveFunctionDControlPointsVal = dObjectiveFunctionDControlPoints(
            controlPoints, tf, splineOrder, knownHazards, gridPoints
        )
        failed = False
        if np.any(np.isnan(dObjectiveFunctionDControlPointsVal)):
            print("control points", controlPoints)
            print("tf", tf)
            print(
                "dObjectiveFunctionDControlPointsVal",
                dObjectiveFunctionDControlPointsVal,
            )
            failed = True
            pos = funcs["pos"]
            dists = np.sqrt(
                ((gridPoints[:, np.newaxis, :] - pos[np.newaxis, :, :]) ** 2).sum(
                    axis=2
                )
            )
            print("dists", np.min(dists))
        dObjectiveFunctionDtfVal = dObjectiveFunctionDtf(
            controlPoints, tf, splineOrder, knownHazards, gridPoints
        )

        dPathLengthDControlPointsVal = dPathLengthDControlPoints(
            controlPoints, tf, splineOrder
        )
        dPathLengthDtfVal = dPathLengthDtf(controlPoints, tf, splineOrder)

        dInitialVelocityDControlPointsVal = dInitialVelocityDControlPoints(
            controlPoints, tf
        )
        dInitialVelocityDtfVal = dInitialVelocityDtf(controlPoints, tf)
        dFinalVelocityDControlPointsVal = dFinalVelocityDControlPoints(
            controlPoints, tf
        )
        dFinalVelocityDtfVal = dFinalVelocityDtf(controlPoints, tf)

        funcsSens["obj"] = {
            "control_points": dObjectiveFunctionDControlPointsVal,
            "tf": dObjectiveFunctionDtfVal,
        }
        funcsSens["start"] = {
            "control_points": dStartDControlPointsVal,
            "tf": np.zeros((2, 1)),
        }
        funcsSens["end"] = {
            "control_points": dEndDControlPointsVal,
            "tf": np.zeros((2, 1)),
        }
        funcsSens["velocity"] = {
            "control_points": dVelocityDControlPointsVal,
            "tf": dVelocityDtfVal,
        }
        funcsSens["turn_rate"] = {
            "control_points": dTurnRateDControlPointsVal,
            "tf": dTurnRateDtfVal,
        }
        funcsSens["curvature"] = {
            "control_points": dCurvatureDControlPointsVal,
            "tf": dCurvatureDtfVal,
        }
        funcsSens["path_length"] = {
            "control_points": dPathLengthDControlPointsVal,
            "tf": dPathLengthDtfVal,
        }
        funcsSens["initial_velocity"] = {
            "control_points": dInitialVelocityDControlPointsVal,
            "tf": dInitialVelocityDtfVal,
        }
        funcsSens["final_velocity"] = {
            "control_points": dFinalVelocityDControlPointsVal,
            "tf": dFinalVelocityDtfVal,
        }

        return funcsSens, failed

    x0, tf = create_initial_spline(
        startingLocation,
        endingLocation,
        numControlPoints,
        splineOrder,
        initialVelocity,
        finalVelocity,
        velocityConstraints,
        pathLengthConstraint,
    )

    # get size of constraints
    tempVelocityContstraints = get_spline_velocity(x0, 1, 3)
    num_constraint_samples = len(tempVelocityContstraints)

    optProb = Optimization("path optimization", objfunc)

    optProb.addVarGroup(
        name="control_points",
        nVars=2 * (numControlPoints),
        varType="c",
        value=x0,
        lower=-50,
        upper=50,
    )
    optProb.addVarGroup(name="tf", nVars=1, varType="c", value=tf, lower=0, upper=None)
    #
    optProb.addConGroup(
        "velocity",
        num_constraint_samples,
        lower=velocityConstraints[0],
        upper=velocityConstraints[1],
        scale=1.0 / velocityConstraints[1],
    )
    optProb.addConGroup(
        "turn_rate",
        num_constraint_samples,
        lower=turnrateConstraints[0],
        upper=turnrateConstraints[1],
        scale=1.0 / turnrateConstraints[1],
    )
    optProb.addConGroup(
        "curvature",
        num_constraint_samples,
        lower=curvatureConstraints[0],
        upper=curvatureConstraints[1],
        scale=1.0 / curvatureConstraints[1],
    )
    optProb.addConGroup("start", 2, lower=startingLocation, upper=startingLocation)
    optProb.addConGroup("end", 2, lower=endingLocation, upper=endingLocation)
    optProb.addConGroup("path_length", 1, lower=0, upper=pathLengthConstraint)
    optProb.addConGroup("final_velocity", 2, lower=finalVelocity, upper=finalVelocity)
    optProb.addConGroup(
        "initial_velocity", 2, lower=initialVelocity, upper=initialVelocity
    )

    optProb.addObj("obj", scale=1.0)

    opt = OPT("ipopt")
    opt.options["print_level"] = 0
    opt.options["max_iter"] = 1000
    opt.options["tol"] = 1e-6
    username = getpass.getuser()
    opt.options["hsllib"] = (
        "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
    )
    opt.options["linear_solver"] = "ma97"
    # opt.options["derivative_test_perturbation"] = 1e-3
    # opt.options["derivative_test"] = "first-order"

    sol = opt(optProb, sens=sens)
    # sol = opt(optProb, sens="FD")
    print(sol)
    if sol.optInform["value"] != 0:
        print("Optimization failed")

    knotPoints = create_unclamped_knot_points(
        0, sol.xStar["tf"][0], numControlPoints, 3
    )
    controlPoints = sol.xStar["control_points"].reshape((numControlPoints, 2))
    return create_spline(knotPoints, controlPoints, splineOrder)
    # controlPoints = x0.reshape((numControlPoints, 2))
    # knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, 3)
    # return create_spline(knotPoints, controlPoints, splineOrder)


def main():
    startingLocation = np.array([0.0, 0.0])
    endingLocation = np.array([10.0, 5.0])
    initialVelocity = endingLocation - startingLocation
    agentSpeed = 1.0
    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed
    print("Initial velocity", initialVelocity)
    velocityConstraints = np.array([0.5, 1.0])

    numControlPoints = 24
    splineOrder = 3
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    boundsX = (-1, 10)
    boundsY = (-1, 6)

    pathLengthConstraint = 2.5 * np.linalg.norm(endingLocation - startingLocation)

    numKnownHazards = 2
    knownHazardsX = np.random.uniform(boundsX[0], boundsX[1], (numKnownHazards,))
    knownHazardsY = np.random.uniform(boundsY[0], boundsY[1], (numKnownHazards,))
    knownHazards = np.array([knownHazardsX, knownHazardsY]).T
    knownHazards = np.vstack([startingLocation, knownHazards, endingLocation])
    # knownHazards = np.vstack([startingLocation, endingLocation])

    gridPointsX = jnp.linspace(boundsX[0], boundsX[1], 50)
    gridPointsY = jnp.linspace(boundsY[0], boundsY[1], 50)
    [X, Y] = jnp.meshgrid(gridPointsX, gridPointsY)
    gridPoints = jnp.array([X.flatten(), Y.flatten()]).T

    # spline = optimize_spline_path(
    #     startingLocation,
    #     endingLocation,
    #     initialVelocity,
    #     initialVelocity,
    #     numControlPoints,
    #     splineOrder,
    #     velocityConstraints,
    #     turn_rate_constraints,
    #     curvature_constraints,
    #     pathLengthConstraint,
    #     knownHazards,
    #     gridPoints,
    # )
    # start = time.time()
    # spline = optimize_spline_path(
    #     startingLocation,
    #     endingLocation,
    #     initialVelocity,
    #     initialVelocity,
    #     numControlPoints,
    #     splineOrder,
    #     velocityConstraints,
    #     turn_rate_constraints,
    #     curvature_constraints,
    #     pathLengthConstraint,
    #     knownHazards,
    #     gridPoints,
    # )
    # print("Time to optimize spline", time.time() - start)
    #
    fig, ax = plt.subplots()
    plot_hazard_prob(knownHazards, gridPoints, fig, ax)
    # plot_spline(spline, knownHazards, gridPoints, fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
