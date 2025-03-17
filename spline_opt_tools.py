import jax.numpy as jnp
from jax import jit, jacfwd
import jax
from functools import partial
import numpy as np

from jax_b_splines.bspline.matrix_evaluation import (
    matrix_bspline_evaluation_at_times,
    matrix_bspline_derivative_evaluation_at_times,
)


@partial(jit, static_argnums=(3,))
def evaluate_spline(times, controlPoints, knotPoints, spline_order):
    knotPoints = knotPoints.reshape((-1,))
    return matrix_bspline_evaluation_at_times(
        times, controlPoints, knotPoints, spline_order
    )


@partial(jit, static_argnums=(4,))
def evaluate_spline_derivative(
    times,
    derivativeOrder,
    controlPoints,
    knotPoints,
    splineOrder,
):
    knotPoints = knotPoints.reshape((-1,))
    return matrix_bspline_derivative_evaluation_at_times(
        times, derivativeOrder, controlPoints, knotPoints, splineOrder
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


@partial(jit, static_argnums=(2, 3))
def get_spline_velocity(controlPoints, tf, splineOrder, numSamples):
    times = jnp.linspace(0, tf, numSamples).flatten()
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        times, 1, controlPoints, knotPoints, splineOrder
    )
    jax.debug.print("out_d1 {x}", x=out_d1)
    return jnp.linalg.norm(out_d1, axis=1)


@partial(jit, static_argnums=(2, 3))
def get_spline_turn_rate(controlPoints, tf, splineOrder, numSamples):
    times = jnp.linspace(0, tf, numSamples).flatten()
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        times, 1, controlPoints, knotPoints, splineOrder
    )
    out_d2 = evaluate_spline_derivative(
        times, 2, controlPoints, knotPoints, splineOrder
    )
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u


@partial(jit, static_argnums=(2, 3))
def get_spline_curvature(controlPoints, tf, splineOrder, numSamples):
    times = jnp.linspace(0, tf, numSamples).flatten()
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        times, 1, controlPoints, knotPoints, splineOrder
    )
    out_d2 = evaluate_spline_derivative(
        times, 2, controlPoints, knotPoints, splineOrder
    )
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u / v


@partial(jit, static_argnums=(2, 3))
def get_spline_heading(controlPoints, tf, splineOrder, numSamples):
    times = jnp.linspace(0, tf, numSamples).flatten()
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        times, 1, controlPoints, knotPoints, splineOrder
    )
    return jnp.arctan2(out_d1[:, 1], out_d1[:, 0])


def get_start_constraint(controlPoints):
    cp1 = controlPoints[0:2]
    cp2 = controlPoints[2:4]
    cp3 = controlPoints[4:6]
    return np.array((1 / 6) * cp1 + (2 / 3) * cp2 + (1 / 6) * cp3)


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


@partial(jax.jit, static_argnums=(2,))
def initial_velocity_constraint(controlPoints, tf, splineOrder):
    jax.debug.print("splineOrder {x}", x=splineOrder)
    numControlPoints = len(controlPoints) // 2
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, 3)
    velocity = evaluate_spline_derivative(
        jnp.array([0]), 1, controlPoints, knotPoints, splineOrder
    )
    return velocity[0]


@partial(jax.jit, static_argnums=(2,))
def final_velocity_constraint(controlPoints, tf, splineOrder):
    jax.debug.print("splineOrder {x}", x=splineOrder)
    numControlPoints = len(controlPoints) // 2
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    velocity = evaluate_spline_derivative(
        jnp.array([tf]), 1, controlPoints, knotPoints, splineOrder
    )
    return velocity[-1]


dInitialVelocityDControlPoints = jax.jit(
    jacfwd(initial_velocity_constraint, argnums=0), static_argnums=(2,)
)
dInitialVelocityDtf = jax.jit(
    jacfwd(initial_velocity_constraint, argnums=1), static_argnums=(2,)
)
dFinalVelocityDControlPoints = jax.jit(
    jacfwd(final_velocity_constraint, argnums=0), static_argnums=(2,)
)
dFinalVelocityDtf = jax.jit(
    jacfwd(final_velocity_constraint, argnums=1), static_argnums=(2,)
)


def get_turn_rate_velocity_and_headings(times, controlPoints, knotPoints, splineOrder):
    out_d1 = evaluate_spline_derivative(
        times, 1, controlPoints, knotPoints, splineOrder
    )
    out_d2 = evaluate_spline_derivative(
        times, 2, controlPoints, knotPoints, splineOrder
    )
    v = np.linalg.norm(out_d1, axis=1)
    u = np.cross(out_d1, out_d2) / (v**2)
    heading = np.arctan2(out_d1[:, 1], out_d1[:, 0])
    return u, v, heading


dVelocityDControlPoints = jacfwd(get_spline_velocity)
dVelocityDtf = jacfwd(get_spline_velocity, argnums=1)


dTurnRateDControlPoints = jacfwd(get_spline_turn_rate)
dTurnRateTf = jacfwd(get_spline_turn_rate, argnums=1)

dCurvatureDControlPoints = jacfwd(get_spline_curvature)
dCurvatureDtf = jacfwd(get_spline_curvature, argnums=1)


def assure_velocity_constraint(
    controlPoints,
    splineOrder,
    t0,
    tf,
    numSamples,
    agentSpeed,
    velocityBounds,
):
    splineOrder = 3
    tf = np.linalg.norm(controlPoints[0] - controlPoints[-1]) / agentSpeed / 2
    print("tf", tf)
    v = get_spline_velocity(controlPoints, tf, splineOrder, numSamples)
    print("max v", np.max(v))
    while np.max(v) > velocityBounds[1]:
        tf += 0.1
        # combined_knot_points = self.create_unclamped_knot_points(0, tf, num_control_points,params.splineOrder)
        v = get_spline_velocity(controlPoints, tf, splineOrder, numSamples)
        print("v", v)
        print("max v", np.max(v))
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
