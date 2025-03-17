import numpy as np
import matplotlib.pyplot as plt
import time


import path_planner
import routing_strategy


def main():
    domain = (0, 100, 0, 100)
    num_original = 3
    num_vehicles = 1
    max_distance = 500

    original_nodes = routing_strategy.generate_random_nodes(
        domain, total_nodes=num_original, seed=12
    )
    edges, path_budget, cellXYList = routing_strategy.find_edges_budget_and_cells(
        original_nodes, domain, max_distance
    )
    print("route found")

    # path planning parameters
    agentSpeed = 8.0
    initialVelocity = edges[0][1] - edges[0][0]
    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed
    velocityConstraints = np.array([0.5, 10.0])

    numControlPoints = 20
    splineOrder = 3
    splineSampledt = 0.1
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)
    #
    # totalPathBudget = 0
    # totalStraitLineDistance = 0
    # for i, edge in enumerate(edges):
    #     print("i", i)
    #     totalPathBudget += path_budget[i]
    #     print("path budget", path_budget[i])
    #     strait_line_distance = np.linalg.norm(edges[i][1] - edges[i][0])
    #     print("strait line distance", strait_line_distance)
    #     totalStraitLineDistance += strait_line_distance
    # print("Total path budget", totalPathBudget)
    # print("Total strait line distance", totalStraitLineDistance)
    #

    splines = []
    for i in range(len(edges)):
        edge = edges[i]

        if i == len(edges) - 1:
            nextVelocity = initialVelocity
        else:
            nextVelocity = edges[i + 1][1] - edges[i + 1][0]
            nextVelocity = nextVelocity / np.linalg.norm(nextVelocity) * agentSpeed

        straitLineDistance = np.linalg.norm(edge[1] - edge[0])
        print("strait line distance", straitLineDistance)
        print("path budget", path_budget[i])

        start = time.time()

        spline = path_planner.optimize_spline_path(
            startingLocation=edge[0],
            endingLocation=edge[1],
            initialVelocity=initialVelocity,
            finalVelocity=nextVelocity,
            numControlPoints=numControlPoints,
            splineOrder=splineOrder,
            velocityConstraints=velocityConstraints,
            turnrateConstraints=turn_rate_constraints,
            curvatureConstraints=curvature_constraints,
            pathLengthConstraint=path_budget[i],
            knownHazards=original_nodes,
            gridPoints=cellXYList[i],
            splineSampledt=splineSampledt,
        )
        print("time to optimize spline", time.time() - start)
        initialVelocity = nextVelocity
        splines.append(spline)

    fig, ax = plt.subplots()
    # plot_hazard_prob(knownHazards, gridPoints, fig, ax)
    for i, spline in enumerate(splines):
        path_planner.plot_spline(
            spline,
            original_nodes,
            cellXYList[i],
            splineSampledt,
            fig,
            ax,
            plotColorbar=False,
        )
    plt.show()


if __name__ == "__main__":
    main()
