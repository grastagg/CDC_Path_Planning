import numpy as np
import matplotlib.pyplot as plt


import path_planner
import routing_strategy


def main():
    domain = (0, 100, 0, 100)
    num_original = 10
    num_vehicles = 1
    max_distance = 1000

    original_nodes = routing_strategy.generate_random_nodes(
        domain, total_nodes=num_original, seed=2
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
    for i in range(1):
        edge = edges[i]

        nextVelocity = edges[i + 1][1] - edges[i + 1][0]
        nextVelocity = nextVelocity / np.linalg.norm(nextVelocity) * agentSpeed
        print("strait line distance", np.linalg.norm(edge[1] - edge[0]))
        print("path budget", path_budget[i])
        print("grid points", cellXYList[i].shape)
        print("starting location", edge[0])
        print("ending location", edge[1])
        print("initial velocity", initialVelocity)
        print("next velocity", nextVelocity)

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
        )
        fig, ax = plt.subplots()
        # plot_hazard_prob(knownHazards, gridPoints, fig, ax)
        path_planner.plot_spline(spline, original_nodes, cellXYList[i], fig, ax)
        plt.show()
        print(edge)


if __name__ == "__main__":
    main()
