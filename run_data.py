import numpy as np
import json
import matplotlib.pyplot as plt
import time
import os


import path_planner
import routing_strategy


def sample_hidden_nodes(original_nodes, num_hidden_nodes, domain, seed=2):
    hidden_nodes = []
    num_nodes = 0
    numGrid = 1000
    gridX = np.linspace(domain[0], domain[1], numGrid)
    gridY = np.linspace(domain[2], domain[3], numGrid)
    [X, Y] = np.meshgrid(gridX, gridY)
    count = 0
    candidate_nodes = np.column_stack((X.ravel(), Y.ravel()))

    probs = path_planner.prior_hazard_prob_vec(candidate_nodes, original_nodes)
    probs = probs / np.sum(probs)
    sampled_indices = np.random.choice(len(candidate_nodes), num_hidden_nodes, p=probs)
    sampled_nodes = candidate_nodes[sampled_indices]
    return sampled_nodes


def plot_known_and_hidden_nodes(original_nodes, hidden_nodes, domain):
    fig, ax = plt.subplots()
    numGrid = 1000
    gridX = np.linspace(domain[0], domain[1], numGrid)
    gridY = np.linspace(domain[2], domain[3], numGrid)
    [X, Y] = np.meshgrid(gridX, gridY)

    gird_locations = np.column_stack((X.ravel(), Y.ravel()))
    probs = path_planner.prior_hazard_prob_vec(gird_locations, original_nodes)
    c = ax.pcolormesh(X, Y, probs.reshape(numGrid, numGrid), vmin=0, vmax=1)
    cbar = fig.colorbar(c, ax=ax)

    ax.scatter(hidden_nodes[:, 0], hidden_nodes[:, 1], c="red", label="hidden")
    ax.scatter(original_nodes[:, 0], original_nodes[:, 1], c="blue", label="original")
    ax.set_aspect("equal")
    ax.set_title("Prior Hazard Distribution", fontsize=30)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    # set x and y tick font size
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    # plt.show()


def plot_route(edges):
    fig, ax = plt.subplots()
    for edge in edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c="blue")
    ax.set_aspect("equal")


def plot_combined_paths(
    splines, original_nodes, splineSampledt, hidden_nodes, node_found, domain
):
    fig, ax = plt.subplots()
    # plot_hazard_prob(knownHazards, gridPoints, fig, ax)
    sampledPoints = sample_splines(splines, splineSampledt)

    x = np.linspace(domain[0], domain[1], 1000)
    y = np.linspace(domain[2], domain[3], 1000)
    [X, Y] = np.meshgrid(x, y)
    grid_locations = np.column_stack((X.ravel(), Y.ravel()))
    probs = path_planner.batch_hazard_probs(
        grid_locations, sampledPoints, original_nodes
    )
    ax.pcolormesh(X, Y, probs.reshape(1000, 1000))
    ax.scatter(sampledPoints[:, 0], sampledPoints[:, 1])
    ax.scatter(hidden_nodes[:, 0], hidden_nodes[:, 1], c=node_found, cmap="cool")
    ax.set_aspect("equal")


def sample_splines(splines, splineSampledt):
    sampledPoints = []
    for i, spline in enumerate(splines):
        spline = splines[i]
        t0 = 0
        tf = spline.t[-splines[i].k - 1]
        numPoints = int((tf - t0) / splineSampledt)
        t = np.linspace(t0, tf, numPoints)
        sample = spline(t)
        sampledPoints.append(sample)
    sampledPoints = np.vstack(sampledPoints)
    return sampledPoints


def evaluate_path(splines, hidden_nodes, splineSampledt):
    sampledPoints = sample_splines(splines, splineSampledt)

    node_found = np.zeros(len(hidden_nodes))
    for i in range(len(hidden_nodes)):
        probs = path_planner.hazard_at_x_given_searched_at_points(
            hidden_nodes[i], sampledPoints, path_planner.steepness
        )
        ran = np.random.uniform(0, 1, len(probs))
        found = np.any(ran < probs)
        node_found[i] = found
    return np.sum(node_found) / len(hidden_nodes), node_found


def run_test(index, filename):
    domain = (0, 1000, 0, 1000)
    # filename = "original_5_virtual_1"
    with open("data/" + filename + ".json") as f:
        data = json.load(f)[0]
        original_nodes = np.array(data["original_nodes"])
        print("original nodes", original_nodes)
        hidden_nodes = sample_hidden_nodes(
            original_nodes, num_hidden_nodes=200, domain=domain
        )
        virtual_nodes = np.array(data["edge_virtual_nodes"])
        combinded_nodes = np.vstack([original_nodes, virtual_nodes])

        plot_known_and_hidden_nodes(original_nodes, hidden_nodes, domain)

        route = data["routes_cvt"]
        edges = []
        budgets = data["final_budget"]
        for i in range(len(route) - 1):
            if route[i] == 0:
                start = [0.0, 0.0]
            else:
                start = combinded_nodes[route[i] - 1]
            if route[i + 1] == 0:
                end = [0.0, 0.0]
            else:
                end = combinded_nodes[route[i + 1] - 1]
            edges.append((start, end))

    # path planning parameters
    agentSpeed = 8.0
    initialVelocity = edges[0][1] - edges[0][0]
    initialVelocity = initialVelocity / np.linalg.norm(initialVelocity) * agentSpeed
    velocityConstraints = np.array([0.5, 10.0])

    numControlPoints = 25
    splineOrder = 3
    splineSampledt = 0.1
    turn_rate_constraints = (-50.0, 50.0)
    curvature_constraints = (-10.0, 10.0)

    edge_vor, cellPoints = routing_strategy.approximate_edge_voronoi(
        edges, domain, grid_size=100
    )
    cellXYList = [cellPoints[edge_vor == i] for i in range(len(edges))]

    splines = []
    lawnMowerSPlines = []
    straitSplines = []
    for i in range(len(edges)):
        # for i in range(1):
        edge = edges[i]

        if i == len(edges) - 1:
            nextVelocity = initialVelocity
        else:
            nextVelocity = edges[i + 1][1] - edges[i + 1][0]
            nextVelocity = nextVelocity / np.linalg.norm(nextVelocity) * agentSpeed

        lm_spline = path_planner.optimize_spline_path(
            startingLocation=edge[0],
            endingLocation=edge[1],
            initialVelocity=initialVelocity,
            finalVelocity=nextVelocity,
            numControlPoints=numControlPoints,
            splineOrder=splineOrder,
            velocityConstraints=velocityConstraints,
            turnrateConstraints=turn_rate_constraints,
            curvatureConstraints=curvature_constraints,
            pathLengthConstraint=budgets[i],
            knownHazards=original_nodes,
            gridPoints=cellXYList[i],
            splineSampledt=splineSampledt,
            lawnMowerPath=True,
            straightLine=False,
        )
        strait_spline = path_planner.optimize_spline_path(
            startingLocation=edge[0],
            endingLocation=edge[1],
            initialVelocity=initialVelocity,
            finalVelocity=nextVelocity,
            numControlPoints=numControlPoints,
            splineOrder=splineOrder,
            velocityConstraints=velocityConstraints,
            turnrateConstraints=turn_rate_constraints,
            curvatureConstraints=curvature_constraints,
            pathLengthConstraint=budgets[i],
            knownHazards=original_nodes,
            gridPoints=cellXYList[i],
            splineSampledt=splineSampledt,
            lawnMowerPath=False,
            straightLine=True,
        )
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
            pathLengthConstraint=budgets[i],
            knownHazards=original_nodes,
            gridPoints=cellXYList[i],
            splineSampledt=splineSampledt,
            lawnMowerPath=False,
            straightLine=False,
        )
        print("time to optimize spline", time.time() - start)
        lawnMowerSPlines.append(lm_spline)
        splines.append(spline)
        straitSplines.append(strait_spline)
        initialVelocity = nextVelocity

    percent_found, node_found = evaluate_path(splines, hidden_nodes, splineSampledt)
    percent_found_lm, node_found_lm = evaluate_path(
        lawnMowerSPlines, hidden_nodes, splineSampledt
    )
    percent_found_strait, node_found_strait = evaluate_path(
        straitSplines, hidden_nodes, splineSampledt
    )
    print("optimized percent found", percent_found)
    print("lawn mower percent found", percent_found_lm)
    print("strait line percent found", percent_found_strait)

    lawnMowerFileName = "lawnmower.txt"
    optimizedFileName = "optimized.txt"
    straitLineFileName = "strait_line.txt"

    saveData = False
    if saveData:
        saveFolder = "processedData/"
        if not os.path.exists(saveFolder + filename):
            os.mkdir(saveFolder + filename)
        with open(saveFolder + filename + "/" + lawnMowerFileName, "a") as f:
            f.write(f"{percent_found_lm}\n")
        with open(saveFolder + filename + "/" + optimizedFileName, "a") as f:
            f.write(f"{percent_found}\n")
        with open(saveFolder + filename + "/" + straitLineFileName, "a") as f:
            f.write(f"{percent_found_strait}\n")
    else:
        plot_combined_paths(
            splines, original_nodes, splineSampledt, hidden_nodes, node_found, domain
        )
        plt.show()


def run_all(filename):
    for i in range(100):
        run_test(i, filename)


def run_all_different_numbers():
    folder_path = "data/"
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            name_without_ext = os.path.splitext(file)[0]
            print(name_without_ext)
            run_all(name_without_ext)


def compare_data(folder):
    opdata = np.genfromtxt(folder + "optimized.txt")
    lmdata = np.genfromtxt(folder + "lawnmower.txt")
    strait_line = np.genfromtxt(folder + "strait_line.txt")

    print("mean optimized", np.mean(opdata))
    print("mean lawn mower", np.mean(lmdata))
    print("mean strait line", np.mean(strait_line))


if __name__ == "__main__":
    run_test(1, "original_10_virtual_1")
    #
    # folder = "processedData/original_5_virtual_3/"
    # compare_data(folder)
    # run_all(folder)
    # run_all_different_numbers()
