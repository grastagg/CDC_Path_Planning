import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_plots(file, title, ax):
    def load_means(file, strategy):
        return np.array(
            [
                np.mean(np.genfromtxt(f"fig6Data/{file}_virtual_{i}/{strategy}.txt"))
                for i in range(4)
            ]
        )

    optPlot = load_means(file, "optimized")
    lmPlot = load_means(file, "lawnmower")
    straitPlot = load_means(file, "strait_line")

    ax.plot(range(4), optPlot, label="Optimized", marker="o")
    ax.plot(range(4), lmPlot, label="Lawn Mower", marker="s")
    ax.plot(range(4), straitPlot, label="Strait Line", marker="^")

    ax.set_title(title, fontsize=24)
    ax.set_ylim(0.2, 0.7)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xlabel("Number of Virtual Nodes", fontsize=20)
    ax.set_ylabel("Percent Found", fontsize=20)
    ax.legend(fontsize=16, loc="lower right")


def plot_final_comparison():
    # this is for fig 6
    fig, axis = plt.subplots(1, 3, figsize=(20, 6))
    generate_plots("original_5", title="Five Known Hazards", ax=axis[0])
    generate_plots("original_10", title="Ten Known Hazards", ax=axis[1])
    generate_plots("original_15", title="Fifteen Known Hazards", ax=axis[2])


def plot_known_and_hidden_nodes():
    # this is for fig 3
    fig, ax = plt.subplots()
    X = np.genfromtxt("fig3Data/X.txt")
    Y = np.genfromtxt("fig3Data/Y.txt")
    probs = np.genfromtxt("fig3Data/probs.txt")
    hidden_nodes = np.genfromtxt("fig3Data/hidden_nodes.txt")
    original_nodes = np.genfromtxt("fig3Data/original_nodes.txt")

    c = ax.pcolormesh(X, Y, probs, vmin=0, vmax=1)
    cbar = fig.colorbar(c, ax=ax)
    # increase cbar font size
    cbar.ax.tick_params(labelsize=20)
    # set label of cbar
    cbar.set_label("Unkwown Hazard Probability", fontsize=20)

    ax.scatter(hidden_nodes[:, 0], hidden_nodes[:, 1], c="red", label="Unknown Hazard")
    ax.scatter(
        original_nodes[:, 0], original_nodes[:, 1], c="blue", label="Known Hazard"
    )
    ax.set_aspect("equal")
    ax.set_title("Prior Hazard Distribution", fontsize=30)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    # set x and y tick font size
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.legend(fontsize=14)


def plot_combined_paths(
    title,
    ax,
    fig,
    plotColorbar=False,
):
    print("TEST", "fig5Data/" + title + "/X.txt")
    X = np.genfromtxt("fig5Data/" + title + "/X.txt")
    Y = np.genfromtxt("fig5Data/" + title + "/Y.txt")
    probs = np.genfromtxt("fig5Data/" + title + "/probs.txt")
    hidden_nodes_found = np.genfromtxt("fig5Data/" + title + "/hidden_nodes_found.txt")
    hidden_nodes_not_found = np.genfromtxt(
        "fig5Data/" + title + "/hidden_nodes_not_found.txt", dtype=bool
    )
    known_hazards = np.genfromtxt("fig5Data/" + title + "/known_hazards.txt")
    psuedo_nodes = np.genfromtxt("fig5Data/" + title + "/psuedo_nodes.txt")
    sampledPoints = np.genfromtxt("fig5Data/" + title + "/sampled_points.txt")

    c = ax.pcolormesh(X, Y, probs, vmin=0, vmax=1)

    ax.scatter(sampledPoints[:, 0], sampledPoints[:, 1])
    ax.scatter(
        hidden_nodes_found[:, 0],
        hidden_nodes_found[:, 1],
        c="green",
        label="Found",
    )
    ax.scatter(
        hidden_nodes_not_found[:, 0],
        hidden_nodes_not_found[:, 1],
        c="yellow",
        label="Not Found",
    )
    ax.scatter(
        known_hazards[:, 0],
        known_hazards[:, 1],
        s=100,
        c="blue",
        label="Known Hazards",
    )
    ax.scatter(
        psuedo_nodes[:, 0],
        psuedo_nodes[:, 1],
        s=100,
        marker="s",
        c="r",
        label="Psuedo Node",
    )

    ax.set_aspect("equal")
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_title(title, fontsize=26)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # size and padding
    cbar = fig.colorbar(c, cax=cax)  # attach colorbar to the new axis
    cbar.ax.tick_params(labelsize=20)
    if plotColorbar:
        cbar.set_label("Unknown Hazard Probability", fontsize=20)
        ax.legend(fontsize=12)


def plot_paths():
    fig, axis = plt.subplots(1, 3, figsize=(22, 6))
    plot_combined_paths("Optimized Paths", ax=axis[0], fig=fig)
    plot_combined_paths("Lawnmower Paths", ax=axis[1], fig=fig)
    plot_combined_paths("Straight Paths", ax=axis[2], fig=fig, plotColorbar=True)
    plt.tight_layout()


if __name__ == "__main__":
    # code for figure 2
    plot_known_and_hidden_nodes()

    # code for figure 5
    plot_paths()

    # cope for figure 6
    plot_final_comparison()
    plt.show()
