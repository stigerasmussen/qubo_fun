# Import the Dwave packages
import dimod
import neal
import dwavebinarycsp

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from ising import plot_enumerate, plot_energies


def toQUBO(A, b, c):
    rho = np.sum(np.abs(c)) + 1
    Q = rho * np.matmul(A.T, A)
    Q += np.diag(c)
    Q -= rho * 2 * np.diag(np.matmul(b.T, A))
    cQ = rho * np.matmul(b.T, b)
    return Q, cQ


def not_both_1(v, u):
    return not (v and u)


def plot_map(sample):
    """ Function that plots a returned sample as a colored graph """
    # Translate from binary to integer color representation
    color_map = {}
    for node in V:
          for i in range(colors):
            if sample['x'+str(node)+','+str(i)]:
                color_map[node] = i
    # Plot the sample with color-coded nodes
    node_colors = [color_map.get(node) for node in G.nodes()]
    nx.draw(G, with_labels=True, pos=layout, node_color=node_colors)
    plt.show()


if __name__ == "__main__":
    # # Define problem
    # A = np.array([[1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    #               [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    #               [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]])
    # b = np.array([1, 1, 1])
    # c = np.array([2, 4, 4, 4, 4, 4, 5, 4, 5, 6, 5])
    #
    # Q, cQ = toQUBO(A, b, c)
    #
    # # G = nx.from_numpy_matrix(Q)
    # # nx.draw(G, with_labels=True)
    #
    # model = dimod.BinaryQuadraticModel.from_qubo(Q, offset=cQ)
    simAnnSampler = neal.SimulatedAnnealingSampler()
    # simAnnSamples = simAnnSampler.sample(model, num_reads=1000)
    # plot_enumerate(simAnnSamples, title='Simulated annealing in default parameters')
    # plot_energies(simAnnSamples, title='Simulated annealing in default parameters')

    V = range(1, 12 + 1)
    E = [(1, 2), (2, 3), (1, 4), (1, 6), (1, 12), (2, 5), (2, 7), (3, 8), (3, 10), (4, 11), (4, 9), (5, 6), (6, 7),
         (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (5, 12), (5, 9), (6, 10), (7, 11), (8, 12)]
    layout = {i: [np.cos((2 * i + 1) * np.pi / 8), np.sin((2 * i + 1) * np.pi / 8)] for i in np.arange(5, 13)}
    layout[1] = [-1.5, 1.5]
    layout[2] = [1.5, 1.5]
    layout[3] = [1.5, -1.5]
    layout[4] = [-1.5, -1.5]
    G = nx.Graph()
    G.add_edges_from(E)
    # nx.draw(G, with_labels=True, pos=layout)

    # Valid configurations for the constraint that each node select a single color, in this case we want to use 3 colors
    one_color_configurations = {(0, 0, 1), (0, 1, 0), (1, 0, 0)}
    colors = len(one_color_configurations)

    # Create a binary constraint satisfaction problem
    csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

    # Add constraint that each node select a single color
    for node in V:
        variables = ['x' + str(node) + ',' + str(i) for i in range(colors)]
        csp.add_constraint(one_color_configurations, variables)

    # Add constraint that each pair of nodes with a shared edge not both select one color
    for edge in E:
        v, u = edge
        for i in range(colors):
            variables = ['x' + str(v) + ',' + str(i), 'x' + str(u) + ',' + str(i)]
            csp.add_constraint(not_both_1, variables)

    bqm = dwavebinarycsp.stitch(csp)
    simAnnSamples = simAnnSampler.sample(bqm, num_reads=1000)
    # plot_enumerate(simAnnSamples, title='Simulated annealing in default parameters')
    # plot_energies(simAnnSamples, title='Simulated annealing in default parameters')

    # Check that a good solution was found
    sample = simAnnSamples.first.sample  # doctest: +SKIP
    if not csp.check(sample):  # doctest: +SKIP
        print("Failed to color map. Try sampling again.")
    else:
        print(sample)
    plot_map(sample)
    plt.show()
