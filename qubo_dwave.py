import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from ising import plot_enumerate, plot_energies

import dwave_networkx as dnx
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import neal


def simumated_annealing(model, num_reads=1000):
    simAnnSampler = neal.SimulatedAnnealingSampler()
    simAnnSamples = simAnnSampler.sample(model, num_reads=num_reads)

    plot_enumerate(simAnnSamples, title='Simulated annealing in default parameters')
    plot_energies(simAnnSamples, title='Simulated annealing in default parameters')

    energies = [datum.energy for datum in simAnnSamples.data(['energy'], sorted_by=None)]
    energy = energies[0]
    print(f"Minimum energy using simulated annealing:\t {energy}")
    plt.show()


def d_wave_annealing(model, num_reads=100, annealing_time=20, show_embeding=False):
    qpu = DWaveSampler()
    qpu_edges = qpu.edgelist
    qpu_nodes = qpu.nodelist
    print(qpu.solver.id)
    if qpu.solver.id == "DW_2000Q_6":
        X = dnx.chimera_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    #     dnx.draw_chimera(X, node_size=1)
    elif qpu.solver.id[:9] == "Advantage":
        X = dnx.pegasus_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    #     dnx.draw_pegasus(X, node_size=1)

    print('Number of qubits =', len(qpu_nodes))
    print('Number of couplers =', len(qpu_edges))

    DWavesampler = EmbeddingComposite(DWaveSampler())
    DWaveSamples = DWavesampler.sample(bqm=model, num_reads=num_reads, return_embedding=True,
                                       annealing_time=annealing_time
                                       )
    print(DWaveSamples.info)

    if show_embeding:
        embedding = DWaveSamples.info['embedding_context']['embedding']
        if qpu.solver.id == "DW_2000Q_6":
            dnx.draw_chimera_embedding(X, embedding, node_size=2)
        elif qpu.solver.id[:9] == "Advantage":
            dnx.draw_pegasus_embedding(X, embedding, node_size=2)
        plt.show()

    plot_enumerate(DWaveSamples, title='Quantum annealing in default parameters')
    plot_energies(DWaveSamples, title='Quantum annealing in default parameters')

    energies = [datum.energy for datum in DWaveSamples.data(['energy'], sorted_by=None)]
    energy = energies[0]
    print(f"Minimum energy using D-Wave:\t {energy}")


if __name__ == "__main__":
    # This is the problem min c^T x s.t. Ax=b
    A = np.array([[1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]])
    b = np.array([1, 1, 1])
    c = np.array([2, 4, 4, 4, 4, 4, 5, 4, 5, 6, 5])

    # Exploiting x^2 = x for x in {0,1} we restate the problem, such that the linear terms appear on the diagonal of Q
    # min c^T x + rho(Ax-b)^T(Ax-b) = min c^Tx + rho(x^T (A^TA)x - 2b^T Ax+ b^TB)
    # for this problem a good penalty factor is given by rho > sum |c|
    rho = np.sum(np.abs(c)) + 1
    Q = rho*np.matmul(A.T, A)
    Q += np.diag(c)
    Q -= rho*2*np.diag(np.matmul(b.T, A))
    cQ = rho*np.matmul(b.T, b)
    print(Q)
    print(cQ)

    # G = nx.from_numpy_matrix(Q)
    # nx.draw(G, with_labels=True)
    # plt.show()

    mdl = dimod.BinaryQuadraticModel.from_qubo(Q, offset=cQ)
    d_wave_annealing(mdl)
    plt.show()
