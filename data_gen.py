import numpy as np
from scipy.sparse import coo_matrix
from pgmpy.readwrite import BIFReader
import networkx as nx
import cdt
import pandas as pd


def gen_graph(graph_type, n, mean_deg):
    """Generates and returns a nx.Digraph and its adjacency matrix. Nodes are randomly permutated.

    Arguments:
    - graph_type (string): type of graph Erdos-Renyi, scale-free, sachs or any graph in BNRepo
    - n (int): number of nodes
    - mean_deg (float): average degree of nodes
    """
    # beta is the unpermutated adjacency matrix
    if graph_type == "erdos-renyi":
        beta = gen_random_graph(n, mean_deg)
    elif graph_type == "scale-free":
        # select
        import igraph as ig
        G_ig = ig.Graph.Barabasi(n=n, m=int(round(mean_deg/2)), directed=True)
        beta = np.array(G_ig.get_adjacency().data)
    elif graph_type == "sachs":
        G = nx.DiGraph()
        # Hand-made
        G.add_edge("Pclg", "PIP2")
        G.add_edge("PIP3", "PIP2")
        G.add_edge("Pclg", "PIP2")
        G.add_edge("PKC", "P38")
        G.add_edge("PKC", "Jnk")
        G.add_edge("PKC", "PKA")
        G.add_edge("PKC", "Raf")
        G.add_edge("PKC", "Mek")
        G.add_edge("PKA", "P38")
        G.add_edge("PKA", "Jnk")
        G.add_edge("PKA", "Ark")
        G.add_edge("PKA", "Erk")
        G.add_edge("PKA", "Mek")
        G.add_edge("PKA", "Raf")
        G.add_edge("Raf", "Mek")
        G.add_edge("Mek", "Erk")
        G.add_edge("Erk", "Ark")

        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        beta = nx.adjacency_matrix(G, nodelist=sorted(G.nodes))
        assert np.trace(np.linalg.matrix_power(np.eye(n) + beta.toarray(), n)) == n  # is_dag?
        return G, beta.toarray()
    else:
        _, beta = loadBNrepo(graph_type)

    # Randomly permute nodes
    perm_mat = np.random.permutation(np.eye(n))
    adj_matrix = perm_mat.T @ beta @ perm_mat

    # Sanity check, is the graph acyclic?
    assert np.trace(np.linalg.matrix_power(np.eye(n) + adj_matrix, n)) == n

    # Create and return directed graph
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return graph, adj_matrix


def gen_random_graph(n, mean_deg):
    """Returns the adjacency matrix of an Erdos Renyi DAG

    Args:
    - n (int): number of nodes
    - mean_deg (float): average degree of a node
    """
    assert mean_deg <= n-1
    prob_one_edge = mean_deg/(n-1)
    beta = np.triu(np.random.random((n, n)) < prob_one_edge, k=1)
    return np.float32(beta)


def simulate_parameter(adj_matrix, w_ranges):
    """Simulate SEM parameters for a DAG.
    Args:
        adj_matrix (np.array): [n, n] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges
    Returns:
        weighted_adj_matrix (np.array): [n, n] weighted adj matrix of DAG
    """
    weighted_adj_matrix = np.zeros(adj_matrix.shape)
    range_choice = np.random.randint(len(w_ranges), size=adj_matrix.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        weights = np.random.uniform(low=low, high=high, size=adj_matrix.shape)
        weighted_adj_matrix += adj_matrix * (range_choice == i) * weights
    return weighted_adj_matrix


def sample_lin_scms(graph_type, noise_type, adj_matrix, nb_samples=1000,
                    weighted=False,
                    w_ranges=((-2.0, -.5), (.5, 2.0))):
    """ Given a directed graph and a particular noise type, generates edge weights and samples

    Args:
        graph_type (string): type of graph
        noise_type (string): one of gaussian, exp, gumbel, type of random noise
        adj_matrix (np.array): [n, n] binary adjacency matrix
        nb_samples (int): number of samples to generate
        weighted (bool): whether to use uniformly weighted edges or all edges are
        w_ranges (tuple): negative and positive ranges to sample edge weights (if weighted)

    Returns:
        X (np.array): [nb_samples, n] sample matrix
        beta (np.array): [n, n] weighted adjacency matrix
        sigma_n (np.array): [n, n] sample covariance matrix
    """
    n = adj_matrix.shape[0]
    if weighted and graph_type != "sachs":
        beta = simulate_parameter(adj_matrix, w_ranges)
    else:
        beta = adj_matrix
    aux_inv = np.linalg.inv(np.eye(n) - beta)

    if graph_type == "sachs":
        # need to sort features - alphabetically is random
        df = pd.read_csv("Data/sachs.data.txt", sep="\t")
        X = np.array(df[sorted(df.columns)])
    else:
        if noise_type == "gaussian":
            epsilon = np.random.normal(size=(nb_samples, n))
        elif noise_type == "exp":
            epsilon = np.random.exponential(size=(nb_samples, n))
        elif noise_type == "gumbel":
            epsilon = np.random.gumbel(size=(nb_samples, n))
        else:
            raise NotImplementedError
        X = epsilon @ aux_inv
    sigma_n = np.cov(X.T, bias=True)
    return X, beta, sigma_n


def loadBNrepo(dataset):
    """ Reads BIF file from BNRepo and returns adjacency matrix

    Args:
        dataset: name of file
    """
    try:
        reader = BIFReader(dataset + ".bif")
    except FileNotFoundError:
        return
    names_nodes = np.unique(np.array(reader.get_edges()))
    np.random.shuffle(names_nodes)
    names_nodes = list(names_nodes)
    n = len(names_nodes)
    p = len(reader.get_edges())

    rows = [names_nodes.index(e[0]) for e in reader.get_edges()]
    cols = [names_nodes.index(e[1]) for e in reader.get_edges()]
    vals = [1] * p  # Linear models
    beta = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.int8)

    return beta.todense()
