import networkx as nx
import numpy as np


def gen_graph(graph_type, n, mean_deg):
    """Generates and returns a nx.Digraph and its adjacency matrix. Nodes are randomly permutated.

    Arguments:
        graph_type (string): type of graph Erdos-Renyi, scale-free, sachs or any graph in BNRepo
        n (int): number of nodes
        mean_deg (float): average degree of nodes
    """
    # beta is the unpermutated adjacency matrix
    if graph_type == "erdos-renyi":
        beta = gen_random_graph(n, mean_deg)
    elif graph_type == "scale-free":
        # select
        import igraph as ig
        G_ig = ig.Graph.Barabasi(n=n, m=int(round(mean_deg / 2)), directed=True)
        beta = np.array(G_ig.get_adjacency().data)
    else:
        raise NotImplementedError

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
        n (int): number of nodes
        mean_deg (float): average degree of a node
    """
    assert mean_deg <= n - 1
    prob_one_edge = mean_deg / (n - 1)
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
    # Sample edge weights
    if weighted:
        beta = simulate_parameter(adj_matrix, w_ranges)
    else:
        beta = adj_matrix
    aux_inv = np.linalg.inv(np.eye(n) - beta)

    # Sample noise
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
