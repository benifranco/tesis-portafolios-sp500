# scripts/graph_clustering.py

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering

from inputs.experiment_config import EXPERIMENT_PARAMS


def graph_from_adj(adj):
    """
    Build an undirected NetworkX graph from a binary adjacency matrix.

    Parameters
    ----------
    adj : DataFrame
        Binary adjacency matrix; index and columns are node labels.

    Returns
    -------
    G : networkx.Graph
    """
    G = nx.Graph()
    nodes = list(adj.columns)
    G.add_nodes_from(nodes)

    mat = adj.values
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j]:
                G.add_edge(nodes[i], nodes[j])

    return G


def spectral_communities(G, k, epsilon=None, eigen_tol=None):
    """
    Run spectral clustering on a graph using a precomputed affinity matrix.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    k : int
        Number of clusters (communities).
    epsilon : float or None
        Small value used to replace zeros in the affinity matrix.
        If None, it is taken from EXPERIMENT_PARAMS["epsilon_graph"].
    eigen_tol : float or None
        Tolerance passed to SpectralClustering (eigen_tol).
        If None, it is taken from EXPERIMENT_PARAMS["eigen_tol"].

    Returns
    -------
    communities : dict
        Mapping cluster_label -> list of node names.
    """
    if epsilon is None:
        epsilon = EXPERIMENT_PARAMS.get("epsilon_graph", 1e-9)
    if eigen_tol is None:
        eigen_tol = EXPERIMENT_PARAMS.get("eigen_tol", 1e-4)

    random_state = EXPERIMENT_PARAMS.get("spectral_random_state", 42)

    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)

    # Replace zeros with a small epsilon to avoid completely disconnected entries
    mask = A == 0
    np.fill_diagonal(mask, False)
    A[mask] = epsilon

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        eigen_tol=eigen_tol,
    )
    labels = sc.fit_predict(A)

    communities = {}
    for node, label in zip(nodes, labels):
        communities.setdefault(label, []).append(node)

    return communities
