import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import torch_geometric
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import cdist
import seaborn as sns

def find_centroids(activation, concepts, y, tau=0.5):
    concepts = concepts.detach().numpy()
    centroids = []
    used_centroid_labels = np.zeros_like(y) - 1
    centroid_labels = []

    # gets boolean encoding of concepts
    cluster_general_labels = np.unique(concepts>tau, axis=0)

    for concept in range(len(cluster_general_labels)):
        # get all concept rows where have matching boolean encoding
        cluster_samples = np.where(((concepts>tau)==(cluster_general_labels[concept])).all(axis=1))[0]

        # take mean of those activations fitting the concept
        centroid = np.mean(activation[cluster_samples], axis=0)

        # sample - concept mapping
        used_centroid_labels[cluster_samples] = concept
        centroid_labels.append(concept)
        centroids.append(centroid)

    centroids = np.vstack(centroids)
    centroid_labels = np.stack(centroid_labels)

    return centroids, centroid_labels, used_centroid_labels

def get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data=None, graph_name=None):
    graphs = []
    color_maps = []
    labels = []
    node_labels = []

    df = pd.DataFrame(edges)

    for idx in top_indices:
        # get neighbours
        neighbours = list()
        neighbours.append(idx)

        for i in range(0, num_expansions):
            new_neighbours = list()
            for e in edges:
                if (e[0] in neighbours) or (e[1] in neighbours):
                    new_neighbours.append(e[0])
                    new_neighbours.append(e[1])

            neighbours = neighbours + new_neighbours
            neighbours = list(set(neighbours))

        new_G = nx.Graph()
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        remaining_edges = df_neighbours.to_numpy()
        new_G.add_edges_from(remaining_edges)

        color_map = []
        node_label = {}
        color_pal = sns.color_palette("colorblind", 2).as_hex()
        if graph_data is None:
            for node in new_G:
                if node in top_indices:
                    color_map.append(color_pal[0])
                else:
                    color_map.append(color_pal[1])
        else:
            if graph_name == "Mutagenicity":
                ids = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
            elif graph_name == "REDDIT-BINARY":
                ids = []

            for node in zip(new_G):
                node = node[0]
                color_idx = graph_data[node]
                color_map.append(color_idx)
                node_label[node] = f"{ids[color_idx]}"

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])
        node_labels.append(node_label)


    return graphs, color_maps, labels, node_labels


def get_node_distances(clustering_model, data, concepts=None, concepts2=None):
    if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
        x, y_predict = data
        clf = NearestCentroid()
        clf.fit(x, y_predict)
        centroids = clf.centroids_
        res = pairwise_distances(centroids, x)
        res_sorted = np.argsort(res, axis=-1)
    elif isinstance(clustering_model, KMeans):
        res_sorted = clustering_model.transform(data)
    elif clustering_model is None and concepts is not None:
        res_sorted = cdist(data, concepts)

    return res_sorted


def find_cluster_centroids(data, concepts, K, tau=0.5):
    activation = data.detach().numpy()
    concepts = concepts.detach().numpy()

    centroids = []
    for c in range(K):
        cluster_samples = np.where(((concepts > tau) == c).all(axis=1))[0]

        if len(cluster_samples) == 0:
            cluster_samples = np.argsort(concepts[:, c])[:10]

        centroid = np.average(activation[cluster_samples], axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    return torch.transpose(torch.from_numpy(centroids), 0, 1)
