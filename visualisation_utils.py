import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn as sns
from matplotlib import rc
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import clustering_utils
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
import torch


def set_rc_params():
    small = 14
    medium = 20
    large = 28

    plt.rc('figure', autolayout=True, figsize=(12, 8))
    plt.rc('font', size=medium)
    plt.rc('axes', titlesize=medium, labelsize=small, grid=True)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=large, facecolor='white')
    plt.rc('legend', loc='upper left')


def plot_model_accuracy(train_accuracies, test_accuracies, model_name, path):
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Testing Accuracy")
    plt.title(f"Accuracy of {model_name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_accuracy_plot.png"))
    plt.show()


def plot_model_loss(train_losses, test_losses, model_name, path):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.title(f"Loss of {model_name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_loss_plot.png"))
    plt.show()


# should be called with concepts
def plot_concept_heatmap(centroids, concepts, y, used_centroid_labels, model_name, layer_num, path, id_title="", id_path=""):
    LATEX_SYMBOL = ""  # Change to "$" if working out of server
    rc('text', usetex=(LATEX_SYMBOL == "$"))
    plt.style.use('seaborn-whitegrid')

    plt.figure(figsize=[15, 5])
    fig, ax = plt.subplots(len(np.unique(used_centroid_labels)), 2, gridspec_kw={'width_ratios': [5, 1]})
    # fig.set_size_inches(12, len(centroids) * 0.8)
    fig.suptitle(f"{id_title}Concept Heatmap of the {model_name} with concepts extracted from Layer {layer_num}")

    if torch.is_tensor(concepts):
        concepts = concepts.detach().numpy()
    else:
        concepts = concepts[:, np.newaxis]

    used_centroid_labels = used_centroid_labels.squeeze()

    nclasses = len(np.unique(y))
    if len(centroids) == 1:
        sns.heatmap(concepts[used_centroid_labels==0] > 0.5, cbar=None, ax=ax[0])
        sns.heatmap(y[used_centroid_labels==0].unsqueeze(-1), vmin=0, vmax=4, cmap="Set2", ax=ax[1])
    else:
        for i in range(len(np.unique(used_centroid_labels))):
            sns.heatmap(concepts[used_centroid_labels==i] > 0.5, cbar=None, ax=ax[i, 0],
                        xticklabels=False, yticklabels=False)
            sns.heatmap(y[used_centroid_labels==i].unsqueeze(-1), vmin=0, vmax=nclasses, cmap="Set2", ax=ax[i, 1],
                        xticklabels=False, yticklabels=False, cbar=None)

    legend_elements = [Patch(facecolor=c, label=f'Class {i // 2}') for i, c in
                       enumerate(sns.color_palette("Set2", 2 * nclasses)) if i % 2 == 0]
    fig.legend(handles=legend_elements, fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(legend_elements))
    plt.savefig(os.path.join(path, f"{id_path}concept_heatmap_layer{layer_num}.png"))


def plot_clustering(seed, activation, y, centroids, centroid_labels, used_centroid_labels, model_name, layer_num, path, id_title="Node ", id_path="node"):
    all_data = np.vstack([activation, centroids])

    tsne_model = TSNE(n_components=2, random_state=seed)
    all_data2d = tsne_model.fit_transform(all_data)

    d = all_data2d[:len(activation)]
    centroids2d = all_data2d[len(activation):]

    fig = plt.figure(figsize=[15, 5])
    fig.suptitle(f"{id_title}Clustering of Activations in Layer {layer_num} of {model_name}")

    ax = plt.subplot(1, 3, 1)
    ax.set_title("Real Labels")
    p = sns.color_palette("husl", len(np.unique(y)))
    sns.scatterplot(d[:, 0], d[:, 1], hue=y, palette=p)

    ax = plt.subplot(1, 3, 2)
    ax.set_title("Model's Clusters")
    p = sns.color_palette("husl", len(np.unique(used_centroid_labels)))
    sns.scatterplot(d[:, 0], d[:, 1], hue=used_centroid_labels, palette=p, legend=None)

    ax = plt.subplot(1, 3, 3)
    ax.set_title("Model's Cluster Centroids")
    p = sns.color_palette("husl", len(np.unique(used_centroid_labels)))
    sns.scatterplot(d[:, 0], d[:, 1], hue=used_centroid_labels, palette=p, legend=None, alpha=0.3)

    p = sns.color_palette("husl", len(centroids))
    sns.scatterplot(centroids2d[:, 0], centroids2d[:, 1], hue=list(range(len(centroids))), palette=p, alpha=0.7,
                    legend=None, **{'s': 600, 'marker': '*', 'edgecolors': None})

    plt.savefig(os.path.join(path, f"DifferentialClustering_Raw_Layer{layer_num}{id_path}.png"))
    plt.show()

def print_cluster_counts(used_centroid_labels):
    cluster_counts = np.unique(used_centroid_labels, return_counts=True)

    cluster_ids, counts = zip(*sorted(zip(cluster_counts[0], cluster_counts[1])))
    print("Cluster sizes by cluster id:")
    for id, c in zip(cluster_ids, counts):
        print(f"\tCluster {id}: {c}")

    return cluster_counts


def plot_activation_space(data, labels, activation_type, layer_num, path, note="", dr_type="", naming_help=""):
    rows = len(data)
    fig, ax = plt.subplots()
    fig.suptitle(f"{activation_type} Activations of Layer {layer_num} {note}")

    scatter = ax.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')
    ax.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(labels)), bbox_to_anchor=(1.05, 1))

    plt.savefig(os.path.join(path, f"{dr_type}_{layer_num}_layer{naming_help}.png"))
    plt.show()


def plot_clusters(data, labels, clustering_type, k, layer_num, path, data_type, reduction_type="", note="", cool=False):
    fig, ax = plt.subplots()
    fig.suptitle(f'{clustering_type} Clustered {data_type} Activations of Layer {layer_num} {note}')

    for i in range(k):
        if cool:
            scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster this', marker='x', s=100)
        else:
            scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}')

    ncol = 1
    if k > 20:
        ncol = int(k / 20) + 1
    ax.legend(bbox_to_anchor=(1.05, 1), ncol=ncol)
    plt.savefig(os.path.join(path, f"{clustering_type.replace(' ', '')}_{layer_num}layer_{data_type}{reduction_type}.png"))
    plt.show()


def plot_samples(clustering_model, data, y, layer_num, k, clustering_type, reduction_type, num_nodes_view, edges, num_expansions, path, concepts=None, concepts2=None, graph_data=None, graph_name=None, if_save=True):
    res_sorted = clustering_utils.get_node_distances(clustering_model, data, concepts=concepts, concepts2=concepts2)

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    if k > 200:
        k = 200

    fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
    fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num}', y=1.005)

    if graph_data is not None:
        fig2, axes2 = plt.subplots(k, col, figsize=(18, 3 * k + 2))
        fig2.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num} (by node index)', y=1.005)

    l = list(range(0, k))
    sample_graphs = []
    sample_feat = []

    for i, ax_list in zip(l, axes):
        if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
            distances = res_sorted[i]
        elif isinstance(clustering_model, KMeans):
            distances = res_sorted[:, i]
        elif clustering_model is None:
            distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            tg, cm, labels, node_labels = clustering_utils.get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data, graph_name)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if k == 1:
            ax_list = [ax_list]

        if graph_data is None:
            for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)
        else:
            for ax, new_G, color_map, g_label, n_labels in zip(ax_list, top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax, labels=n_labels)
                ax.set_title(f"label {g_label}", fontsize=14)

            for ax, new_G, color_map, g_label, n_labels in zip(axes2[i], top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    if if_save:
        views = ''.join((str(i) + "_") for i in num_nodes_view)
        if isinstance(clustering_model, AgglomerativeClustering):
            fig.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
        else:
            fig.savefig(os.path.join(path, f"{clustering_type}_{reduction_type}_{layer_num}layer_{views}view_samples.png"))

        if graph_data is not None:
            if isinstance(clustering_model, AgglomerativeClustering):
                fig2.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view_by_node.png"))
            else:
                fig2.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view_by_node.png"))

    plt.show()

    return sample_graphs, sample_feat
