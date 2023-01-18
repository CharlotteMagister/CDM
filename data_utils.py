import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader

def relabel_batch(batch):
    new_batch = []
    real_label = 0
    prev_label = None

    for b in batch:
        if prev_label is None:
            prev_label = b
        elif b != prev_label:
            real_label += 1
            prev_label = b

        new_batch.append(real_label)

    return torch.from_numpy(np.array(new_batch))

def reshape_graph_to_node_data(graph_data, batch):
    node_data = []

    i = 0
    for val in batch:
        if i == val:
            if torch.is_tensor(graph_data[i]):
                node_data.append(graph_data[i].detach().numpy())
            else:
                node_data.append(graph_data[i])
        else:
            i += 1
            if torch.is_tensor(graph_data[i]):
                node_data.append(graph_data[i].detach().numpy())
            else:
                node_data.append(graph_data[i])

    return torch.from_numpy(np.array(node_data))

def reshape_batch(batch, mask):
    new_batch = []

    counter = 0
    batch_idx = 0
    for i, val in enumerate(mask):
        if val:
            while batch[batch_idx] == i:
                new_batch.append(counter)

            counter += 1
            batch_idx += 1
        else:
            while batch[batch_idx] == i:
                batch_idx += 1

    return np.array(new_batch)

# class GraphFilter(object):
#     def __init__(self, max_nodes):
#         self.max_nodes = max_nodes
#     def __call__(self, data):
#         return data.num_nodes <= self.max_nodes


def load_real_data(dataset_str):
    print("Param ", dataset_str)
    if dataset_str == "Mutagenicity":
        graphs = TUDataset(root='../data/', name='Mutagenicity')

    elif dataset_str == "Reddit_Binary":
        # graphs = TUDataset(root='../data/', name='REDDIT-BINARY', transform=torch_geometric.transforms.ToDense(max_nodes), pre_filter=GraphFilter(max_nodes))
        graphs = TUDataset(root='../data/', name='REDDIT-BINARY', transform=torch_geometric.transforms.Constant())
    else:
        raise Exception("Invalid Real Dataset Name")

    print()
    print(f'Dataset: {graphs}:')
    print('====================')
    print(f'Number of graphs: {len(graphs)}')
    print(f'Number of features: {graphs.num_features}')
    print(f'Number of classes: {graphs.num_classes}')

    return graphs


def prepare_real_data(graphs, train_split, batch_size, dataset_str):
    graphs = graphs.shuffle()

    train_idx = int(len(graphs) * train_split)
    train_set = graphs[:train_idx]
    test_set = graphs[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    full_train_loader = DataLoader(train_set, batch_size=int(len(train_set) * 0.1), shuffle=True)
    full_test_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1), shuffle=True)

    if dataset_str == "Mutagenicity":
        full_loader = DataLoader(test_set, batch_size=int(len(test_set)), shuffle=True)
        small_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1))
    elif dataset_str == "Reddit_Binary":
        full_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1), shuffle=True)
        small_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.005))

    train_zeros = 0
    train_ones = 0
    for data in train_set:
        train_ones += np.sum(data.y.detach().numpy())
        train_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    test_zeros = 0
    test_ones = 0
    for data in test_set:
        test_ones += np.sum(data.y.detach().numpy())
        test_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    print()
    print(f"Class split - Training 0: {train_zeros} 1:{train_ones}, Test 0: {test_zeros} 1: {test_ones}")


    return train_set, train_loader, test_loader, full_train_loader, full_test_loader, full_loader, small_loader


def load_syn_data(dataset_str, path_prefix=""):
    if dataset_str == "BA_Shapes":
        G = nx.readwrite.read_gpickle(f"{path_prefix}../data/BA_Houses/graph_ba_300_80.gpickel")
        role_ids = np.load(f"{path_prefix}../data/BA_Houses/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Shapes_no_random_edges":
        G = nx.readwrite.read_gpickle(f"{path_prefix}../data/BA_Houses_no_random_edges/graph_ba_300_80.gpickel")
        role_ids = np.load(f"{path_prefix}../data/BA_Houses_no_random_edges/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Grid":
        G = nx.readwrite.read_gpickle(f"{path_prefix}../data/BA_Grid/graph_ba_300_80.gpickel")
        role_ids = np.load("../data/BA_Grid/role_ids_ba_300_80.npy")

    elif dataset_str == "Tree_Cycle":
        G = nx.readwrite.read_gpickle(f"{path_prefix}../data/Tree_Cycle/graph_tree_8_60.gpickel")
        role_ids = np.load(f"{path_prefix}../data/Tree_Cycle/role_ids_tree_8_60.npy")

    elif dataset_str == "Tree_Grid":
        G = nx.readwrite.read_gpickle(f"{path_prefix}../data/Tree_Grid/graph_tree_8_80.gpickel")
        role_ids = np.load(f"{path_prefix}../data/Tree_Grid/role_ids_tree_8_80.npy")

    elif dataset_str == "BA_Community":
        G = nx.readwrite.read_gpickle(f"{path_prefix}../data/BA_Community/graph_ba_350_100_2comm.gpickel")
        role_ids = np.load(f"{path_prefix}../data/BA_Community/role_ids_ba_350_100_2comm.npy")
    else:
        raise Exception("Invalid Syn Dataset Name")

    return G, role_ids


def prepare_syn_data(G, labels, train_split, path=None, if_adj=False):
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1)
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    train_mask = np.random.rand(len(features)) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))

    data = {"x": features, "y": labels, "edges": edges, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}

    if path is not None:
        persistence_utils.persist_experiment(data, path,'data.z')

    return data


def create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
