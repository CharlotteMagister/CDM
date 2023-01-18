import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import networkx as nx
import sys

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader

def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)

    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices, supplement


def load_syn_data2(dataset_str, path_prefix="../"):
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

    return data



def read_syn_data(folder: str, prefix):
    # with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
    #     adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

    # load data
    G, labels = load_syn_data2("Tree_Cycle")
    data = prepare_syn_data(G, labels, 0.8)

    train_mask = data['train_mask']
    test_mask = data['test_mask']
    val_mask = data['test_mask']
    adj = data['edges']

    y = data["y"].detach().numpy()
    y_train = y[train_mask]
    y_test = y[test_mask]
    y_val = y_test

    x = data["x"].float()
    # print(train_mask.reshape(-1, 1))
    # y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    # y = torch.from_numpy(np.where(y)[1])
    y = data["y"]
    edge_index = adj
    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)

    # x = torch.from_numpy(features).float()
    # y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    # y = torch.from_numpy(np.where(y)[1])
    # edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
    # data = Data(x=x, y=y, edge_index=edge_index)
    # data.train_mask = torch.from_numpy(train_mask)
    # data.val_mask = torch.from_numpy(val_mask)
    # data.test_mask = torch.from_numpy(test_mask)

    return data


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        data_list.append(Data(x=torch.from_numpy(node_features[graph_idx]).float(),
                              edge_index=dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0],
                              y=torch.from_numpy(np.where(graph_labels[graph_idx])[0])))
    return data_list


def get_dataset(dataset_dir, dataset_name, task=None):
    sync_dataset_dict = {
        'BA_2Motifs'.lower(): 'BA_2Motifs',
        'BA_Shapes'.lower(): 'BA_shapes',
        'BA_Community'.lower(): 'BA_Community',
        'Tree_Cycle'.lower(): 'Tree_Cycle',
        'Tree_Grids'.lower(): 'Tree_Grids',
    }
    sentigraph_names = ['Graph_SST2', 'Graph_Twitter', 'Graph_SST5']
    sentigraph_names = [name.lower() for name in sentigraph_names]
    molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]

    if dataset_name == 'BA_Shapes':
        return load_syn_data(dataset_dir, dataset_name)
    elif dataset_name.lower() == 'MUTAG'.lower():
        return load_MUTAG(dataset_dir, 'MUTAG')
    else:
        print("FAIL")
        sys.exit(0)

    # if dataset_name.lower() == 'MUTAG'.lower():
    #     return load_MUTAG(dataset_dir, 'MUTAG')
    # elif dataset_name.lower() in sync_dataset_dict.keys():
    #     sync_dataset_filename = sync_dataset_dict[dataset_name.lower()]
    #     return load_syn_data(dataset_dir, sync_dataset_filename)
    # elif dataset_name.lower() in molecule_net_dataset_names:
    #     return load_MolecueNet(dataset_dir, dataset_name, task)
    # elif dataset_name.lower() in sentigraph_names:
    #     return load_SeniGraph(dataset_dir, dataset_name)
    # else:
    #     raise NotImplementedError


class MUTAGDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.upper()
        super(MUTAGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['MUTAG_A', 'MUTAG_graph_labels', 'MUTAG_graph_indicator', 'MUTAG_node_labels']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
            nodes_all_temp = f.read().splitlines()
            nodes_all = [int(i) for i in nodes_all_temp]

        adj_all = np.zeros((len(nodes_all), len(nodes_all)))
        with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
            adj_list = f.read().splitlines()
        for item in adj_list:
            lr = item.split(', ')
            l = int(lr[0])
            r = int(lr[1])
            adj_all[l - 1, r - 1] = 1

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
            graph_indicator_temp = f.read().splitlines()
            graph_indicator = [int(i) for i in graph_indicator_temp]
            graph_indicator = np.array(graph_indicator)

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
            graph_labels_temp = f.read().splitlines()
            graph_labels = [int(i) for i in graph_labels_temp]

        data_list = []
        for i in range(1, 189):
            idx = np.where(graph_indicator == i)
            graph_len = len(idx[0])
            adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
            label = int(graph_labels[i - 1] == 1)
            feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
            nb_clss = 7
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                y=label)
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
              = read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


class SynGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("loaded the other!!!")

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data = read_syn_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Loaded BA-SHAPES MINE")


class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ba2motif_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save(self.collate(data_list), self.processed_paths[0])






def prepare_real_data(graphs, train_split, batch_size, dataset_str):
    graphs = graphs.shuffle()

    train_idx = int(len(graphs) * train_split)
    train_set = graphs[:train_idx]
    test_set = graphs[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

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


    return train_loader, val_loader, test_loader





def load_MUTAG(dataset_dir, dataset_name):
    """ 188 molecules where label = 1 denotes mutagenic effect """
    graphs = TUDataset(root='../data/', name='Mutagenicity')
    # dataset = MUTAGDataset(root=dataset_dir, name=dataset_name)

    input_dim = graphs.num_node_features
    output_dim = graphs.num_classes
    train_loader, val_loader, test_loader = prepare_real_data(graphs, 0.8, 16, "Mutagenicity")

    return graphs, input_dim, output_dim, train_loader, val_loader, test_loader


def load_syn_data(dataset_dir, dataset_name):
    """ The synthetic dataset """
    # if dataset_name.lower() == 'BA_2Motifs'.lower():
    #     dataset = BA2MotifDataset(root=dataset_dir, name=dataset_name)
    # else:
    dataset = SynGraphDataset(root=dataset_dir, name=dataset_name)
    dataset.node_type_dict = {k: v for k, v in enumerate(range(dataset.num_classes))}
    dataset.node_color = None
    return dataset


def load_MolecueNet(dataset_dir, dataset_name, task=None):
    """ Attention the multi-task problems not solved yet """
    molecule_net_dataset_names = {name.lower(): name for name in MoleculeNet.names.keys()}
    dataset = MoleculeNet(root=dataset_dir, name=molecule_net_dataset_names[dataset_name.lower()])
    dataset.data.x = dataset.data.x.float()
    if task is None:
        dataset.data.y = dataset.data.y.squeeze().long()
    else:
        dataset.data.y = dataset.data.y[:, 0].long()
    dataset.node_type_dict = None
    dataset.node_color = None
    return dataset


def load_SeniGraph(dataset_dir, dataset_name):
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name)
    return dataset


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=5):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    # if not random_split_flag and hasattr(dataset, 'supplement'):
    #     assert 'split_indices' in dataset.supplement.keys(), "split idx"
    #     split_indices = dataset.supplement['split_indices']
    #     train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
    #     dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
    #     test_indices = torch.where(split_indices == 2)[0].numpy().tolist()
    #
    #     train = Subset(dataset, train_indices)
    #     eval = Subset(dataset, dev_indices)
    #     test = Subset(dataset, test_indices)
    # else:
    # num_train = int(data_split_ratio[0] * len(dataset))
    # num_test = len(dataset) - num_train - num_eval
    #
    # train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
    #                                      generator=torch.Generator().manual_seed(seed))
    #
    # print(len(dataset))

    dataloader = dict()
    dataloader['train'] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader
