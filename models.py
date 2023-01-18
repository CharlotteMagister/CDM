import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv, Sequential
import torch_explain as te
from torch_explain.logic.nn import entropy
from torch_explain.logic.metrics import test_explanation, complexity

import numpy as np
import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from scipy.spatial.distance import cdist
from sympy import to_dnf, lambdify
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import model_utils
import math

# class ConceptEmbeddings(torch.nn.Linear):
#     def __init__(self, in_features: int, out_features: int, emb_size: int, bias: bool = True,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(ConceptEmbeddings, self).__init__(in_features, out_features, bias, device, dtype)
#         self.weight = torch.nn.Parameter(torch.empty((out_features, in_features, emb_size), **factory_kwargs))
#         if bias:
#             self.bias = torch.nn.Parameter(torch.empty(out_features, emb_size, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         h = (input @ self.weight).permute(1, 0, 2) + self.bias
#         return h.permute(0, 2, 1)
#
#
# class EntropyLinear(nn.Module):
#     """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
#     """
#
#     def __init__(self, in_features: int, out_features: int, n_classes: int, temperature: float = 0.6,
#                  bias: bool = True, conceptizator: str = 'identity_bool') -> None:
#         super(EntropyLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.n_classes = n_classes
#         self.temperature = temperature
#         self.alpha = None
#         self.weight = nn.Parameter(torch.Tensor(n_classes, out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(n_classes, 1, out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if len(x.shape) > 2:
#             x = x.unsqueeze(0)
#         # compute concept-awareness scores
#         gamma = self.weight.norm(dim=1, p=1)
#         self.alpha = torch.exp(gamma/self.temperature) / torch.sum(torch.exp(gamma/self.temperature), dim=1, keepdim=True)
#
#         # weight the input concepts by awareness scores
#         self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
#         self.concept_mask = self.alpha_norm > 0.5
# #         print(x.shape, self.alpha_norm.unsqueeze(1).shape)
#         x = x.multiply(self.alpha_norm.unsqueeze(1).unsqueeze(1))
# #         print(x.shape)
#
#         # compute linear map
# #         print(x.shape, self.weight.permute(0, 2, 1).unsqueeze(1).shape)
#         x = x.matmul(self.weight.permute(0, 2, 1).unsqueeze(1))# + self.bias
# #         print(x.shape)
#         return x.permute(1, 0, 2, 3).squeeze(-1)
#
#     def extra_repr(self) -> str:
#         return 'in_features={}, out_features={}, n_classes={}'.format(
#             self.in_features, self.out_features, self.n_classes
#         )


# model definition
# model definition
class BA_Shapes_GCN(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, concept_embedding_size, num_classes):
        super(GCN, self).__init__()

        self.conv0 = GCNConv(num_in_features, num_hidden_features)
        self.conv1 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv2 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv3 = GCNConv(num_hidden_features, concept_encoding_size)

        # linear layers
        self.linear = nn.Linear(concept_encoding_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        self.gnn_embedding = x

        x = F.softmax(x, dim=-1)
        x = torch.div(x, torch.max(x, dim=-1)[0].unsqueeze(1))
        concepts = x

        x = self.linear(x)

        return concepts, x.squeeze(-1)

class Tree_Cycle_GCN(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes, n_clustering=3, concept_emb_size=4):
        super(Tree_Cycle_GCN, self).__init__()

        self.conv0 = DenseGCNConv(num_in_features, num_hidden_features)
        self.conv1 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv2 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv3 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv4 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv5 = DenseGCNConv(num_hidden_features, num_hidden_features)

        self.ce_layer = ConceptEmbeddings(num_hidden_features, n_clustering, concept_emb_size)
        self.lens = torch.nn.Sequential(EntropyLinear(n_clustering, 1, n_classes=num_classes),
                                        nn.LeakyReLU(),
                                        torch.nn.Linear(concept_emb_size, 1))

    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)

        x = x.squeeze()

        self.gnn_embedding = x

        x = self.ce_layer(x)

        self.concept_embedding = x

        x = F.softmax(x, dim=1)
        x = torch.div(x, torch.max(x, dim=1)[0].unsqueeze(1))
        concepts = x

        x = self.lens(x)

        return concepts, x.squeeze(-1)


class Tree_Grid_GCN(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes, n_clustering=3, concept_emb_size=4):
        super(Tree_Grid_GCN, self).__init__()

        self.conv0 = DenseGCNConv(num_in_features, num_hidden_features)
        self.conv1 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv2 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv3 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv4 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv5 = DenseGCNConv(num_hidden_features, num_hidden_features)
        self.conv6 = DenseGCNConv(num_hidden_features, num_hidden_features)

        self.ce_layer = ConceptEmbeddings(num_hidden_features, n_clustering, concept_emb_size)
        self.lens = torch.nn.Sequential(EntropyLinear(n_clustering, 1, n_classes=num_classes),
                                        nn.LeakyReLU(),
                                        torch.nn.Linear(concept_emb_size, 1))

    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv6(x, edge_index)
        x = F.leaky_relu(x)

        x = x.squeeze()

        self.gnn_embedding = x

        x = self.ce_layer(x)

        self.concept_embedding = x

        x = F.softmax(x, dim=1)
        x = torch.div(x, torch.max(x, dim=1)[0].unsqueeze(1))
        concepts = x

        x = self.lens(x)

        return concepts, x.squeeze(-1)
