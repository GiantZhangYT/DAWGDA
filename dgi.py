"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN_model import GraphConvolution



def add_gaussian_noise(tensor, mean=0, std=0.0000005):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise.cuda()
    return noisy_tensor

class Encoder(nn.Module):
    def __init__(self,in_feats, n_hidden, nclass, activation, dropout):
        super(Encoder, self).__init__()

        # self.conv = GCN(
        #     g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout
        # )
        self.fc1 = GraphConvolution(in_feats, n_hidden)
        self.fc2 = GraphConvolution(n_hidden, nclass)
        self.activation = activation
        self.dropout = dropout

    def forward(self, features,adj ,corrupt=False,domain='t'):
        if corrupt:
            perm = torch.randperm(features.shape[0])
            # perm = torch.randperm(self.g.num_nodes())
            features = features[perm]
        # features = self.conv(features)
        features = F.dropout(features, self.dropout, training=self.training)
        features = self.fc1(features,adj)

        features = self.activation(features)
        # if domain=='s':
        #     features = add_gaussian_noise(features)
        features = self.fc2(features,adj)
        # if domain=='s':
        #     features = add_gaussian_noise(features)

        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self,in_feats, n_hidden, nclass, activation, dropout):
        super(DGI, self).__init__()

        self.encoder = Encoder(
            in_feats, n_hidden, nclass, activation, dropout
        )
        self.discriminator = Discriminator(nclass)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features,adj,domain='t'):
        positive_ori = self.encoder(features,adj, corrupt=False,domain='t')
        negative_ori = self.encoder(features,adj, corrupt=True,domain='t')
        summary = torch.sigmoid(positive_ori.mean(dim=0))

        positive = self.discriminator(positive_ori, summary)
        negative = self.discriminator(negative_ori, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return positive_ori,l1 + l2


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)