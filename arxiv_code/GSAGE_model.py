import torch.nn as nn
from layers import GraphConvolution
import torch.nn.functional as F
import torch
import math
from torch_geometric.nn import SAGEConv
from torch_sparse import SparseTensor



class GSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GSAGE, self).__init__()
        self.gc1 = SAGEConv(nfeat, nclass)


        self.dropout = dropout

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj_ori,domain='t'):

        if torch.is_tensor(adj_ori):
            adj_coo = adj_ori.coalesce()
        else:
            adj_coo = SparseTensor.from_dense(adj_ori)

        adj = SparseTensor(row=adj_coo.indices()[0],
                                col=adj_coo.indices()[1],
                                value=adj_coo.values(),
                                sparse_sizes=(adj_coo.size(0), adj_coo.size(1)))

        x = self.gc1(x, adj)

        x = F.relu(x)

        x = F.dropout(x, self.dropout, training=self.training)

        return None,x,None

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs

       

