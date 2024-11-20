import numpy as np
import scipy.sparse as sp
import torch
import itertools
from scipy import sparse
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.decomposition import PCA
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt


def adj_matrix(graph):
    nodes = []
    for src, v in graph.items():
        nodes.extend([[src, v_] for v_ in v])
        nodes.extend([[v_, src] for v_ in v])
    nodes = [k for k, _ in itertools.groupby(sorted(nodes))]
    nodes = np.array(nodes)
    return sparse.coo_matrix((np.ones(nodes.shape[0]), (nodes[:, 0], nodes[:, 1])),
                             (len(graph), len(graph)))


def norm_x(x):
    return np.diag(np.power(x.sum(axis=1), -1).flatten()).dot(x)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_adj_matrix(matrix):
    matrix += sparse.eye(matrix.shape[0])
    degree = np.array(matrix.sum(axis=1))
    d_sqrt = sparse.diags(np.power(degree, -0.5).flatten())
    return d_sqrt.dot(matrix).dot(d_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def csr_2_sparse_tensor_tuple(csr_matrix):
    if not isinstance(csr_matrix, scipy.sparse.lil_matrix):
        csr_matrix = lil_matrix(csr_matrix)
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape

def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat.toarray()


def load_data_citation(path="data/",dataset="citationv1"):
    file = str(path) + str(dataset)
    net = sio.loadmat(file)
    features, adj, labels = net['attrb'], net['network'], net['group']
    labels = np.array(labels)

    features = torch.FloatTensor(features.astype(float))
    labels = np.argmax(labels, 1)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    return adj, features, labels

def load_data_blog(path="data/",dataset="citationv1"):
    file = str(path) + str(dataset)
    net = sio.loadmat(file)
    features, adj, labels = net['attrb'], net['network'], net['group']
    labels = np.array(labels)

    features = torch.FloatTensor(features.astype(float))
    features = torch.nn.functional.normalize(features, dim=1)
    labels = np.argmax(labels, 1)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    return adj, features, labels


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W

def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A

def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI

def visualization(encoded_source,encoded_target,file_name):
    plt.clf()
    pca = PCA(n_components=2)
    encoded_source_pca = pca.fit_transform(encoded_source.cpu().detach().numpy())
    encoded_target_pca = pca.fit_transform(encoded_target.cpu().detach().numpy())
    plt.scatter(encoded_source_pca[:,0],encoded_source_pca[:,1],c='blue',s=5)
    plt.scatter(encoded_target_pca[:,0],encoded_target_pca[:,1],c='red',s=5)
    plt.savefig(file_name,dpi=300)


def pair_enumeration(x1, x2):
    '''
        input:  [B,D]
        return: [B*B,D]
        input  [[a],
                [b]]
        return [[a,a],
                [b,a],
                [a,b],
                [b,b]]
    '''
    assert x1.ndimension() == 2 and x2.ndimension() == 2, 'Input dimension must be 2'
    # [a,b,c,a,b,c,a,b,c]
    # [a,a,a,b,b,b,c,c,c]
    x1_ = x1.repeat(x2.size(0), 1)
    x2_ = x2.repeat(1, x1.size(0)).view(-1, x1.size(1))
    # print(x1_, x2_)
    return torch.cat((x1_, x2_), dim=1)

def get_pair(data,label,sample_size):
    class_num = torch.unique(label)
    sample_idxs_1 = []
    sample_idxs_2 = []
    sample_per_class = int(np.sqrt(sample_size) / len(class_num))
    for lbl in class_num:
        class_bucket = torch.nonzero(data[label==lbl]).squeeze()
        sample_idxs_1.append(torch.from_numpy(np.random.choice(class_bucket, size=sample_per_class)).long()) # sampling with putback
        sample_idxs_2.append(torch.from_numpy(np.random.choice(class_bucket, size=sample_per_class)).long()) # sampling with putback
    sample_idxs_1 = torch.cat(sample_idxs_1) # (sample_per_class,)
    sample_idxs_2 = torch.cat(sample_idxs_2) # (sample_per_class,)
    # enumeration
    pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)


    sample_idxs_1 = pair_idxs[0]
    sample_idxs_2 = pair_idxs[1]
    return sample_idxs_1, sample_idxs_2
        

def get_pair_balance(data,label,sample_size):
    # x may be raw_feature/hidden_feature
    # sample max_class_num among all classes from which to conduct sampling, to keep label-balance
    
    class_num = torch.unique(label)
    # same-class pair generation
    sample_idxs_1 = []
    sample_idxs_2 = []
    sample_per_class_same = int(0.5 * sample_size / len(class_num))
    sample_per_class_diff = int(0.5 * sample_size / (len(class_num) * (len(class_num) - 1)))
    for lbl_1 in range(len(class_num)):
        class_bucket1 = torch.nonzero(data[label==lbl_1]).squeeze()
        for lbl_2 in range(len(class_num)):
            class_bucket2 = torch.nonzero(data[label==lbl_2]).squeeze()
            if lbl_1 == lbl_2:
                idx1 = torch.from_numpy(np.random.choice(class_bucket1, size=sample_per_class_same)).long() # sampling with putback
                idx2 = torch.from_numpy(np.random.choice(class_bucket2, size=sample_per_class_same)).long() # sampling with putback
            else:
                idx1 = torch.from_numpy(np.random.choice(class_bucket1, size=sample_per_class_diff)).long() # sampling with putback
                idx2 = torch.from_numpy(np.random.choice(class_bucket2, size=sample_per_class_diff)).long() # sampling with putback
            # pair_idxs = torch.stack((idx1, idx2), dim=0) # 2 * pair_num
            sample_idxs_1.append(idx1) 
            sample_idxs_2.append(idx2)
    sample_idxs_1 = torch.cat(sample_idxs_1, dim=0)
    sample_idxs_2 = torch.cat(sample_idxs_2, dim=0)

    return sample_idxs_1, sample_idxs_2

''' BL ODE util '''
# Counter of forward and backward passes.
class Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.cnt += 1

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.sum / self.cnt

    def get_value(self):
        return self.val

