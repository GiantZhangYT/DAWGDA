# coding=utf-8
import os
import torch.backends.cudnn as cudnn
from GCN_model import GCN,Attention
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import random
import torch.utils.data
import argparse
from dgi import DGI
import time
from torch_sparse import coalesce
from torch_geometric.transforms import GDC
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-2)
parser.add_argument('--cuda', type=str, default="0")
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--gfeat', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--nfeat', type=float, default=128,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--classes', type=int, default=40,
                    help='classes number')
parser.add_argument('--lambda_sso', type=float, default=1e-3,
                    help='sso loss')
parser.add_argument('--seed', type=int, default=100,
                    help='seed')
parser.add_argument('--lambda_inter', type=float, default=1e-5,
                    help='inter slpha')

args = parser.parse_args()
cuda = True
cudnn.benchmark = True


device = torch.device(('cuda:' +args.cuda) if torch.cuda.is_available() else 'cpu')
print(device)
manual_seed = random.randint(1, 10000)
manual_seed = int(args.seed)
# manual_seed = 200
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
# dgl.random.seed(manual_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.use_deterministic_algorithms(True)


import numpy as np
import torch


def take_second(element):
    return element[1]

def load_ogb_arxiv(data_dir, year_bound, proportion=1.0):
    import ogb.nodeproppred

    dataset = ogb.nodeproppred.NodePropPredDataset(name='ogbn-arxiv', root=data_dir)
    graph = dataset.graph

    node_years = graph['node_year']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years))

    edges = graph['edge_index']
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] <= year_bound[1] and node_years[edges[1][i]] <= year_bound[1]:
            d[edges[0][i]] += 1
            d[edges[1][i]] += 1

    nodes = []
    for i, year in enumerate(node_years):
        if year <= year_bound[1]:
            nodes.append([i, d[i]])

    nodes.sort(key=take_second, reverse=True)

    nodes = nodes[: int(proportion * len(nodes))]

    result_edges = []
    result_features = []
    result_labels = []

    for node in nodes:
        result_features.append(graph['node_feat'][node[0]])
    result_features = np.array(result_features)

    ids = {}
    for i, node in enumerate(nodes):
        ids[node[0]] = i

    for i in range(edges.shape[1]):
        if edges[0][i] in ids and edges[1][i] in ids:
            result_edges.append([ids[edges[0][i]], ids[edges[1][i]]])
    result_edges = np.array(result_edges).transpose(1, 0)

    result_labels = dataset.labels[[node[0] for node in nodes]]

    edge_index = torch.tensor(result_edges, dtype=torch.long)
    node_feat = torch.tensor(result_features, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': node_feat.size(0)}
    dataset.label = torch.tensor(result_labels)
    node_years_new = [node_years[node[0]] for node in nodes]
    dataset.test_mask = (torch.tensor(node_years_new) > year_bound[0])

    return dataset

def diff_adj(adj_s, features_s):
    gdc = GDC()
    diffusion_kwargs=dict(method='ppr', alpha=0.05,eps=0.001)
    id_s = adj_s.coalesce().indices()
    edge_weight = torch.ones(id_s.size(1))
    edge_index = id_s

    edge_index,edge_weight = gdc.diffusion_matrix_approx(edge_index, edge_weight,adj_s.shape[0],'sym',**diffusion_kwargs)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, features_s.shape[0],features_s.shape[0])

    edge_index, edge_weight = gdc.transition_matrix(edge_index, edge_weight, features_s.shape[0], 'col')

    adj_s = torch.sparse_coo_tensor(edge_index, edge_weight, (features_s.shape[0],features_s.shape[0]))
    return adj_s

# load arxiv data
dataset_source = load_ogb_arxiv("./data/arxiv",  [2011, 2014],1.0)
dataset_target = load_ogb_arxiv("./data/arxiv",  [2018, 2020],1.0)


adj_s = torch.sparse_coo_tensor(dataset_source.graph['edge_index'], torch.ones(dataset_source.graph['edge_index'].shape[1]), (dataset_source.graph['node_feat'].shape[0],dataset_source.graph['node_feat'].shape[0]))
adj_t = torch.sparse_coo_tensor(dataset_target.graph['edge_index'], torch.ones(dataset_target.graph['edge_index'].shape[1]), (dataset_target.graph['node_feat'].shape[0],dataset_target.graph['node_feat'].shape[0]))
features_s = dataset_source.graph['node_feat']
features_t = dataset_target.graph['node_feat']


labels_s = (dataset_source.label).reshape(-1)

labels_t = (dataset_target.label).reshape(-1)

id_s = adj_s.coalesce().indices()
id_t = adj_t.coalesce().indices()


T1 = time.perf_counter()
X_n_s = diff_adj(adj_s, features_s)
X_n_t = diff_adj(adj_t, features_t)
T2 = time.perf_counter()
print('time :{}s'.format((T2-T1)))


def predict(feature,adj,ppmi):
    _,basic_encoded_output,_ = shared_encoder_l(feature,adj)
    _,ppmi_encoded_output,_ = shared_encoder_g(feature,ppmi)

    encoded_output = att_model([basic_encoded_output,ppmi_encoded_output])
    logits = cls_model(encoded_output)
    return logits,encoded_output

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def evaluate(preds, labels):
    accuracy1 = accuracy(preds, labels)
    return accuracy1

def test(feature,adj,ppmi,label):
    for model in models:
        model.eval()
    logits,encoced_output = predict(feature,adj,ppmi)
    labels = label
    accuracy = evaluate(logits[dataset_target.test_mask], labels[dataset_target.test_mask])
    return accuracy,encoced_output

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None

class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

def calculate_sim(encoded_source,encoded_target):
    if encoded_source.shape[0] < encoded_target.shape[0]:
        encoded_source = data_add(encoded_source,encoded_target)
    else:
        encoded_target = data_add(encoded_target,encoded_source)
    loss_sim = torch.norm(torch.mean(encoded_source,dim=0)-torch.mean(encoded_target,dim=0))
    return loss_sim


def data_add(small_data,big_data):
    add_num = big_data.shape[0] - small_data.shape[0]
    rand_index = np.random.randint(0,small_data.shape[0],(add_num,2))
    add_data = small_data[rand_index,:]
    return torch.cat((small_data,torch.mean(add_data,dim=1)))


''' set loss function '''
cls_loss = nn.CrossEntropyLoss().to(device)


''' shared encoder (including Local GCN and Global GCN) '''
shared_encoder_l = GCN(nfeat=args.nfeat, nhid=args.hidden, nclass=args.gfeat, dropout=args.dropout).to(device)
shared_encoder_g = GCN(nfeat=args.nfeat, nhid=args.hidden, nclass=args.gfeat, dropout=args.dropout).to(device)

''' node classifier model '''
cls_model = nn.Sequential(
    nn.Linear(args.gfeat, args.classes),
).to(device)

''' domain discriminator model '''
domain_model = nn.Sequential(
    GRL(),
    nn.Linear(args.gfeat, 10),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(10, 2),
).to(device)

''' attention layer for local and global features '''
att_model = Attention(args.gfeat).to(device)

dgi_s = DGI(
    args.nfeat,
    args.hidden,
    args.gfeat,
    nn.ELU(args.hidden),
    args.dropout,
).to(device)


dgi_t = DGI(
    args.nfeat,
    args.hidden,
    args.gfeat,
    nn.ELU(args.hidden),
    args.dropout,
).to(device)


''' the set of models used in DAWGDA '''
models = [shared_encoder_g,shared_encoder_l,cls_model,domain_model,att_model,dgi_s,dgi_t]
params = itertools.chain(*[model.parameters() for model in models])

''' setup optimizer '''
optimizer = torch.optim.Adam(params, lr=args.lr,weight_decay=5e-4)

''' training '''
best_acc = 0
acc_record = []
loss_out_record =[]
def DiffLoss(input1,input2):
    batch_size = input1.size(0)
    input1 = input1.view(batch_size, -1)
    input2 = input2.view(batch_size, -1)
    epsilon = 1e-8
     
    input1_l2_norm = torch.norm(input1, p=2, dim=0, keepdim=True).detach()
    input1_l2_norm = torch.clamp(input1_l2_norm, min=epsilon)
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1))

    input2_l2_norm = torch.norm(input2, p=2, dim=0, keepdim=True).detach()
    input2_l2_norm = torch.clamp(input2_l2_norm, min=epsilon)
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2))

    diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

    return diff_loss

mask = torch.ones(features_s.shape[0],dtype=torch.uint8)
new_index = torch.tensor(np.array((np.where(mask==1)))).reshape(-1)
ori_labels = labels_s.to(device)



if cuda:
    adj_s = adj_s.to(device)
    adj_t = adj_t.to(device)
    labels_s = labels_s.to(device)
    labels_t = labels_t.to(device)
    features_s = features_s.to(device)
    features_t = features_t.to(device)
    X_n_s = X_n_s.to(device)
    X_n_t = X_n_t.to(device)
    id_s = id_s.to(device)
    id_t = id_t.to(device)



T1 = time.perf_counter()

all_time=0

for epoch in range(args.n_epoch):
    len_dataloader = min(labels_s.shape[0],labels_t.shape[0])
    global rate
    rate = min((epoch + 1) / args.n_epoch, 0.05)

    for model in models:
        model.train()
    optimizer.zero_grad()
    
    time_start = time.perf_counter()
    z_s, shared_encoded_source1, shared_encoded_source2 = shared_encoder_l(features_s, adj_s)
    z_t, shared_encoded_target1, shared_encoded_target2 = shared_encoder_l(features_t, adj_t)

    z_s_p,ppmi_encoded_source,ppmi_encoded_source2 = shared_encoder_g(features_s, X_n_s)
    z_t_p,ppmi_encoded_target,ppmi_encoded_target2 = shared_encoder_g(features_t, X_n_t)

    ''' the node representations after shared encoder for S and T '''
    ori_encoded_source = att_model([shared_encoded_source1,ppmi_encoded_source])
    ori_encoded_target = att_model([shared_encoded_target1,ppmi_encoded_target])
    encoded_source = ori_encoded_source[dataset_source.test_mask]
    encoded_target = ori_encoded_target[dataset_target.test_mask]

    '''dgi loss'''
    positive_s,loss_dgi_s = dgi_s(features_s,adj_s,dataset_source.test_mask)
    positive_t,loss_dgi_t = dgi_t(features_t,adj_t,dataset_source.test_mask)

    loss_dgi = loss_dgi_s + loss_dgi_t

    # loss sso
    loss_diff = DiffLoss(positive_s[dataset_source.test_mask],shared_encoded_source1[dataset_source.test_mask]) + DiffLoss(positive_t[dataset_target.test_mask],shared_encoded_target1[dataset_target.test_mask])


    ''' compute node classification loss for S '''
    source_logits = cls_model(encoded_source)
    cls_loss_source = cls_loss(source_logits, labels_s[dataset_source.test_mask])
    source_acc = evaluate(source_logits, labels_s[dataset_source.test_mask])

    ''' compute domain classifier loss for both S and T '''
    domain_output_s = domain_model(encoded_source)
    domain_output_t = domain_model(encoded_target)
    err_s_domain = cls_loss(domain_output_s,
        torch.zeros(domain_output_s.size(0)).type(torch.LongTensor).to(device))
    err_t_domain = cls_loss(domain_output_t,
        torch.ones(domain_output_t.size(0)).type(torch.LongTensor).to(device))
    loss_grl = err_s_domain + err_t_domain

    ''' compute entropy loss for T '''
    target_logits = cls_model(encoded_target)
    target_probs = F.softmax(target_logits, dim=-1)
    target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
    loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))



    ''' compute d_NL '''
    A = ori_encoded_source[id_s[0,:]]
    B = ori_encoded_source[id_s[1,:]]
    diff = A - B
    squared_diff = diff ** 2
    row_sums = squared_diff.sum()
    dnl_s1 = row_sums/features_s.shape[0]

    A = ori_encoded_target[id_t[0,:]]
    B = ori_encoded_target[id_t[1,:]]
    diff = A - B
    squared_diff = diff ** 2
    row_sums = squared_diff.sum()
    dnl_t1 = row_sums/features_t.shape[0]

    dnl = (dnl_s1/id_s.shape[1] + dnl_t1/id_t.shape[1])
    

    ''' compute d_GL '''
    dGL = calculate_sim(encoded_source,encoded_target)

    ''' compute d_CL '''
    sim_c = 0
    for i in range(args.classes):
        current_class_source = encoded_source[source_logits.max(1)[1]==i]
        current_class_target = encoded_target[target_probs.max(1)[1]==i]
        if len(current_class_source)==0 or len(current_class_target)==0:
            continue
        sim_c += calculate_sim(current_class_source,current_class_target)
    sim_c = sim_c/args.classes
    sim_cc = 0
    for i in range(args.classes):
        for j in range(i,args.classes):
            current_class_source = encoded_source[source_logits.max(1)[1]==i]
            current_class_target = encoded_target[target_probs.max(1)[1]==i]
            if len(current_class_source)==0 or len(current_class_target)==0:
                continue
            sim_cc += calculate_sim(current_class_source,current_class_target)

    sim_cc = sim_cc/((args.classes*(args.classes-1)/2)*features_s.shape[0]*features_t.shape[0])
    dCL = sim_c - sim_cc

    ''' compute overall loss '''
    loss_intra = cls_loss_source + loss_grl + loss_entropy  * (epoch / args.n_epoch * 0.01) +  loss_dgi + loss_diff*args.lambda_sso
    loss_inter = dGL + dnl + dCL
    loss = loss_intra +   args.lambda_inter *loss_inter
   
   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    time_end = time.perf_counter()

    all_time += time_end-time_start


    if (epoch+1)%1 == 0:
        acc_trg,encoded_output = test(features_t,adj_t,X_n_t,labels_t)
        acc_record.append(acc_trg.cpu())
        if acc_trg > best_acc:
            best_acc = acc_trg
            best_mask = mask
            best_epoch = epoch
            best_encoded_output = encoded_output
    if (epoch+1)%5 == 0:
        print('epoch: {}, acc_test_trg: {},loss:{},loss_intra:{},loss_inter:{},loss_sso:{}'.format(epoch,acc_trg,loss,loss_intra,loss_inter,loss_diff))


T2 = time.perf_counter()
print('time :{}s'.format((T2-T1)))
print('best acc :{}'.format(best_acc))
print('best epoch :{}'.format(best_epoch))
print('done')
print('lr:{},epoch:{},lambda_inter:{},seed:{}'.format(args.lr,args.n_epoch,args.lambda_inter,manual_seed))
print('all_time:{}'.format(all_time))
print('per_epoch_time:{}'.format(all_time/args.n_epoch))
