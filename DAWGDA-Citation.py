
# coding=utf-8
import os
import torch.backends.cudnn as cudnn
from GCN_model import GCN,Attention
from utils import *
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


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-2)
parser.add_argument('--cuda', type=str, default="0")
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--gfeat', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--nfeat', type=float, default=6775,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data_src', type=str, default='dblpv7',
                    help='source dataset name')
parser.add_argument('--data_trg', type=str, default='acmv9',
                    help='target dataset name')
parser.add_argument('--classes', type=int, default=5,
                    help='classes number')
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--seed', type=int, default=100,
                    help='seed')
parser.add_argument('--lambda_sso', type=float, default=1,
                    help='lambda_sso')
parser.add_argument('--lambda_inter', type=float, default=0.1,
                    help='lambda_inter')

args = parser.parse_args()
cuda = True
cudnn.benchmark = True

device = torch.device(('cuda:' +args.cuda) if torch.cuda.is_available() else 'cpu')
print(device)

manual_seed = int(args.seed)
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.use_deterministic_algorithms(True)


adj_s, features_s, labels_s = load_data_citation(dataset=args.data_src)
adj_t, features_t, labels_t = load_data_citation(dataset=args.data_trg)


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

id_s = adj_s.coalesce().indices()
id_t = adj_t.coalesce().indices()

T1 = time.perf_counter()
X_n_s = diff_adj(adj_s, features_s)
X_n_t = diff_adj(adj_t, features_t)
T2 = time.perf_counter()
print('GDC time :{}s'.format((T2-T1)))


value_s = adj_s.coalesce().values()
value_t = adj_t.coalesce().values()
adj_s_dense = adj_s.to_dense()
adj_t_dense = adj_t.to_dense()


def predict(feature,adj,ppmi):
    _,basic_encoded_output,_ = shared_encoder_l(feature,adj)
    _,ppmi_encoded_output,_ = shared_encoder_g(feature,ppmi)

    encoded_output = att_model([basic_encoded_output,ppmi_encoded_output])
    logits = cls_model(encoded_output)
    return logits,encoded_output

def evaluate(preds, labels):
    accuracy1 = accuracy(preds, labels)
    return accuracy1

def test(feature,adj,ppmi,label):
    for model in models:
        model.eval()
    logits,encoced_output = predict(feature,adj,ppmi)
    labels = label
    accuracy = evaluate(logits, labels)
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

    input1_l2_norm = torch.norm(input1, p=2, dim=0, keepdim=True).detach()
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1))

    input2_l2_norm = torch.norm(input2, p=2, dim=0, keepdim=True).detach()
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2))

    diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
    
    return diff_loss




if cuda:
    adj_s = adj_s.to(device)
    adj_t = adj_t.to(device)
    labels_s = labels_s.to(device)
    labels_t = labels_t.to(device)
    features_s = features_s.to(device)
    features_t = features_t.to(device)
    X_n_s = X_n_s.to(device)
    X_n_t = X_n_t.to(device)
    value_s = value_s.to(device)
    value_t = value_t.to(device)
    id_s = id_s.to(device)
    id_t = id_t.to(device)
    adj_s_dense = adj_s_dense.to(device)
    adj_t_dense = adj_t_dense.to(device)

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
    encoded_source = att_model([shared_encoded_source1,ppmi_encoded_source])
    encoded_target = att_model([shared_encoded_target1,ppmi_encoded_target])

    '''dgi loss'''
    positive_s,loss_dgi_s = dgi_s(features_s,adj_s)
    positive_t,loss_dgi_t = dgi_t(features_t,adj_t)

    loss_dgi = loss_dgi_s + loss_dgi_t

    loss_diff = DiffLoss(positive_s,shared_encoded_source1) + DiffLoss(positive_t,shared_encoded_target1)


    ''' compute node classification loss for S '''
    source_logits = cls_model(encoded_source)
    cls_loss_source = cls_loss(source_logits, labels_s)
    source_acc = evaluate(source_logits, labels_s)

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
    dis = torch.cdist(encoded_source,encoded_source)**2
    dnl_s1 = torch.mean(torch.sum(torch.mul(adj_s_dense,dis), 1))
    dis = torch.cdist(encoded_target,encoded_target)**2
    dnl_t1 = torch.mean(torch.sum(torch.mul(adj_t_dense,dis), 1))
    
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
    loss_intra = cls_loss_source + loss_grl + loss_entropy  * (epoch / args.n_epoch * 0.01) + loss_diff*args.lambda_sso  + loss_dgi
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
            best_trg_label = target_logits.max(1)[1].type_as(labels_t)
            best_epoch = epoch
            best_encoded_output = encoded_output
    if (epoch+1)%50 == 0:
        print('epoch: {}, acc_test_trg: {},loss:{},loss_intra:{},loss_inter:{}'.format(epoch,acc_trg,loss,loss_intra,loss_inter))

T2 = time.perf_counter()
print('time :{}s'.format((T2-T1)))
print('best acc :{}'.format(best_acc))
print('best epoch :{}'.format(best_epoch))
print('done')
print('lr:{},epoch:{},lambda_inter:{},seed:{}'.format(args.lr,args.n_epoch,args.lambda_inter,manual_seed))
print('all_time:{}'.format(all_time))
print('per_epoch_time:{}'.format(all_time/args.n_epoch))
