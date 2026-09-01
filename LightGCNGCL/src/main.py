import argparse
import pickle
from scipy.sparse import csr_matrix
# from sklearn.preprocessing import normalize
import numpy as np
import time
# from scipy.sparse import csr_matrix
import scipy as sp
import torch
from scipy.sparse import coo_matrix, hstack, vstack
from DataHandler import DataHandler
from data_utils import create_dataset, create_dataloader, LightGCN_Dataset
from utils import init_seed, scipy_sparse_mat_to_torch_sparse_tensor
from LightGCNmodel import LightGCN, GAE, cateLightGCN, vgae, vgae_decoder, vgae_encoder
from trainer import Trainer
# ssh -p 39112 root@connect.cqa1.seetacloud.com
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="the random seed")
parser.add_argument('--data_path', type=str, default='../data', help="data path")
parser.add_argument('--dataset', type=str, default='api', help="dataset")
parser.add_argument('--epochs', type=int, default=10, help="")
parser.add_argument('--batch_size', type=int, default=4096, help="")
parser.add_argument('--test_batch_size', type=int, default=64, help='')
parser.add_argument('--learning_rate', type=float, default=0.0001, help="")
parser.add_argument('--reg_weight', type=float, default=1e-3, help="the weight decay of BPR Loss")
parser.add_argument('--weight_decay', type=float, default=1e-3, help="the weight decay of optimizer")
parser.add_argument('--topk', type=int, default=20, help="")
parser.add_argument('--metrics', type=list, default=['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'], help="")
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer to update parameters. optimizer in [adam, sgd, adagrad, rmsprop]')
parser.add_argument('--data_split_ratio', type=list, default=[0.6],
                    help="the ration to split dataset to train, eval and test")
parser.add_argument('--neg_sample_num', type=int, default=1, help="each user-item interaction sampled negative num")
parser.add_argument('--embedding_size', type=int, default=128, help="the latent vector embedding size")
parser.add_argument('--n_layers', type=int, default=7, help="the graph convolution layer num")
parser.add_argument('--device', type=str, default='cuda', help="")
parser.add_argument('--gpu_id', type=int, default=0, help="")
parser.add_argument('--neg_prefix', type=str, default='neg_', help="")
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_log', help='the path to save tensorboard')

args = parser.parse_args()

# init random seed
init_seed(args.seed)

# def preprocess_graph(adj):
#     adj = sp.coo_matrix(adj)
#     adj_ = adj + sp.sparse.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.sparse.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
    # if not sp.isspmatrix_coo(sparse_mx):
    #     sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    # adj = sp.coo_matrix(adj)
    adj_ = adj + sp.sparse.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.sparse.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

print('Used dataset is {}'.format(args.dataset))
dataset = LightGCN_Dataset('api', args)
# cate_dataset = LightGCN_Dataset('cate', args)
train_dataset, interaction_matrix, mask_index = dataset.get_train_dataset()
test_users, ground_true_items, test_data = dataset.get_test_data()


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)
zaro1 = coo_matrix((6298, 6298))
zaro2 = coo_matrix((1609, 1609))
top = interaction_matrix
top = hstack([top, zaro1])
right = interaction_matrix.transpose()
bottom = hstack([zaro2, right])
ppr = vstack([top, bottom])
ppr = ppr.toarray()
ppr_matrix = get_ppr_matrix(ppr, alpha=0.1)
ppr_matrix = ppr_matrix > 0.02
ppr_matrix = ppr_matrix.astype(int)
ppr_matrix = csr_matrix(ppr_matrix)
ppr_matrix = preprocess_graph(ppr_matrix)
ppr_adj_norm = torch.sparse.FloatTensor(torch.LongTensor(ppr_matrix[0].T),
                            torch.FloatTensor(ppr_matrix[1]),
                            torch.Size(ppr_matrix[2]))

# ppr_matrix = torch.tensor(ppr_matrix)
# load dataloader
train_data = create_dataloader(train_dataset, args.batch_size, training=True)
test_data = create_dataloader(test_data, args.test_batch_size, training=False)

n_users = dataset.user_num
n_items = dataset.item_num
# nonzero_indices = torch.nonzero(contrast)
model = LightGCN(args, dataset, interaction_matrix).to(args.device)
tensor = torch.FloatTensor(interaction_matrix.toarray())

# 将PyTorch张量转换为tuple格式
tuple_tensor = tuple(map(tuple, tensor.tolist()))
generator = GAE(ppr_adj_norm)


print("----------Training-----------------------")
trainer = Trainer(args, model, generator)
for epoch in range(args.epochs):
    # training
    trainer.train_an_epoch(train_data,  epoch_id=epoch + 1)
    # testing
    trainer.evaluate(test_data, ground_true_items, mask_index, interaction_matrix, epoch_id=epoch + 1)

