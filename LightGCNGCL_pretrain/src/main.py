import argparse
from data_utils import create_dataset, create_dataloader, LightGCN_Dataset
from utils import init_seed, scipy_sparse_mat_to_torch_sparse_tensor
from LightGCNmodel import LightGCN, GAE, cateLightGCN
from trainer import Trainer
from Model import Model, vgae_encoder, vgae_decoder, vgae, DenoisingNet

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="the random seed")
parser.add_argument('--data_path', type=str, default='../data', help="data path")
parser.add_argument('--dataset', type=str, default='api', help="dataset")
parser.add_argument('--epochs', type=int, default=100, help="")
parser.add_argument('--batch_size', type=int, default=4096, help="")
parser.add_argument('--test_batch_size', type=int, default=64, help='')
parser.add_argument('--learning_rate', type=float, default=0.001, help="")
parser.add_argument('--reg_weight', type=float, default=1e-3, help="the weight decay of BPR Loss")
parser.add_argument('--weight_decay', type=float, default=0, help="the weight decay of optimizer")
parser.add_argument('--topk', type=int, default=25, help="")
parser.add_argument('--metrics', type=list, default=['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'], help="")
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer to update parameters. optimizer in [adam, sgd, adagrad, rmsprop]')
parser.add_argument('--data_split_ratio', type=list, default=[0.8],
                    help="the ration to split dataset to train, eval and test")
parser.add_argument('--neg_sample_num', type=int, default=1, help="each user-item interaction sampled negative num")
parser.add_argument('--embedding_size', type=int, default=64, help="the latent vector embedding size")
parser.add_argument('--n_layers', type=int, default=6, help="the graph convolution layer num")
parser.add_argument('--device', type=str, default='cuda', help="")
parser.add_argument('--gpu_id', type=int, default=0, help="")
parser.add_argument('--neg_prefix', type=str, default='neg_', help="")
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_log', help='the path to save tensorboard')

args = parser.parse_args()
# init random seed
init_seed(args.seed)

print('Used dataset is {}'.format(args.dataset))
# load dataset
dataset = LightGCN_Dataset('api', args)
train_dataset, interaction_matrix, mask_index = dataset.get_train_dataset()
test_users, ground_true_items, test_data = dataset.get_test_data()
# load dataloader
train_data = create_dataloader(train_dataset, args.batch_size, training=True)
test_data = create_dataloader(test_data, args.test_batch_size, training=False)

encoder = vgae_encoder().cuda()
decoder = vgae_decoder().cuda()
generator = vgae(encoder, decoder).cuda()

model = LightGCN(args, dataset, interaction_matrix).to(args.device)




#____________图扩散_________________________________
import numpy as np
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.sparse import csr_matrix
import scipy as sp
import torch
def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)
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

print("----------Training-----------------------")
trainer = Trainer(args, model, generator)
loss = []
for epoch in range(args.epochs):
    # training
    l = trainer.train_an_epoch(epoch, train_data, ppr_adj_norm, test_data, ground_true_items, mask_index, epoch_id=epoch + 1)
    loss.append(l)
    # testing
trainer.evaluate(interaction_matrix, test_data, ground_true_items, mask_index, epoch_id=epoch + 1)
import json

# my_list = [1, 2, 3, 'a', 'b', 'c']

# 保存到文件
with open('list.json', 'w') as f:
    json.dump(loss, f)

# 从文件读取
# with open('list.json', 'r') as f:
#     loaded_list = json.load(f)
