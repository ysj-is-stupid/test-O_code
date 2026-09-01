import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
import scipy.sparse as sp
import torch.nn.functional as F
import copy
import os


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(128, 128)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        self.features = nn.Parameter(nn.init.xavier_uniform_(torch.empty(adj.shape[0], 128)))
        self.base_gcn = GraphConvSparse(128, 128, adj)
        # self.gcn_mean = GraphConvSparse(128, 128, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        # z = self.mean = self.gcn_mean(hidden)
        return hidden

    def forward(self):
        Z = self.encode(self.features)
        A_pred = dot_product_decode(Z)
        return A_pred

class LightGCN(nn.Module):
    def __init__(self, args, dataset, interaction_matrix):
        super(LightGCN, self).__init__()
        self.args = args
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.embeddings_list = [None] * args.n_layers
        self.interaction_matrix = interaction_matrix

        self.n_layers = args.n_layers
        self.reg_weight = args.reg_weight
        self.gamma = 1e-10
        # define layers and loss

        # get adjacent matrix
        self.norm_adj_matrix = self._get_norm_adj_mat(self.interaction_matrix)
        self.restore_user_e = None
        self.restore_item_e = None
        self.weights = nn.Parameter(torch.FloatTensor(self.n_layers, 1))

        # parameters initialization
        self._init_weights()
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        # self.attentions = GraphAttentionLayer(64, 64, dropout=0.6, alpha=0.2, concat=True)

    def _init_weights(self):
        xavier_uniform_(self.user_embedding.weight.data)
        xavier_uniform_(self.item_embedding.weight.data)

    def _get_norm_adj_mat(self, interaction_matrix):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)

        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # covert norm_adj matrix to tensor  time4: 0.0040
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = np.array([row, col])
        i = torch.LongTensor(i)
        # i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL.to(self.args.device)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def light_forward(self, norm_adj_matrix):
        """
        LightGCN forward propagation.
        Args:
            norm_adj_matrix: Normalized adjacency matrix (sparse or dense).
        Returns:
            lightgcn_all_embeddings: Combined embeddings of users and items.
        """
        # Get initial embeddings
        all_embeddings = self.get_ego_embeddings()
        self.embeddings_list[0] = all_embeddings

        # Propagate embeddings through layers
        for layer_idx in range(1, self.n_layers):
            if norm_adj_matrix.is_sparse:
                # Sparse matrix multiplication
                self.embeddings_list[layer_idx] = torch.sparse.mm(norm_adj_matrix, self.embeddings_list[layer_idx - 1])
            else:
                # Dense matrix multiplication
                self.embeddings_list[layer_idx] = torch.mm(norm_adj_matrix, self.embeddings_list[layer_idx - 1])

        # Combine embeddings from all layers
        lightgcn_all_embeddings = torch.stack(self.embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        return lightgcn_all_embeddings

    def predict(self, user):
        """
        :param user: the id of batch users
        # :return:
        # """
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, self.c_restore_user_e, self.c_restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        # cate_u_embeddings = self.cate_restore_user_e[user]
        # u_embeddings = torch.cat()
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores


class vgae_encoder(LightGCN):
    def __init__(self, args, dataset, interaction_matrix):
        super(vgae_encoder, self, ).__init__(args, dataset, interaction_matrix)
        # hidden = args.latdim
        self.encoder_mean = nn.Sequential(nn.Linear(args.embedding_size, args.embedding_size), nn.ReLU(inplace=True),
                                          nn.Linear(args.embedding_size, args.embedding_size))
        self.encoder_std = nn.Sequential(nn.Linear(args.embedding_size, args.embedding_size), nn.ReLU(inplace=True),
                                         nn.Linear(args.embedding_size, args.embedding_size),
                                         nn.Softplus())

    def forward(self):
        x = self.light_forward(self.norm_adj_matrix)
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).cuda()
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std


class vgae_decoder(nn.Module):
    def __init__(self, hidden=128):
        super(vgae_decoder, self).__init__()
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, 1))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):

        x_user, x_item = torch.split(x, [6300, 1608], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda())
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda())
        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = torch.clamp(- (scoreDiff).sigmoid().log().sum() / 4096, -5.0, 5.0)
        regLoss = calcRegLoss(encoder) * 1e-5

        beta = 0.1
        loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()

        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder()
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
        return loss

    def generate(self):
        x, _, _ = self.encoder()
        edge_index = self.encoder.norm_adj_matrix._indices()
        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))

        vals = self.encoder.norm_adj_matrix._values()
        idxs = self.encoder.norm_adj_matrix._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)

        newVals = vals[mask]

        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, self.encoder.norm_adj_matrix.shape)


def innerProduct(usrEmbeds, itmEmbeds):
    return torch.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


class cateLightGCN(nn.Module):
    def __init__(self, args, dataset, cate_matrix, cate_contrast, mashup_v, api_v):
        super(cateLightGCN, self).__init__()
        self.args = args
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        self.embedding_size = args.embedding_size
        # self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        # self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.mashup_cate_vector = nn.Embedding(self.n_users, self.embedding_size).cuda()
        self.api_cate_vector = nn.Embedding(self.n_items, self.embedding_size).cuda()
        self.mashup_cate_vector.weight = torch.nn.Parameter(mashup_v.cuda())
        self.api_cate_vector.weight = torch.nn.Parameter(api_v.cuda())
        self.C_list = [None] * args.n_layers
        self.C_list[0] = torch.cat((self.mashup_cate_vector.weight, self.api_cate_vector.weight), dim=0)
        self.c_C_list = [None] * args.n_layers
        self.c_C_list[0] = torch.cat((self.mashup_cate_vector.weight, self.api_cate_vector.weight), dim=0)
        # load dataset info
        self.cate_matrix = cate_matrix
        self.contrast = cate_contrast
        self.n_layers = args.n_layers
        self.reg_weight = args.reg_weight
        self.gamma = 1e-10
        # define layers and loss
        # get adjacent matrix
        self.norm_cate_matrix = self._get_norm_adj_mat(self.cate_matrix)
        # adj used to add and delete edge
        self.restore_user_cate = None
        self.restore_item_cate = None
        self.c_restore_user_cate = None
        self.c_restore_item_cate = None

        # parameters initialization
        self._init_weights()
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        # self.liner = nn.Linear(self.mashup_cate_vector.shape[1], 64).cuda()

    def _init_weights(self):
        xavier_uniform_(self.mashup_cate_vector.weight)
        xavier_uniform_(self.api_cate_vector.weight)

    def _get_norm_adj_mat(self, interaction_matrix):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)

        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # covert norm_adj matrix to tensor  time4: 0.0040
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = np.array([row, col])
        i = torch.LongTensor(i)
        # i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL.to(self.args.device)

    def get_c_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        mashup_cate_vector = self.mashup_cate_vector.weight
        api_cate_vector = self.api_cate_vector.weight
        cate_embeddings = torch.cat([mashup_cate_vector, api_cate_vector], dim=0)
        return cate_embeddings

    def forward(self):
        self.c_C_list[0] = self.C_list[0] = self.get_c_ego_embeddings().cuda()

        for layer_idx in range(1, self.n_layers):
            self.C_list[layer_idx] = torch.sparse.mm(self.norm_cate_matrix.float(), self.C_list[layer_idx - 1].float())
            self.c_C_list[layer_idx] = torch.sparse.mm(self.contrast.float(), self.c_C_list[layer_idx - 1].float())

        cate_all_embeddings = torch.stack(self.C_list, dim=1)
        cate_all_embeddings = torch.mean(cate_all_embeddings, dim=1)

        c_cate_all_embeddings = torch.stack(self.C_list, dim=1)
        c_cate_all_embeddings = torch.mean(c_cate_all_embeddings, dim=1)

        user_cate_embeddings, item_cate_embeddings = torch.split(cate_all_embeddings,
                                                                 [self.n_users, self.n_items])
        c_user_cate_embeddings, c_item_cate_embeddings = torch.split(c_cate_all_embeddings,
                                                                     [self.n_users, self.n_items])
        return user_cate_embeddings, item_cate_embeddings, c_user_cate_embeddings, c_item_cate_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        # if self.restore_user_e is not None or self.restore_item_e is not None:
        #     self.restore_user_e, self.restore_item_e = None, None

        user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
        user = user.cuda()
        pos_item = pos_item.cuda()
        neg_item = neg_item.cuda()
        user_cate_embeddings, item_cate_embeddings, c_user_cate_embeddings, c_item_cate_embeddings = self.forward()

        u_cate_embeddings = user_cate_embeddings[user]
        item_cate_pos_embeddings = item_cate_embeddings[pos_item]
        item_cate_neg_embeddings = item_cate_embeddings[neg_item]
        iids = torch.concat([pos_item, neg_item], dim=0)
        i_emb = item_cate_embeddings[iids]
        # c_u_embeddings = c_user_cate_embeddings[user]
        # c_i_emb = c_item_cate_embeddings[iids]
        # neg_score = torch.log(torch.exp(c_u_embeddings @ user_cate_embeddings.T / 0.5).sum(1) + 1e-8).mean()
        # neg_score += torch.log(torch.exp(c_i_emb @ item_cate_embeddings.T / 0.5).sum(1) + 1e-8).mean()
        # pos_score = (torch.clamp((c_u_embeddings * u_cate_embeddings).sum(1) / 0.5, -5.0, 5.0)).mean() + (
        #     torch.clamp((c_i_emb * i_emb).sum(1) / 0.5, -5.0, 5.0)).mean()
        # loss_s = -pos_score + neg_score

        # calculate BPR Loss
        pos_scores = torch.mul(u_cate_embeddings, item_cate_pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_cate_embeddings, item_cate_neg_embeddings).sum(dim=1)
        mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))  # LightGCN source code used

        # calculate BPR Loss
        u_ego_embeddings = self.mashup_cate_vector(user)
        pos_ego_embeddings = self.api_cate_vector(pos_item)
        neg_ego_embeddings = self.api_cate_vector(neg_item)
        reg_loss = (1 / 2) * (u_ego_embeddings.norm(2).pow(2) +
                              pos_ego_embeddings.norm(2).pow(2) +
                              neg_ego_embeddings.norm(2).pow(2)) / float(len(user))  # LightGCN source code used

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, user):
        """
        :param user: the id of batch users
        # :return:
        # """
        self.restore_user_cate, self.restore_item_cate = self.mashup_cate_vector.weight, self.api_cate_vector.weight
        # ,   self.c_restore_user_cate, self.c_restore_item_cate,
        u_embeddings = self.restore_user_cate[user]
        scores = torch.matmul(u_embeddings, self.restore_item_cate.transpose(0, 1))
        return scores
