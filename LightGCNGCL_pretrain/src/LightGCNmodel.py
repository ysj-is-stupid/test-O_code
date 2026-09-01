import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
import scipy.sparse as sp
import torch.nn.functional as F

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
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
        self.features = nn.Parameter(nn.init.xavier_uniform_(torch.empty(adj.shape[0], 64)))
        self.base_gcn = GraphConvSparse(64, 64, adj)
        self.gcn_mean = GraphConvSparse(64, 64, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

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
        self.embeddings_list = [None]*args.n_layers
        self.embeddings_list[0] = [torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)]
        self.K_list = [None]*args.n_layers
        self.K_list[0] = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        self.c_embeddings_list = [None]*args.n_layers
        self.c_embeddings_list[0] = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        # self.C_list = [None]*args.n_layers
        # self.C_list[0] = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        # load dataset info

        self.interaction_matrix = interaction_matrix
        # self.cate_matrix = cate_matrix
        # self.contrast = contrast
        # load parameters

        self.n_layers = args.n_layers
        self.reg_weight = args.reg_weight
        self.gamma = 1e-10
        # define layers and loss

        # self.user_embedding = user_embedding
        # self.item_embedding = item_embedding

        # get adjacent matrix
        self.norm_adj_matrix = self._get_norm_adj_mat(self.interaction_matrix)
        # self.cate_adj_matrix = self._get_norm_adj_mat(self.cate_matrix)

        # adj used to add and delete edge
        # self.contrast_adj = self._get_norm_adj_mat(self.contrast)
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.c_restore_user_e = None
        self.c_restore_item_e = None
        self.c_weight = 0
        # self.cate_restore_user_e = None
        # self.cate_restore_item_e = None

        # parameters initialization
        self._init_weights()
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

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

    def forward(self, contrast_view):
        all_embeddings = self.get_ego_embeddings()
        self.embeddings_list[0] = all_embeddings
        self.c_embeddings_list[0] = all_embeddings
        self.K_list[0] = all_embeddings
        # self.C_list[0] = all_embeddings

        for layer_idx in range(1, self.n_layers):
            self.K_list[layer_idx] = torch.sparse.mm(self.norm_adj_matrix, self.K_list[layer_idx-1])
            # self.C_list[layer_idx] = torch.sparse.mm(self.cate_adj_matrix, self.C_list[layer_idx - 1])
            self.c_embeddings_list[layer_idx] = torch.sparse.mm(contrast_view, self.c_embeddings_list[layer_idx-1])
            self.embeddings_list[layer_idx] = self.K_list[layer_idx]

        lightgcn_all_embeddings = torch.stack(self.embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        c_lightgcn_all_embeddings = torch.stack(self.c_embeddings_list, dim=1)
        c_lightgcn_all_embeddings = torch.mean(c_lightgcn_all_embeddings, dim=1)

        # cate_lightgcn_all_embeddings = torch.stack(self.C_list, dim=1)
        # cate_lightgcn_all_embeddings = torch.mean(cate_lightgcn_all_embeddings, dim=1)


        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings,
                                                               [self.n_users, self.n_items])

        c_user_all_embeddings, c_item_all_embeddings = torch.split(c_lightgcn_all_embeddings,
                                                               [self.n_users, self.n_items])

        # cate_user_all_embeddings, cate_item_all_embeddings = torch.split(cate_lightgcn_all_embeddings,
        #                                                            [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings, c_user_all_embeddings, c_item_all_embeddings

    def calculate_loss(self, interaction, contrast_view):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # 提取用户、正样本和负样本
        user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
        iids = torch.cat([pos_item, neg_item], dim=0)

        # 将数据移动到设备（如 GPU）
        user = user.to(self.args.device)
        pos_item = pos_item.to(self.args.device)
        neg_item = neg_item.to(self.args.device)

        # 获取用户和物品的嵌入
        user_all_embeddings, item_all_embeddings, c_user_all_embeddings, c_item_all_embeddings = self.forward(
            contrast_view)

        # 提取特定用户和物品的嵌入
        u_embeddings = user_all_embeddings[user]
        i_emb = item_all_embeddings[iids]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        c_pos_embeddings = c_item_all_embeddings[pos_item]
        c_neg_embeddings = c_item_all_embeddings[neg_item]
        c_u_embeddings = c_user_all_embeddings[user]
        c_i_emb = c_item_all_embeddings[iids]

        # 温度系数
        temperature = 0.5

        # 计算负样本对的得分（负样本对比损失）
        def compute_neg_score(emb1, emb2, temperature):
            logits = torch.matmul(emb1, emb2.T) / temperature
            exp_logits = torch.exp(logits)
            return torch.log(exp_logits.sum(dim=1) + 1e-8).mean()

        neg_score = (
                compute_neg_score(c_u_embeddings, user_all_embeddings, temperature) +
                compute_neg_score(u_embeddings, c_user_all_embeddings, temperature) +
                compute_neg_score(c_i_emb, item_all_embeddings, temperature) +
                compute_neg_score(i_emb, c_item_all_embeddings, temperature)
        )

        # 计算正样本对的得分（正样本对比损失）
        def compute_pos_score(emb1, emb2, temperature):
            logits = torch.sum(emb1 * emb2, dim=1) / temperature
            return torch.clamp(logits, -5.0, 5.0).mean()

        pos_score = (
                compute_pos_score(c_u_embeddings, u_embeddings, temperature) +
                compute_pos_score(c_i_emb, i_emb, temperature) +
                compute_pos_score(u_embeddings, c_pos_embeddings, temperature) +
                compute_pos_score(c_u_embeddings, pos_embeddings, temperature)
        )

        # 总对比损失
        loss_s = -pos_score + neg_score

        # user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
        # iids = torch.concat([pos_item, neg_item], dim=0)
        # user = user.to(self.args.device)
        # pos_item = pos_item.to(self.args.device)
        u = loss_s * self.c_weight
        # neg_item = neg_item.to(self.args.device)
        # user_all_embeddings, item_all_embeddings, c_user_all_embeddings, c_item_all_embeddings = self.forward(contrast_view)
        # u_embeddings = user_all_embeddings[user]
        # i_emb = item_all_embeddings[iids]
        # pos_embeddings = item_all_embeddings[pos_item]
        # neg_embeddings = item_all_embeddings[neg_item]
        # c_pos_embeddings = c_item_all_embeddings[pos_item]
        # c_neg_embeddings = c_item_all_embeddings[neg_item]
        # c_u_embeddings = c_user_all_embeddings[user]
        # c_i_emb =c_item_all_embeddings[iids]
        #
        # neg_score = torch.log(torch.exp(c_u_embeddings @ user_all_embeddings.T / 0.5).sum(1) + 1e-8).mean() + torch.log(torch.exp(u_embeddings @ c_user_all_embeddings.T / 0.5).sum(1) + 1e-8).mean()
        #
        # neg_score += torch.log(torch.exp(c_i_emb @ item_all_embeddings.T / 0.5).sum(1) + 1e-8).mean() + torch.log(torch.exp(i_emb @ c_item_all_embeddings.T / 0.5).sum(1) + 1e-8).mean()
        # pos_score = (torch.clamp((c_u_embeddings * u_embeddings).sum(1) / 0.5, -5.0, 5.0)).mean() + (torch.clamp((c_i_emb * i_emb).sum(1) / 0.5, -5.0, 5.0)).mean()
        #
        # pos_score += torch.log(torch.exp(u_embeddings @ c_pos_embeddings.T / 0.5).sum(1) + 1e-8).mean() + torch.log(torch.exp(c_u_embeddings @ pos_embeddings.T / 0.5).sum(1) + 1e-8).mean()
        #
        # loss_s = -pos_score + neg_score

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        c_pos_scores = torch.mul(c_u_embeddings, c_pos_embeddings).sum(dim=1)
        c_neg_scores = torch.mul(c_u_embeddings, c_neg_embeddings).sum(dim=1)
        mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores)) + 0.5*torch.mean(torch.nn.functional.softplus(c_neg_scores - c_pos_scores)) # Light
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = (1 / 2) * (u_ego_embeddings.norm(2).pow(2) +
                              pos_ego_embeddings.norm(2).pow(2) +
                              neg_ego_embeddings.norm(2).pow(2)) / float(len(user))  # LightGCN source code used

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, user, contrast_view):
        """
        :param user: the id of batch users
        # :return:
        # """
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, self.c_restore_user_e, self.c_restore_item_e = self.forward(contrast_view)
        u_embeddings = self.restore_user_e[user]
        c_u_embeddings = self.c_restore_user_e[user]
        # cate_u_embeddings = self.cate_restore_user_e[user]
        # u_embeddings = torch.cat()
        # dot with all item embedding to accelerate
        scores1 = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        scores2 = torch.matmul(c_u_embeddings, self.c_restore_item_e.transpose(0, 1))
        scores = scores2 + scores1
        return scores



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

        # norm adj matrix`
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



