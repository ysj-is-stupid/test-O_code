from copy import deepcopy

from tqdm import tqdm
import torch.optim as optim
import torch
# from tensorboardX import SummaryWriter
import os
import numpy as np

from utils import set_color


class Trainer(object):

    def __init__(self, args, model, generator):
        self.args = args
        self.model = model
        self.generator = generator
        # self.cate_model = cate_model
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.optimizer = self._build_optimizer(name=args.optimizer, params=self.model.parameters())
        # self.cate_optimizer = self._build_optimizer(name=args.optimizer, params=self.cate_model.parameters())

        # self._writer = SummaryWriter(log_dir=args.tensorboard_dir)
        self.NDCG_best = 0
        self.epoch_best = 0

    def _build_optimizer(self, name, params):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            print('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def generator_generate(self, generator, ppr_adj):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = ppr_adj.cuda()
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(adj, idxs)

        return view

    def train_an_epoch(self, epoch, train_data, ppr_adj, test_data, ground_true_items, mask_index, epoch_id):
        self.model.train()
        self.generator.train()
        total_loss = 0
        iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_id:>5}", 'pink'))
        contrast_view = self.generator_generate(self.generator, ppr_adj)
        for batch_id, interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(interaction, contrast_view)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            iter_data.set_postfix(Batch_Loss=loss.item())
        topk = self.args.topk
        a = 0.5
        pred_list = []  # predicted item listssh -p 15324 root@region-41.seetacloud.com
        ground_true_list = []  # ture item list
        self.model.eval()
        with torch.no_grad():
            iter_data = tqdm(test_data, total=len(test_data), ncols=100, desc=set_color(f"Evaluate   ", 'pink'))
            for batch_idx, batch_users in enumerate(iter_data):
                batch_users = batch_users.to(self.args.device)
                scores1 = self.model.predict(batch_users[0], contrast_view).cpu()  # batch_user * n_items
                scores = scores1
                batch_users = batch_users[0]
                index = 0
                for i in batch_users:
                    # key = int(i.detach())
                    # mask_item = mask_index[key]
                    key = int(i.detach())
                    # indices = torch.where(batch_users == i)
                    # indices = int(indices.detach())
                    mask_item = mask_index[key]
                    map_key = key % self.args.test_batch_size
                    scores[index][mask_item] = -np.inf
                    index += 1
                # train_adj_coo = interaction_matrix.tocoo()
                # train_rows, train_cols = train_adj_coo.row, train_adj_coo.col
                for i, user_id in enumerate(batch_users):
                    user_id = int(user_id.detach())
                    mask_item = mask_index[user_id]
                # user_row = train_adj_coo[user_id, :]
                # associated_item_indices = np.where(user_row > 0)[0]
                _, pred = torch.topk(scores, k=topk)
                pred_list.append(pred.numpy().tolist())
                pred_list = pred_list[0]
        Recall, Precision, NDCG = 0, 0, 0
        i = 0
        for item in ground_true_items:
            # maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
            if item in pred_list[i]:
                rank = pred_list[i].index(item)
                dcg = 1 / np.log2(rank + 2)  # 计算 DCG
                # idcg = 1 / np.log2(min(rank + 2, topk))  # 计算 IDCG
                NDCG += dcg
                # j += 1
                Precision += 1
                Recall += 1
                # Precision += 1
            i += 1
        Recall /= i
        NDCG /= i
        Precision /= len(pred_list[0]) * i
        F1_score = 2 * (Precision * Recall) / (Precision + Recall)
        print("Recall: {:.4f}, NDCG: {:.4f}, Pre: {:.4f}, F1: {}".format(Recall, NDCG, Precision, F1_score))
        return loss
    # def evaluate(self, interaction_matrix, , epoch_id):
    #     topk = self.args.topk
    #     a = 0.5
    #     pred_list = []  # predicted item listssh -p 15324 root@region-41.seetacloud.com
    #     ground_true_list = []  # ture item list
    #     self.model.eval()
    #     with torch.no_grad():
    #         iter_data = tqdm(test_data, total=len(test_data), ncols=100, desc=set_color(f"Evaluate   ", 'pink'))
    #         for batch_idx, batch_users in enumerate(iter_data):
    #             batch_users = batch_users.to(self.args.device)
    #             scores1 = self.model.predict(batch_users[0]).cpu()  # batch_user * n_items
    #             scores2 = self.cate_model.predict(batch_users[0]).cpu()
    #             scores = scores1
    #             batch_users = batch_users[0]
    #             index = 0
    #             for i in batch_users:
    #                 # key = int(i.detach())
    #                 # mask_item = mask_index[key]
    #                 key = int(i.detach())
    #                 # indices = torch.where(batch_users == i)
    #                 # indices = int(indices.detach())
    #                 mask_item = mask_index[key]
    #                 map_key = key % self.args.test_batch_size
    #                 scores[index][mask_item] = -np.inf
    #                 index += 1
    #             # train_adj_coo = interaction_matrix.tocoo()
    #             # train_rows, train_cols = train_adj_coo.row, train_adj_coo.col
    #             for i, user_id in enumerate(batch_users):
    #                 user_id = int(user_id.detach())
    #                 mask_item = mask_index[user_id]
    #                 # 确保i在scores的边界内，然后再访问它以避免“超出边界”错误
    #                 # if i < scores.shape[0]:
    #                 #     connected_items = train_cols[train_rows == user_id]
    #                 #     scores[i, connected_items] = -float("inf")
    #                 # else:
    #                     # 处理i超出边界的情况
    #                     # print(f"索引{i}超出了scores的边界。")
    #             # user_row = train_adj_coo[user_id, :]
    #             # associated_item_indices = np.where(user_row > 0)[0]
    #             _, pred = torch.topk(scores, k=topk)
    #             pred_list.append(pred.numpy().tolist())
    #             pred_list = pred_list[0]
    #             # list shape: batch_num* batch_user * k
    #             # ground_true_list.append([ground_true_items[int(u)] for u in batch_users])  # list
    #
    #     X = zip(pred_list, ground_true_list)
    #     Recall, Precision, NDCG = 0, 0, 0
    #     i = 0
    #     for item in ground_true_items:
    #         # maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
    #         if item in pred_list[i]:
    #             rank = pred_list[i].index(item)
    #             dcg = 1 / np.log2(rank + 2)  # 计算 DCG
    #             # idcg = 1 / np.log2(min(rank + 2, topk))  # 计算 IDCG
    #             NDCG += dcg
    #             # j += 1
    #             Recall += 1
    #             # Precision += 1
    #         i += 1
    #     # for i, x in enumerate(X):
    #     #     precision, recall, ndcg = self._evaluate_one_batch(x, topk)
    #     #     Recall += recall
    #     #     Precision += precision
    #     #     NDCG += ndcg
    #
    #     # n_user = self.model.n_users
    #     # Precision /= n_user
    #     Recall /= i
    #     NDCG /= i
    #     F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    #     # self._writer.add_scalar('model/NDCG', NDCG, epoch_id)
    #     # if NDCG > self.NDCG_best:
    #     #     self.NDCG_best = NDCGssh -p 41995 root@region-45.autodl.pro
    #     #     self.epoch_best = epoch_id
    #     print("Recall: {:.4f}, NDCG: {:.4f}, F1: {}".format(Recall, NDCG, F1_score))

    def _evaluate_one_batch(self, x, topk):
        pred_items = x[0]  # list: batch_user * k
        ground_true_items = x[1]  # list: batch_user * n (n is the num of ground true items)
        hit = []
        for i in range(len(pred_items)):
            ground_true = ground_true_items[i]
            pred = pred_items[i]
            pred_in_groundtrue = list(map(lambda x: x in ground_true, pred))  # [True, False, ...] len: topk
            pred_in_groundtrue = np.array(pred_in_groundtrue).astype('float')
            hit.append(pred_in_groundtrue)
        hit = np.array(hit).astype('float')  # np.array: batch_user * k
        precision = 0
        recall = 0
        ndcg = self._NDCG_AT_K(ground_true_items, hit, topk)

        return precision, recall, ndcg

    def _NDCG_AT_K(self, ground_true_items, hit, topk):
        assert len(ground_true_items) == len(hit)
        batch_users_num = len(hit)

        k = topk
        # calculate dcg
        dcg = hit * (1.0 / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(dcg, axis=1)

        # calculate idcg
        idcg_matrix = np.zeros((batch_users_num, k))
        for i, items in enumerate(ground_true_items):
            length = k if k <= len(items) else len(items)
            idcg_matrix[i, :length] = 1
        idcg = idcg_matrix * 1.0 / np.log2(np.arange(2, k + 2))
        idcg = np.sum(idcg, axis=1)

        # some test item is [], so dcg and idcg is zero.
        idcg[idcg==0] = 1

        # calculate ndcg
        ndcg = dcg / idcg
        return np.sum(ndcg)

