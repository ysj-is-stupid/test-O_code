from tqdm import tqdm
import torch.optim as optim
import torch
# from tensorboardX import SummaryWriter
import os
import numpy as np

from utils import set_color


class Trainer(object):

    def __init__(self, args, model, generator):
        self.n_users = 6298
        self.n_items = 1609
        self.args = args
        self.model = model
        self.generator = generator
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.optimizer = self._build_optimizer(name=args.optimizer, params=self.model.parameters())
        self.generator_optimizer = self._build_optimizer(name=args.optimizer, params=self.generator.parameters())

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

    def train_an_epoch(self, train_data, epoch_id):
        self.model.train()
        self.generator.train()
        total_loss = 0
        iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_id:>5}", 'pink'))
        contrast = self.generator.forward()
        contrast = contrast.to(self.args.device)
        for batch_id, interaction in enumerate(iter_data):
            user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
            iids = torch.concat([pos_item, neg_item], dim=0)
            users = user.to(self.args.device)
            pos_item = pos_item.to(self.args.device)
            neg_item = neg_item.to(self.args.device)
            self.optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            # out1 = self.model.light_forward(self.model.norm_adj_matrix)
            # out2 = self.model.light_forward(contrast)

            out1 = self.model.light_forward(self.model.norm_adj_matrix)  # norm_adj_matrix 是稀疏张量
            out2 = self.model.light_forward(contrast)  # 将 contrast 转换为密集张量

            # 分割用户和物品的嵌入
            user_all_embeddings, item_all_embeddings = torch.split(out1, [self.n_users, self.n_items])
            c_user_all_embeddings, c_item_all_embeddings = torch.split(out2, [self.n_users, self.n_items])

            # 获取特定用户和物品的嵌入
            u_embeddings = user_all_embeddings[users]
            i_emb = item_all_embeddings[iids]
            pos_embeddings = item_all_embeddings[pos_item]
            neg_embeddings = item_all_embeddings[neg_item]

            c_pos_embeddings = c_item_all_embeddings[pos_item]
            c_neg_embeddings = c_item_all_embeddings[neg_item]
            c_u_embeddings = c_user_all_embeddings[users]
            c_i_emb = c_item_all_embeddings[iids]

            # 计算对比损失
            x = (u_embeddings @ c_user_all_embeddings.T / 0.5).sum(1).mean()
            x = torch.log(x)
            y = (i_emb @ c_item_all_embeddings.T / 0.5).sum(1).mean()
            y = torch.log(y)
            neg_score = x + y

            x = (u_embeddings @ c_u_embeddings.T / 0.5).sum(1).mean()
            x = torch.log(x)
            y = (i_emb @ c_i_emb.T / 0.5).sum(1).mean()
            y = torch.log(y)
            pos_score = x + y

            # 计算总损失
            loss_s = -pos_score + neg_score
            loss = loss_s

            # 反向传播
            loss.backward(retain_graph=True)



            # user_all_embeddings, item_all_embeddings = torch.split(out1, [self.n_users, self.n_items])
            # c_user_all_embeddings, c_item_all_embeddings = torch.split(out2, [self.n_users, self.n_items])
            # u_embeddings = user_all_embeddings[users]
            # i_emb = item_all_embeddings[iids]
            # pos_embeddings = item_all_embeddings[pos_item]
            # neg_embeddings = item_all_embeddings[neg_item]
            # c_pos_embeddings = c_item_all_embeddings[pos_item]
            # c_neg_embeddings = c_item_all_embeddings[neg_item]
            # c_u_embeddings = c_user_all_embeddings[users]
            # c_i_emb = c_item_all_embeddings[iids]


            # calculate BPR Loss
            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

            c_pos_scores = torch.mul(c_u_embeddings, c_pos_embeddings).sum(dim=1)
            c_neg_scores = torch.mul(c_u_embeddings, c_neg_embeddings).sum(dim=1)
            # mf_loss = -torch.log(self.gamma + torch.sigmoid(pos_scores - neg_scores)).mean()  # recbole used
            mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores)) + 0.5*torch.mean(torch.nn.functional.softplus(c_neg_scores - c_pos_scores))      # Light
            # GCN source code used
            loss = mf_loss
            loss.backward()
            loss_reg = 0
            for param in self.model.parameters():
                loss_reg += param.norm(2).square()
            # loss_reg *= self.lambda_2
            # loss = mf_loss + 1e-5 * loss_reg + 0.2*loss_s
            loss = loss_reg
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # x = (u_embeddings @ c_user_all_embeddings.T / 0.5).sum(1).mean()
            # x = torch.log(x)
            # y = (i_emb @ c_item_all_embeddings.T / 0.5).sum(1).mean()
            # y = torch.log(y)
            # neg_score = x + y
            # x = (u_embeddings @ c_u_embeddings.T / 0.5).sum(1).mean()
            # x = torch.log(x)
            # y = (i_emb @ c_i_emb.T / 0.5).sum(1).mean()
            # y = torch.log(y)
            # pos_score = x + y
            # loss_s = -pos_score + neg_score
            # loss = loss_s
            # loss.backward()
            # loss.backward()
            loss_1 = self.generator(self.model.norm_adj_matrix.cuda(), user, pos_item, neg_item)
            loss_1.backward()
            self.generator_optimizer.step()

    def evaluate(self, test_data, ground_true_items, mask_index, interaction_matrix, epoch_id):
        topk = self.args.topk
        a = 0.5
        pred_list = []  # predicted item listssh -p 15324 root@region-41.seetacloud.com
        ground_true_list = []  # ture item list
        self.model.eval()
        with torch.no_grad():
            iter_data = tqdm(test_data, total=len(test_data), ncols=100, desc=set_color(f"Evaluate   ", 'pink'))
            for batch_idx, batch_users in enumerate(iter_data):
                batch_users = batch_users.to(self.args.device)
                contrast = self.generator.forward().cuda()
                out1 = self.model.light_forward(self.model.norm_adj_matrix)
                out2 = self.model.light_forward(contrast)
                user_all_embeddings, item_all_embeddings = torch.split(out1, [self.n_users, self.n_items])
                c_user_all_embeddings, c_item_all_embeddings = torch.split(out2, [self.n_users, self.n_items])

                u_embeddings = user_all_embeddings[batch_users[0]]
                c_u_embeddings = c_user_all_embeddings[batch_users[0]]

                scores1 = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1)).cpu()
                scores2 = torch.matmul(c_u_embeddings, c_item_all_embeddings.transpose(0, 1)).cpu()
                scores = scores1 + scores2
                # scores1 = self.model.predict(batch_users).cpu()  # batch_user * n_items
                # scores2 = self.cate_model.predict(batch_users).cpu()
                # scores = scores1
                batch_users = batch_users[0]
                index = 0
                for i in batch_users:
                    # mask_item = mask_index[key]
                    key = int(i.detach())
                    mask_item = mask_index[key]
                    map_key = key % self.args.test_batch_size
                    scores[index][mask_item] = -np.inf
                    index += 1
                _, pred = torch.topk(scores, k=topk)
                pred_list.append(pred.numpy().tolist())
                pred_list = pred_list[0]
        X = zip(pred_list, ground_true_list)
        Recall, Precision, NDCG = 0, 0, 0
        i = 0
        for item in ground_true_items:
            if item in pred_list[i]:
                rank = pred_list[i].index(item)
                dcg = 1 / np.log2(rank + 2)  # 计算 DCG
                # idcg = 1 / np.log2(min(rank + 2, topk))  # 计算 IDCG
                NDCG += dcg
                # j += 1
                Recall += 1
                Precision += 1
            i += 1
        Precision /= i
        Recall /= i
        NDCG /= i
        F1_score = 2 * (Precision * Recall) / (Precision + Recall)
        print("Recall: {:.4f}, NDCG: {:.4f}".format(Recall, NDCG))

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
        idcg[idcg == 0] = 1

        # calculate ndcg
        ndcg = dcg / idcg
        return np.sum(ndcg)
