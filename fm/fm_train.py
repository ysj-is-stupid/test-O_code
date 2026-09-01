'''
pytorch实现因子分解机模型
author: huangjunheng
date: 2020/8/6
'''
from utils import set_color
from DataHandler import DataHandler
from data_utils import create_dataset, create_dataloader
from utils import init_seed
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

learning_rate = 1e-3
weight_decay = 1e-4
epochs = 100
batch_szie = 1024
min_val, max_val = 1.0, 5.0
device = torch.device('cuda:0')
embedding_dim = 128  # id嵌入向量的长度
import torch
import torch.nn as nn
import torch.optim as optim

file_path = 'mashup_relat_emb.pth'

    # 使用torch.load加载.pth文件
m_data = torch.load(file_path)

file_path = 'api_relat_emb.pth'

    # 使用torch.load加载.pth文件
a_data = torch.load(file_path)
features = torch.cat((m_data, a_data), dim=0)

# 创建模型
num_features = m_data.shape[1] + a_data.shape[1]

class FMModel(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(FMModel, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # 第一层是输入特征的线性层
        # 确保输入特征的维度与线性层的输入维度相匹配
        self.linear = nn.Linear(num_features, 1)

        # 第二层是特征嵌入层
        self.embedding = nn.Embedding(num_features, embedding_dim)

    def forward(self, x):
        # 线性部分
        linear_output = self.linear(x)
        x = x.long()

        # 嵌入部分
        embeddings = self.embedding(x)
        sum_square = torch.sum(embeddings ** 2, dim=1)
        square_sum = torch.sum(embeddings, dim=1) ** 2
        cross_term = 0.5 * (sum_square - square_sum)
        cross_term = torch.sum(cross_term, dim=1, keepdim=True)
        # sum_square = torch.sum(embeddings ** 2, dim=1)  # 每个样本的嵌入向量的平方和
        # square_sum = torch.sum(embeddings, dim=1) ** 2  # 所有嵌入向量和的平方
        # cross_term = 0.5 * (sum_square - square_sum)  # 二阶交互项

        # FM模型输出
        output = linear_output + cross_term
        return output


def val_iter(model, data_loader):
    model.eval()
    num = data_loader.dataset.__len__()
    labels, predicts = list(), list()
    Recall = 0
    NDCG = 0
    n_api = a_data.shape[0]  # API 数量，替代写死的 1608
    chunk = 512  # 每次前向的行数，控制显存峰值
    with torch.no_grad():  # 验证不需要梯度，否则计算图逐用户累积会把显存耗尽
        for usr, trnMask in data_loader:
            usr = usr.long().to(device)
            l = []
            for u in usr:
                replicated_tensor = m_data[u].repeat(n_api, 1)
                x = torch.cat((replicated_tensor, a_data), dim=1)
                empty_tensor = torch.cat([model(x[s:s + chunk]) for s in range(0, n_api, chunk)])
                l.append(empty_tensor)
            all_preds = torch.stack(l).squeeze(-1)
            _, topLocs = torch.topk(all_preds, 10)
            recall, ndcg = calcRes(topLocs.cpu().numpy(), data_loader.dataset.tstLocs, usr)
            Recall += recall
            NDCG += ndcg
    Recall /= num
    NDCG /= num
    return Recall, NDCG

# def calcRes(topLocs, tstLocs, users):
#     recall = 0.0
#     ndcg = 0.0
#     num_users = len(users)
#
#     for i, user in enumerate(users):
#         # 获取当前用户的真实正样本位置
#         true_positives = tstLocs[i]
#         # 获取当前用户的预测top-k位置
#         predicted_positives = topLocs[i]
#
#         # 计算召回率
#         recall += len(set(true_positives).intersection(set(predicted_positives))) / len(true_positives)
#
#         # 计算DCG
#         # dcg = 0.0
#         # for idx, loc in enumerate(predicted_positives):
#         #     if loc in true_positives:
#         #         dcg += 1 / np.log2(idx + 2)
#         # # 计算IDCG
#         # idcg = 0.0
#         # for idx, loc in enumerate(true_positives):
#         #     idcg += 1 / np.log2(idx + 2)
#         # # 计算NDCG
#         # if idcg > 0:
#         #     ndcg += dcg / idcg
#
#     recall /= num_users
#     # ndcg /= num_users
#
#     return recall

def calcRes(topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, 10))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

model = FMModel(num_features, embedding_dim).to(device)
handler = DataHandler()
handler.LoadData()
train_data = handler.trnLoader
test_data = handler.tstLoader
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.BCELoss()  # 使用二元交叉熵损失函数

for epoch in range(epochs):
    optimizer.zero_grad()
    model.train()
    total_loss = 0
    total_len = 0
    iter_data = tqdm(train_data, total=len(train_data), ncols=100)

    for batch_id, interaction in enumerate(iter_data):
        x_u, x_i, x_i_n = interaction[0].long().to(device), interaction[1].long().to(device), interaction[2].long().to(
            device)
        # y = (y - min_val) / (max_val - min_val) + 0.01
        pos_scores = model(torch.cat((m_data[x_u], a_data[x_i]), dim=1))
        neg_scores = model(torch.cat((m_data[x_u], a_data[x_i_n]), dim=1))
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))  # mf_loss

        # loss = criterion(y.view(-1, 1), y_pre)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x_u)
        total_len += len(x_u)

    loss = total_loss / total_len
    print(loss)
    recall, NDCG = val_iter(model, test_data)
    print("recall:" + str(recall))
    print('NDCG:' + str(NDCG))
