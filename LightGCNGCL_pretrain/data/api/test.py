# import pickle
# from scipy.sparse import csr_matrix
# import numpy as np
#
# # 读取文本文件并构建数据列表
# data = []
# row_ind = []
# col_ind = []
# with open('train.txt', 'r') as f:
#     for line in f:
#         parts = line.strip().split()
#         user_id = int(parts[0])
#         item_ids = list(map(int, parts[1:]))
#         for item_id in item_ids:
#             row_ind.append(user_id)
#             col_ind.append(item_id)
#             data.append(1)  # 或者根据需要使用其他值
#
# # 确定矩阵的大小，假设用户ID和物品ID是从0开始连续的
# num_users = max(row_ind) + 1
# num_items = max(col_ind) + 1
#
# # 创建CSR矩阵
# train_csr_data = csr_matrix((data, (row_ind, col_ind)), shape=(num_users, num_items))
#
#
# data = []
# row_ind = []
# col_ind = []
# with open('test.txt', 'r') as f:
#     for line in f:
#         parts = line.strip().split()
#         user_id = int(parts[0])
#         item_ids = list(map(int, parts[1:]))
#         for item_id in item_ids:
#             row_ind.append(user_id)
#             col_ind.append(item_id)
#             data.append(1)
# test_csr_data = csr_matrix((data, (row_ind, col_ind)), shape=(num_users, num_items))
#
# # 保存CSR矩阵到PKL文件
# with open('train_csr_matrix.pkl', 'wb') as f:
#     pickle.dump(train_csr_data, f)
#
# with open('test_csr_matrix.pkl', 'wb') as f:
#     pickle.dump(test_csr_data, f)
#

import pickle
#
# with open('train_csr_matrix.pkl', 'rb') as file:
#     data = pickle.load(file)
# print(data)
# print(data.shape)
#
# with open('test_csr_matrix.pkl', 'rb') as file:
#     data = pickle.load(file)
# print(data)
# print(data.shape)

#
# def find_max_itemid(filename):
#     max_itemid = -1
#     with open(filename, 'r') as file:
#         for line in file:
#             items = line.strip().split()
#             if len(items) > 1:  # 确保至少有一个itemid
#                 itemids = map(int, items[1:])  # 跳过userid，转换剩余的itemid为整数
#                 current_max = max(itemids)
#                 if current_max > max_itemid:
#                     max_itemid = current_max
#     return max_itemid
#
# # 假设文件名为'data.txt'
# max_itemid = find_max_itemid('train.txt')
# print(f"The maximum itemid is: {max_itemid}")
#

import json
with open(r'C:\Users\24737\PycharmProjects\LightGCNGCL_pretrain\src\newlist.json', 'r') as f:
    loaded_list = json.load(f)

print(loaded_list)