# # import pickle
# # import numpy as np
# # from scipy.sparse import lil_matrix
# #
# # # 读取.txt文件
# # with open('./data/api/train.txt', 'r') as txt_file:
# #     lines = txt_file.readlines()
# #
# # # 获取mashup和API的数量
# # num_mashups = len(lines)
# # num_apis = max(int(api) for line in lines for api in line.split()[1:]) + 1  # 计算API的最大编号
# #
# # # 创建一个空的稀疏矩阵
# # sparse_matrix = lil_matrix((num_mashups, num_apis), dtype=np.int8)
# #
# # # 填充稀疏矩阵
# # for mashup_id, line in enumerate(lines):
# #     api_indices = [int(api) for api in line.split()[1:]]
# #     sparse_matrix[mashup_id, api_indices] = 1
# #
# # # 保存稀疏矩阵为.pkl文件
# # with open('sparse_matrix.pkl', 'wb') as pkl_file:
# #     pickle.dump(sparse_matrix, pkl_file)
# #
# # print("成功将.txt文件转换为.pkl文件，保存为稀疏矩阵。")
# import pickle
#
# with open(r'C:\Users\24737\PycharmProjects\LightGCN-main\sparse_matrix.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data)
#
# import pickle
# import numpy as np
# from scipy.sparse import coo_matrix, hstack, vstack
#
# # 读取.pkl文件以获取原始的 n×m 稀疏矩阵
# with open(r'C:\Users\24737\PycharmProjects\LightGCN-main\sparse_matrix.pkl', 'rb') as f:
#     original_sparse_matrix = pickle.load(f)
#
# # 获取原始稀疏矩阵的形状
# n, m = original_sparse_matrix.shape
#
# # 创建两个零矩阵，一个大小为 n×n，另一个大小为 m×m
# zero_matrix_n = coo_matrix((n, n), dtype=np.int8)
# zero_matrix_m = coo_matrix((m, m), dtype=np.int8)
#
# # 将三个矩阵按行连接，创建 n+m × n+m 的稀疏矩阵
# extended_sparse_matrix = vstack([hstack([original_sparse_matrix, zero_matrix_n]),
#                                  hstack([zero_matrix_m, original_sparse_matrix.T])], format='csr')
#
# # 保存扩展后的稀疏矩阵为.pkl文件
# with open('extended_sparse_matrix.pkl', 'wb') as pkl_file:
#     pickle.dump(extended_sparse_matrix, pkl_file)
#
# print("成功将原始稀疏矩阵扩展为 n+m × n+m 的 CSR 矩阵并保存为.pkl文件。")
with open('../data/cate/test.txt') as f:
    with open('test_data.txt', 'w') as test_file:
        for line in f:
            x = line.split(' ')
            if len(x) > 1:
                test_file.write(line)