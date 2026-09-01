# # # # import pickle
# # # # import numpy as np
# # # # from scipy.sparse import lil_matrix
# # # #
# # # # # 读取.txt文件
# # # # with open('./data/api/train.txt', 'r') as txt_file:
# # # #     lines = txt_file.readlines()
# # # #
# # # # # 获取mashup和API的数量
# # # # num_mashups = len(lines)
# # # # num_apis = max(int(api) for line in lines for api in line.split()[1:]) + 1  # 计算API的最大编号
# # # #
# # # # # 创建一个空的稀疏矩阵
# # # # sparse_matrix = lil_matrix((num_mashups, num_apis), dtype=np.int8)
# # # #
# # # # # 填充稀疏矩阵
# # # # for mashup_id, line in enumerate(lines):
# # # #     api_indices = [int(api) for api in line.split()[1:]]
# # # #     sparse_matrix[mashup_id, api_indices] = 1
# # # #
# # # # # 保存稀疏矩阵为.pkl文件
# # # # with open('sparse_matrix.pkl', 'wb') as pkl_file:
# # # #     pickle.dump(sparse_matrix, pkl_file)
# # # #
# # # # print("成功将.txt文件转换为.pkl文件，保存为稀疏矩阵。")
# # # import pickle
# # #
# # # with open(r'C:\Users\24737\PycharmProjects\LightGCN-main\sparse_matrix.pkl', 'rb') as f:
# # #     data = pickle.load(f)
# # # print(data)
# # #
# # # import pickle
# # # import numpy as np
# # # from scipy.sparse import coo_matrix, hstack, vstack
# # #
# # # # 读取.pkl文件以获取原始的 n×m 稀疏矩阵
# # # with open(r'C:\Users\24737\PycharmProjects\LightGCN-main\sparse_matrix.pkl', 'rb') as f:
# # #     original_sparse_matrix = pickle.load(f)
# # #
# # # # 获取原始稀疏矩阵的形状
# # # n, m = original_sparse_matrix.shape
# # #
# # # # 创建两个零矩阵，一个大小为 n×n，另一个大小为 m×m
# # # zero_matrix_n = coo_matrix((n, n), dtype=np.int8)
# # # zero_matrix_m = coo_matrix((m, m), dtype=np.int8)
# # #
# # # # 将三个矩阵按行连接，创建 n+m × n+m 的稀疏矩阵
# # # extended_sparse_matrix = vstack([hstack([original_sparse_matrix, zero_matrix_n]),
# # #                                  hstack([zero_matrix_m, original_sparse_matrix.T])], format='csr')
# # #
# # # # 保存扩展后的稀疏矩阵为.pkl文件
# # # with open('extended_sparse_matrix.pkl', 'wb') as pkl_file:
# # #     pickle.dump(extended_sparse_matrix, pkl_file)
# # #
# # # print("成功将原始稀疏矩阵扩展为 n+m × n+m 的 CSR 矩阵并保存为.pkl文件。")
# # with open('../data/cate/test.txt') as f:
# #     with open('test_data.txt', 'w') as test_file:
# #         for line in f:
# #             x = line.split(' ')
# #             if len(x) > 1:
# #                 test_file.write(line)
#
#
#
# import matplotlib.pyplot as plt
#
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体作为默认字体以支持中文显示
# plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题
#
# # 示例数据：推荐列表长度
# recommendation_lengths = [5, 10, 15, 20, 25]  # 请根据实际情况调整
#
#
# # 示例数据：不同实验下的各个指标值，请用你的实际数据替换这些值
# metrics_data = {
#     'Recall': {
#         'NGCF': [0.3442,0.4178,0.4634,0.4960,0.5231],
#         'SRCLML': [0.3990,0.4415,0.4878,0.4966,0.5213],
#         'MTFM': [0.5476,0.6040,0.6458,0.6702,0.6920],
#         'Div_PreAPI': [0.5854,0.6133,0.6568,0.6764,0.7256],
#         '本文方法': [0.6079,0.6812,0.7334,0.7525,0.7630]
#     },
#     'Precision': {
#         'NGCF': [0.0746,0.0556,0.0389,0.0344,0.0217],
#         'SRCLML': [0.0822,0.0660,0.0515,0.0347,0.0215],
#         'MTFM': [0.1127,0.0903,0.0772,0.0468,0.0286],
#         'Div_PreAPI': [0.1202,0.1003,0.0799,0.0512,0.0290],
#         '本文方法': [0.1252,0.1019,0.0819,0.0526,0.0315]
#     },
#     'F1': {
#         'NGCF': [0.1091, 0.0969, 0.0623, 0.0295, 0.0213],
#         'SRCLML': [0.1363, 0.1148, 0.0721, 0.0383, 0.0326],
#         'MTFM': [0.1869, 0.1571, 0.1455, 0.1182, 0.1005],
#         'Div_PreAPI': [0.1920, 0.1694, 0.1495, 0.1266, 0.1023],
#         '本文方法': [0.2076, 0.1773, 0.1630, 0.1309, 0.1094]
#     },
#     'NDCG': {
#         'NGCF': [0.3216,0.3371,0.3660,0.3688,0.3709],
#         'SRCLML': [0.3288,0.3328,0.3569,0.3696,0.3720],
#         'MTFM': [0.3971,0.4019,0.4229,0.4344,0.4394],
#         'Div_PreAPI': [0.3855,0.4032,0.4350,0.4364,0.4369],
#         '本文方法': [0.3997,0.4159,0.4399,0.4474,0.4475]
#     }
# }
# markers = ['D', 's', '^', 'o', '*']  # 五种不同的标记形状
# # 绘制每个指标的折线图
# for metric_name, experiments in metrics_data.items():
#     plt.figure()  # 创建一个新的图形
#     for j, (exp_name, values) in enumerate(experiments.items()):
#         plt.plot(recommendation_lengths, values, marker=markers[j], label=exp_name,




import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体作为默认字体以支持中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 示例数据：推荐列表长度
recommendation_lengths = [5, 10, 15, 20, 25]

# 示例数据：不同实验下的各个指标值，请用你的实际数据替换这些值
metrics_data = {
    'Recall': {
        'NGCF': [0.3442,0.4178,0.4634,0.4960,0.5231],
        'SRCLML': [0.3990,0.4415,0.4878,0.4996,0.5255],
        'MTFM': [0.5476,0.6040,0.6458,0.6702,0.6920],
        'Div_PreAPI': [0.5854,0.6333,0.6768,0.7064,0.7256],
        '本文方法': [0.6079,0.6812,0.7334,0.7525,0.7630]
    },
    'Precision': {
        'NGCF': [0.0746,0.0556,0.0389,0.0344,0.0217],
        'SRCLML': [0.0822,0.0660,0.0515,0.0357,0.0227],
        'MTFM': [0.1047,0.0961,0.0852,0.0698,0.0566],
        'Div_PreAPI': [0.1202,0.1073,0.0949,0.0802,0.0650],
        '本文方法': [0.1292,0.1199,0.1083,0.0926,0.0795]
    },
    # 'F1': {
    #     'NGCF': [0.1091, 0.0969, 0.0623, 0.0295, 0.0213],
    #     'SRCLML': [0.1363, 0.1148, 0.0721, 0.0383, 0.0326],
    #     'MTFM': [0.1869, 0.1571, 0.1455, 0.1182, 0.1005],
    #     'Div_PreAPI': [0.1920, 0.1694, 0.1495, 0.1266, 0.1023],
    #     '本文方法': [0.2076, 0.1773, 0.1630, 0.1309, 0.1094]
    # },
    'F1':{
'NGCF': [0.1226, 0.0981, 0.0719, 0.0644, 0.0415],
'SRCLML': [0.1363, 0.115, 0.0932, 0.0661, 0.0423],
'MTFM': [0.1758, 0.1659, 0.1506, 0.1263, 0.1047],
'Div_PreAPI': [0.1999, 0.1835, 0.1664, 0.144, 0.1194],
'本文方法': [0.213, 0.2036, 0.1883, 0.1651, 0.144]
    },

    'NDCG': {
        'NGCF': [0.3316,0.3471,0.3610,0.3788,0.3839],
        'SRCLML': [0.3488,0.3598,0.3769,0.3870,0.3950],
        'MTFM': [0.3771,0.3919,0.4059,0.4144,0.4194],
        'Div_PreAPI': [0.3865,0.4072,0.4250,0.4364,0.4399],
        '本文方法': [0.3977,0.4169,0.4359,0.4434,0.4475]
    }
}
markers = ['D', 's', '^', 'o', '*']  # 五种不同的标记形状
# 绘制每个指标的折线图
for metric_name, experiments in metrics_data.items():
    plt.figure()  # 创建一个新的图形
    for j, (exp_name, values) in enumerate(experiments.items()):
        plt.plot(recommendation_lengths, values, marker=markers[j], label=exp_name, linestyle='-', markersize=6)
    # plt.title(f'{metric_name} Comparison across Experiments')
    plt.xlabel('推荐列表长度', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.legend(fontsize=10)  # 显示图例
    plt.xticks(ticks=recommendation_lengths, labels=recommendation_lengths, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()
