# import pandas as pd
# #
# # # 读取数据
# # data = pd.read_csv('edges_with_ids.data', header=None, sep=' ')
# #
# # # 随机选择20%的数据作为测试集
# # test_data = data.sample(frac=0.2)
# #
# # # 剩余的数据作为训练集
# # train_data = data.drop(test_data.index)
# #
# # # 保存训练集和测试集，使用空格作为分隔符
# # train_data.to_csv('train_data.txt', header=False, index=False, sep=' ')
# # test_data.to_csv('test_data.txt', header=False, index=False, sep=' ')
# train_data = {}
# m = set()
# data = pd.read_csv('test_data.txt', header=None, sep=' ')
# print(data[0])
# for index, row in data.iterrows():
#     # 访问每行的第一个元素
#     if row[0] not in m:
#         train_data[row[0]] = [row[1]]
#         m.add(row[0])
#     else:
#         train_data[row[0]].append(row[1])
# print(train_data)
# for key, value in train_data.items():
#     train_data[key] = ' '.join(map(str, value))
# with open('test.txt', 'w') as file:
#     for key, value in train_data.items():
#         file.write(f"{key}  {value}\n")
# 打开文件并读取数据
with open("edges_with_ids.data", "r") as file:
    lines = file.readlines()

# 处理数据，添加权重1
processed_lines = [line.strip() + " 1" for line in lines]

# 将处理后的数据写回文件
with open("data_with_weights.data", "w") as file:
    file.write("\n".join(processed_lines))
