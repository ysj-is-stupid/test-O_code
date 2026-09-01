# 读取原始文本文件并处理数据
processed_data = []
with open('newtest.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:  # 确保至少有一个物品ID
            processed_data.append(line)

# 将处理后的数据保存到新的文本文件
with open('newtest.txt', 'w') as f:
    for line in processed_data:
        f.write(line)

# 如果您想要覆盖原始文件，可以将文件名改为'output.txt'
# with open('output.txt', 'w') as f:
#     for line in processed_data:
#         f.write(line)
