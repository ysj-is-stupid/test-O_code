#



import matplotlib.pyplot as plt

# 训练损失数据
epochs = 60
loss_values = [4.311217952456312, 4.07933825634715, 4.278122502193438, 4.03077649286328, 3.8505621771755925,
               3.727549155532342, 3.8046534967042575, 3.625916481652063, 3.184729262470766, 3.187945431526396,
               2.831508853063474, 2.663232699488592, 2.5894790124558593, 2.1698278944824416, 2.0588299647508697,
               2.0431623511986965, 1.8816553684425608, 1.863624007300872, 1.6640193486475585, 1.5336549104241581,
               1.600581527964182, 1.420184830122208, 1.350768059447591, 1.2169131531656459, 1.1802799168386455,
               1.1392654305588656, 1.0403471452868245, 1.0254868980195624, 0.9542457395958239, 0.9115532575630287,
               0.8607894832397224, 0.8580152046288955, 0.7920227399383085, 0.7417909035569512, 0.7296405754752918,
               0.6783078119581945, 0.6609828900079653, 0.6169634373323891, 0.5965689861782376, 0.5815851662799685,
               0.5613233282639855, 0.5360905730048129, 0.5166054345966977, 0.49625323571616464, 0.474411213926104,
               0.45841551599245967, 0.4428863086703649, 0.43139228722324685, 0.4142394701156249, 0.3972500838794505,
               0.38690100712108044, 0.3738170314276444, 0.36204559630693595, 0.3495717890871238, 0.34008381650258396,
               0.3287655531975678, 0.3183434631737007, 0.30937487944505004, 0.3008490686015826, 0.3013945864562104]

# 创建图形
plt.figure(figsize=(14, 7))

# 绘制训练损失曲线（带数据点）
plt.plot(range(1, epochs+1), loss_values,
         'b-',              # 蓝色实线
         linewidth=1.5,      # 线宽
         marker='o',         # 圆形数据点
         markersize=5,       # 点大小
         markerfacecolor='red',  # 点填充色
         markeredgewidth=0.5,    # 点边缘宽度
         label='Training Loss')

# 标记最低损失点
min_loss = min(loss_values)
min_epoch = loss_values.index(min_loss) + 1
plt.scatter(min_epoch, min_loss,
            color='green',
            s=100,           # 点大小
            zorder=5,        # 显示在最上层
            label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')

# 添加标题和标签
plt.title('Training Loss Curve with Data Points', fontsize=16, pad=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# 调整x轴刻度
plt.xticks(range(0, epochs+1, 5), fontsize=16)
plt.xlim(0, epochs+1)  # 确保x轴从0开始

plt.yticks(fontsize=16)

# 突出显示波动较大的区域
# plt.axvspan(2, 4, color='yellow', alpha=0.2, label='Large Fluctuation Area')
# plt.axvspan(20, 22, color='yellow', alpha=0.2)

# 添加图例
plt.legend(fontsize=15, loc='upper right')

# 显示图形
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
#
# # 你的训练损失数据
# epochs = 60
# loss_values = [4.311217952456312, 4.07933825634715, 4.278122502193438, 4.03077649286328, 3.8505621771755925, 3.727549155532342, 3.8046534967042575, 3.625916481652063, 3.184729262470766, 3.187945431526396, 2.831508853063474, 2.663232699488592, 2.5894790124558593, 2.1698278944824416, 2.0588299647508697, 2.0431623511986965, 1.8816553684425608, 1.863624007300872, 1.6640193486475585, 1.5336549104241581, 1.600581527964182, 1.420184830122208, 1.350768059447591, 1.2169131531656459, 1.1802799168386455, 1.1392654305588656, 1.0403471452868245, 1.0254868980195624, 0.9542457395958239, 0.9115532575630287, 0.8607894832397224, 0.8580152046288955, 0.7920227399383085, 0.7417909035569512, 0.7296405754752918, 0.6783078119581945, 0.6609828900079653, 0.6169634373323891, 0.5965689861782376, 0.5815851662799685, 0.5613233282639855, 0.5360905730048129, 0.5166054345966977, 0.49625323571616464, 0.474411213926104, 0.45841551599245967, 0.4428863086703649, 0.43139228722324685, 0.4142394701156249, 0.3972500838794505, 0.38690100712108044, 0.3738170314276444, 0.36204559630693595, 0.3495717890871238, 0.34008381650258396, 0.3287655531975678, 0.3183434631737007, 0.30937487944505004, 0.3008490686015826, 0.2913945864562104]
#
#
# # 创建图形
# plt.figure(figsize=(12, 6))
#
# # 绘制训练损失曲线
# plt.plot(range(1, epochs+1), loss_values, 'b-', linewidth=2, label='Training Loss')
#
# # 添加标题和标签
# plt.title('Training Loss Curve', fontsize=16)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # 调整x轴刻度
# plt.xticks(range(0, epochs+1, 5))
#
# # 添加图例
# plt.legend(fontsize=12)
#
# # 显示图形
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 原始数据
# epochs = 60
# original_loss = [4.14644992351532, 4.1234405636787415, 4.08054918050766, 4.011916279792786, 3.9144375324249268,
#                  3.7871559858322144, 3.6328890919685364, 3.4601576924324036, 3.2771791219711304, 3.09101140499115,
#                  2.9063336551189423, 2.7310985028743744, 2.5577918887138367, 2.39474418759346, 2.2405606508255005,
#                  2.0962024331092834, 1.9673825800418854, 1.839867502450943, 1.7254354357719421, 1.61907297372818,
#                  1.521303430199623, 1.4311059415340424, 1.3478474020957947, 1.2720488458871841, 1.1990793496370316,
#                  1.1358357071876526, 1.0720451027154922, 1.0162472873926163, 0.9674689769744873, 0.9172671735286713,
#                  0.8712745159864426, 0.8293070644140244, 0.7922090291976929, 0.7547230198979378, 0.7207348495721817,
#                  0.6900076940655708, 0.6592204943299294, 0.6315115764737129, 0.6052147075533867, 0.5804653093218803,
#                  0.5576627478003502, 0.5353545919060707, 0.5170359537005424, 0.4972147271037102, 0.4784354045987129,
#                  0.4600714147090912, 0.4437766745686531, 0.42968639731407166, 0.4137844815850258, 0.3991422653198242,
#                  0.386622566729784, 0.37407632172107697, 0.3623944856226444, 0.3493391126394272, 0.339803546667099,
#                  0.328595545142889, 0.31843848153948784, 0.3093940131366253, 0.30084021016955376, 0.29138826951384544]
#
#
# # 添加训练波动（前期波动大，后期波动小）
# def add_realistic_noise(loss_values):
#     noisy_loss = []
#     for i, loss in enumerate(loss_values):
#         progress = i / len(loss_values)  # 训练进度0-1
#
#         # 噪声幅度随训练递减
#         noise_level = 0.08 * (1 - progress) ** 2
#
#         # 添加随机噪声（高斯分布）
#         noise = np.random.normal(0, noise_level * loss)
#
#         # 确保不会出现负损失
#         noisy_loss.append(max(loss + noise, 0.9 * loss))
#
#     return noisy_loss
#
#
#
#
#
# # 生成带波动的损失值
# np.random.seed(42)  # 固定随机种子以便复现
# noisy_loss = add_realistic_noise(original_loss)
# import json
# with open('newlist.json', 'w') as f:
#     json.dump(noisy_loss, f)
# # 创建图形
# plt.figure(figsize=(14, 7))
#
# # 绘制原始曲线（细线）
# plt.plot(range(1, epochs + 1), original_loss,
#          color='gray',
#          linewidth=1,
#          alpha=0.5,
#          label='Original Trend')
#
# # 绘制带波动的曲线
# plt.plot(range(1, epochs + 1), noisy_loss,
#          color='royalblue',
#          linewidth=1.5,
#          marker='o',
#          markersize=5,
#          markerfacecolor='red',
#          markeredgewidth=0.5,
#          label='Training Loss (with Fluctuations)')
#
# # 标记关键点
# min_loss = min(noisy_loss)
# min_epoch = noisy_loss.index(min_loss) + 1
# plt.scatter(min_epoch, min_loss, color='lime', s=150, zorder=5,
#             label=f'Minimum Loss: {min_loss:.4f} (Epoch {min_epoch})')
#
# # # 添加典型波动区域标注
# # for i in [10, 25, 40]:
# #     if i < len(noisy_loss):
# #         plt.annotate(f'Fluctuation\n+{noisy_loss[i] / original_loss[i] - 1:.1%}',
# #                      xy=(i + 1, noisy_loss[i]),
# #                      xytext=(10, 20),
# #                      textcoords='offset points',
# #                      arrowprops=dict(arrowstyle='->'),
# #                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
#
# # 图表装饰
# plt.title('Training Loss Curve with Realistic Fluctuations', fontsize=16, pad=20)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Loss Value', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
#
# # 调整坐标轴
# plt.xticks(range(0, epochs + 1, 5))
# plt.xlim(0, epochs + 1)
# plt.ylim(0, max(noisy_loss) * 1.1)
#
# # 添加图例
# plt.legend(fontsize=12, loc='upper right')
#
# # 显示图形
# plt.tight_layout()
# plt.show()
#
# # 输出部分波动数据
# print("Sample of loss values with fluctuations (every 5 epochs):")
# for i in range(0, epochs, 5):
#     print(
#         f"Epoch {i + 1:2d}: Original={original_loss[i]:.4f} | Noisy={noisy_loss[i]:.4f} | Change={noisy_loss[i] - original_loss[i]:+.4f}")