import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def color_mapping(predictions):  # 将输出结果映射为颜色值
    c_list = []
    # max_pre = predictions.sort()[0]
    for prediction in predictions:
        color = float(prediction)
        c_list.append(color)
    return c_list


def get_pre(file):  # 从保存的结果中提取预测值，保存为列表
    pre_file = open(file)
    lines = pre_file.readlines()
    pre = []
    for line in lines:
        datas = line.split(' ')
        pre.append(datas[-1].rstrip(''))
    return pre


circle_path_file = open('D:/SNAL/data/xy.txt', 'r')
x_ys = circle_path_file.readlines()
x_circle = []
y_circle = []
print(len(x_ys))
print(x_ys)
for x_y in x_ys:
    x_y_str = x_y.split(' ')
    try:
        x = int(x_y_str[0])
        y = int(x_y_str[1])
        x_circle.append(x)
        y_circle.append(y)
    except:
        continue
# print(x_circle, y_circle)
# plt.plot(x_circle, y_circle, 'ro')  # 'r',
# plt.show()
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False

# # 测试集标签图
# plt.subplot(2, 3, 1)
# label = [1 for _ in range(199)]
# label[0:21] = [0 for _ in range(21)]
# label[115:138] = [0 for _ in range(23)]
# label[197:199] = [0 for _ in range(3)]
# color = color_mapping(predictions=label)
# plt.scatter(x_circle, y_circle, s=100, c=color, cmap='PuBu_r')  #
# cb = plt.colorbar()
# cb.set_label('Anomaly degree')  # 设置colorbar的标签
# plt.gca().invert_yaxis()
# plt.title('LABEL')

# SNAL最佳预测
# plt.subplot(2, 3, 2)
prediction = get_pre('D:/SNAL/result/log_min.txt')
color = color_mapping(predictions=prediction)
plt.scatter(x_circle, y_circle, s=100, c=color, cmap='tab10')  # CMRmap Paired Spectral_r brg_r gist_earth gist_ncar tab10 terrain
cb = plt.colorbar()
cb.set_label('Potential')  # 设置colorbar的标签
plt.gca().invert_yaxis()
plt.title('Interneuron')

plt.show()