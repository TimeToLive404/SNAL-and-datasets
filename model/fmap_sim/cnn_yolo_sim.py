import os
from utils import FmapAnalysis, show_fmap
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# 将mycnn的特征图读入
cnn_fmap_list = {}
for file in os.listdir(r'D:\save_for_mycnn\cnn_np_save'):
    # print(file)
    deep = file.split(sep='-')
    # print(deep[2])
    mycnn_fmap = np.load(r'D:\save_for_mycnn\cnn_np_save\\' + file)
    cnn_fmap_list[deep[2]] = mycnn_fmap

# print(cnn_fmap_list['0'].shape)

# 将myyolo的特征图读入
yolo_fmap_list = {}
for file in os.listdir(r'D:\save_for_myyolo\np_save'):
    deep = file.split(sep='-')
    myyolo_fmap = np.load(r'D:\save_for_myyolo\np_save\\' + file)
    # print(len(myyolo_fmap))
    yolo_fmap_list[deep[2]] = myyolo_fmap

# plt.title('fmap')
print('如下的列表总共有：', len(cnn_fmap_list))


# show_fmap(feature=yolo_fmap_list[0])

def all_layer_sim_analysis(cnn_maps):
    deep_res = {}
    for key in cnn_maps:
        cnn_featuremap = cnn_fmap_list[key].squeeze(1)
        yolo_featuremap = yolo_fmap_list[key].squeeze(1)
        print(f'第{key}层', cnn_featuremap.shape, yolo_featuremap.shape)
        sim_analysis = FmapAnalysis(cnn_featuremap, yolo_featuremap, int(key) + 1)
        result = sim_analysis.sim_dis()
        deep_res[key] = result
        print(result)
    y = []
    for d in range(len(deep_res)):
        y.append(deep_res[str(d)])
    x = [i for i in range(len(y))]
    plt.plot(x, y)
    plt.xlabel('Deep')
    plt.ylabel('Proportion of similar feature map')
    # plt.title('multi')
    plt.show()


all_layer_sim_analysis(cnn_fmap_list)

# 进行相似性计算
# cnn_featuremap = cnn_fmap_list['0'].squeeze(1)
# yolo_featuremap = yolo_fmap_list['0'].squeeze(1)
# print(cnn_featuremap.shape, yolo_featuremap.shape)
# sim_analysis = FmapAnalysis(cnn_featuremap, yolo_featuremap, 9)
# result = sim_analysis.sim_dis()
# print(result)
#
# cnn_featuremap = cnn_fmap_list['25'].squeeze(1)
# yolo_featuremap = yolo_fmap_list['25'].squeeze(1)
# print(cnn_featuremap.shape, yolo_featuremap.shape)
# sim_analysis = FmapAnalysis(yolo_featuremap, cnn_featuremap, 8)
# result = sim_analysis.sim_dis()
# print(result)
