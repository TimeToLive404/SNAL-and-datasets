import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import cv2


class FmapAnalysis:
    def __init__(self, fmap1, fmap2, deep=1):  # famp为一层的全部特征图
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.deep = deep

    def mean_match(self, f1, f2):  # 两幅特征图进行整体均值平移匹配
        assert f1.ndim == 2 and f2.ndim == 2
        f1_sum = np.sum(f1)
        f2_sum = np.sum(f2)
        f1_mean = f1_sum / f1.size
        f2_mean = f2_sum / f2.size
        if ((f1_mean - f2_mean) < 0.5) and ((f1_mean - f2_mean) > 0):
            tmp_np = np.ones(f2.shape) * (f1_mean - f2_mean)
            f2 = f2 + tmp_np  # 使两者均值相同
            return f1, f2
        elif ((f1_mean - f2_mean) < 0.5) and ((f1_mean - f2_mean) < 0):
            tmp_np = np.ones(f1.shape) * (f2_mean - f1_mean)
            f1 = f1 + tmp_np  # 使两者均值相同
            return f1, f2
        else:
            return None, None

    def denoise(self, map, kernel_size=3, stride=3):  # 一幅特征图进行下采样去噪
        assert map.ndim == 2
        h, w = map.shape
        new_map = []
        # denoise_process=process_tqdm()
        for j in range(0, h, stride):
            # denoise_process.set_description(f'第{j}行：')
            for i in range(0, w, stride):
                if j + kernel_size <= h and i + kernel_size <= w:
                    tmp = np.sum(map[j:j + kernel_size, i:i + kernel_size]) / map.size  # 均值池化
                    new_map.append(tmp)
        map_array = np.array(new_map)
        map_array.reshape(((h - kernel_size) // stride + 1, -1))
        return map_array

    def size_match(self):  # 除以当前特征图面积，使数值相对化
        ...

    def sim_dis(self, min_threshold=5 * 0.0001 / 2):  # 求取一层的特征图相似性值
        # 均值匹配与过滤
        map1s = []
        map2s_map1s = []
        for map1 in self.fmap1:
            map2s = []
            for map2 in self.fmap2:
                if np.all(map1) == None or np.all(map2) == None:
                    continue
                map1_new, map2_new = self.mean_match(map1, map2)
                if np.any(map1):  # 如果这一对是可能相似的
                    map2s.append(map2_new)
            map1s.append(map1)
            map2s_map1s.append(map2s)

        # 下采样去噪
        map1s = [self.denoise(map) for map in process_tqdm(map1s) if not np.all(map) == None]
        new_map2s_map1s = []
        for maps in process_tqdm(map2s_map1s):
            map2s = [self.denoise(map) for map in maps if not np.all(map) == None]
            new_map2s_map1s.append(map2s)

        # 求特征图相似性
        if len(new_map2s_map1s[0]) == 0:
            print("两幅特征图过于不相似")
            return None
        layer_sim = []
        for map1, map2s in zip(map1s, new_map2s_map1s):
            map_sim = [1]  # 保证map_sim[0]存在且不影响排序
            for map2 in map2s:
                map_dis = map1 - map2
                map_dis = np.array(map_dis)
                map_dis = np.maximum(map_dis, -map_dis)  # 所有元素取绝对值
                sim_co = np.sum(map_dis) / map_dis.size  # 计算相似性系数
                map_sim.append(sim_co)  # 保存特征图2相对于特征图1的相似性系数   # self.deep
            map_sim.sort()
            layer_sim.append(map_sim[0])  # 保存各个特征图1中最小的系数
        sim_map_num = 0
        for co in layer_sim:
            if co < self.deep * min_threshold:  # 如果足够相似，就使相似性计数加一
                sim_map_num += 1
        prop = sim_map_num / len(map1s)  # 输出相似的特征图占全部图的百分比
        return prop

        # 测试代码,查看co的大小
        # co_sum = 0
        # for co in layer_sim:
        #     co_sum += co
        # p = co_sum / len(map1s)
        # return p


def show_fmap(feature, nrow=8, padding=10, pad_value=1):
    print('图的shape:', feature.shape)
    feature = torch.tensor(feature)
    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))
    title = 'fmap'
    plt.title(title)
    plt.imshow(images)
    plt.show()


def process_tqdm(obj, *args, **kwargs):
    try:
        return tqdm(obj, *args, **kwargs)
    except:
        return obj


def frame_handle(frame, input_dim):
    # 缩放到416*416
    frame = frame.reshape(1, 1080, 1920)
    # frame = frame.transpose((2, 1, 0))
    frame = frame.transpose((1, 2, 0))
    w, h = input_dim, input_dim
    new_w = int(frame.shape[1] * min(w / frame.shape[1], h / frame.shape[0]))
    new_h = int(frame.shape[0] * min(w / frame.shape[1], h / frame.shape[0]))
    new_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((input_dim, input_dim), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w] = new_frame
    # H*W*C->C*H*W->B*C*H*W
    # canvas = canvas[:, :, ::-1].transpose((2, 0, 1)).copy()  # 先转换成RGB格式，再把C放到第一个位置，为转换成tensor做准备
    canvas = torch.from_numpy(canvas).float().div(255.0).unsqueeze(0)  # 每个元素除以255以归一化,添加一个batch维
    return canvas
