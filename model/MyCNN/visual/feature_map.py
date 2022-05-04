import matplotlib.pyplot as plt
import torchvision
import os
import keras
import numpy as np

deep_counter = 0


# 直接提取特征图
def get_feature_map(feature, num_ch=-1, nrow=8, padding=10, pad_value=1, save_feature=True, show_feature=False,
                    save_path=r'D:\MyCNN\visual\feature_map', np_save=True):
    '''

    :param feature: 要可视化的特征图
    :param num_ch: 可视化通道数量
    :param nrow: 画布上每行显示多少个特征图
    :param padding: 画布参数
    :param pad_value: 画布参数
    :param save_feature: 是否保存特征图文件
    :param show_feature: 是否显示提取的特征图
    :param save_path: 保存路径
    :return:
    '''
    global deep_counter
    b, c, h, w = feature.shape
    title = str(deep_counter) + '-' + str('hwc-') + str(h) + '-' + str(w) + '-' + str(c)
    feature = feature[0]
    feature = feature.unsqueeze(1)
    if c > num_ch > 0:
        feature = feature[:num_ch]
    if np_save:
        feature_copy = feature.detach().cpu()
        feature_save = feature_copy.numpy()
        np.save(f'D:/save_for_mycnn/cnn_np_save/mycnn-np-{title}.npy', feature_save)
    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))
    plt.title(title)
    plt.imshow(images)
    if show_feature:
        plt.show()
    if save_feature:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, title))
    deep_counter += 1


# 使用反卷积提取特征图
class ZFNetFmapVisual:
    def __init__(self, net, model_cfg):
        self.model = net
        self.get_model_arch(self.model, model_cfg)

    def get_model_arch(self, net, model_cfg):  # 储存神经网络的结构
        model = {}
        all_kernels = []
        weight_key = net.state_dict().keys()
        print("weight_key:", weight_key)
        for key in weight_key:
            module = key.split('.')[2]
            if 'Conv' in module:  # 提取所有卷积层的卷积核的值，并按顺序储存
                kernel = net.state_dict()[key].numpy()
                print('kernel0:', kernel)
                all_kernels.append(kernel[0])

        # 读取出模型的配置文件的内容
        with open(model_cfg, 'r') as f:
            lines = f.read().split('\n')  # 分行，顺便去掉\n，从这个角度说，比使用readlines好
            lines = [x for x in lines if len(x) > 0]  # 去掉空行
            lines = [x for x in lines if x[0] != '#']  # 去掉注释行
        # 将每个块放入列表保存
        block = {}  # 将一个块的有效的内容保存为字典
        blocks = []  # 将所有块保存为一个列表
        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1]
            else:
                key, value = line.split('=')
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

        prev_filters = 3
        for i, block in enumerate(blocks[1:]):
            if block['type'] == 'convolutional':  # 对于卷积块！！！！！！！！！！！！！！！
                try:
                    batch_normalize = block['batch_normalize']
                    bais = False
                except:
                    batch_normalize = 0
                    bais = True
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                if block['pad']:  # 检查是否需要pad
                    padding = (int(block['size']) - 1) // 2
                else:
                    padding = 0

                if batch_normalize:
                    ...
                if block['activation'] == 'leaky':
                    ...
            elif block['type'] == 'shortcut':  # 对于残差连接！！！！！！！！！！！！！！！
                ...
            ...

    def fmap_path(self):  # 将当前特征图反卷积要经过的路径储存为列表
        ...

    def handle_fmap(self):  # 按路径顺序处理特征图
        ...

    def save_visualmap(self):  # 保存反卷积的特征图
        ...

    def demaxpooling(self):  # 逆池化操作
        ...

    def deconv(self):  # 反卷积（转置卷积）操作
        ...


class GetDeconvFmap():
    def __init__(self, net):
        self.model = net
        weight_key = net.state_dict().keys()
        for key in weight_key:
            module = key.split('.')[2]
            if 'Conv' in module:
                self.kernel = net.state_dict()[key].numpy()
                print(self.kernel[0])

    def get_fmap(self, feature, num_ch=-1, nrow=8, padding=10, pad_value=1, save_feature=True, show_feature=True,
                 save_path=r'D:\MyCNN\visual\feature_map'):
        b, c, h, w = feature.shape
        feature = feature[0]
        feature = feature.unsqueeze(1)
        if c > num_ch > 0:
            feature = feature[:num_ch]
        img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
        img = img.detach().cpu()
        img = img.numpy()

        images = img.transpose((1, 2, 0))
        title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c)
        plt.title(title)
        plt.imshow(images)
        if show_feature:
            plt.show()
        if save_feature:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, title))

    def convisual(self, activation, image_per_row=16):
        '''
        一个网上的特征图可视化方法
        :param activation:模型中间层输出
        :param image_per_row: 每行显示几个特征图
        :return:
        '''
        layer_names = []
        for layer in self.model.state_dict().keys():
            layer_names.append(layer.split('.')[2])

        for layer_name, layer_activation in zip(layer_names, activation):
            n_features = layer_activation.shape[0]  # 所有特征通道数量
            size = layer_activation.shape[1]  # (n_feature,size,size)
            n_col = n_features // image_per_row  # 总行数
            display_grid = np.zeros((size * n_col, image_per_row * size))
            for col in range(n_col):
                for row in range(image_per_row):
                    channel_image = layer_activation[col * image_per_row + row, :, :]
                    channel_image = channel_image - channel_image.mean()
                    channel_image = channel_image / channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = channel_image.cpu().numpy()
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

    # 提取对应的卷积核并处理
    # def conv_kernel():
    #     ...
