import torch
import torch.nn as nn
import numpy as np
import cv2
import math

import pickle as pkl


# from visual.feature_map import get_feature_map

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def predict_transform(prediction, input_dim, anchor, num_class):
    '''
    :param prediction:各通道下的锚框参数
    :return: 416*416的坐标下各通道的锚框参数
    '''
    num_anchor = len(anchor)
    bbox_attrs = 5 + int(num_class)
    # width = prediction.size(2)
    stride = int(input_dim) // prediction.size(2)
    grid_size = prediction.size(2)

    # 比如目标是将（1，3*85，13，13）转换成（1，13*13*3，85）
    # 比如将（1，3*85，13，13）转换成（1，3*85，13*13）
    prediction = prediction.view(prediction.size(0), prediction.size(1), prediction.size(2) * prediction.size(3))
    # （1，3*85，13*13）->（1，13*13*3，85）
    prediction = prediction.transpose(1, 2).contiguous()  # 加上contiguous才能在transpose后再次view
    prediction = prediction.view(prediction.size(0), prediction.size(1) * num_anchor, bbox_attrs)
    # 按照公式处理锚框宽高
    anchor = [(a[0] / stride, a[1] / stride) for a in anchor]
    anchor = torch.FloatTensor(anchor)
    anchor = anchor.cuda()
    anchor = anchor.repeat(grid_size * grid_size, 1).unsqueeze(0)  # 将三个锚框重复格数次
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchor
    # 处理锚框坐标
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    grid_range = np.arange(grid_size)
    x, y = np.meshgrid(grid_range, grid_range)
    x = torch.FloatTensor(x).view(-1, 1)
    y = torch.FloatTensor(y).view(-1, 1)
    x = x.cuda()
    y = y.cuda()
    x_y = torch.cat((x, y), 1).repeat(1, num_anchor).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y
    # 将锚框数值还原回（416*416）的坐标
    prediction[:, :, :4] = prediction[:, :, :4] * stride
    # 处理置信度
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # 处理类别预测值
    prediction[:, :, 5:5 + int(num_class)] = torch.sigmoid((prediction[:, :, 5:5 + int(num_class)]))
    return prediction


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


# 读取模型的配置文件，返回网络超参数配置和网络层配置
def model_load(model_cfg):
    # 读取模型的配置文件
    with open(model_cfg, 'r') as f:
        lines = f.read().split('\n')  # 分行，顺便去掉\n，从这个角度说，比使用readlines好
        lines = [x for x in lines if len(x) > 0]  # 去掉空行
        lines = [x for x in lines if x[0] != '#']  # 去掉注释行
        # lines = [x.rstrip().lstrip() for x in lines]
        # print(lines)
    # 提取每个模块
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

    # 建立相应的模型
    module_list = nn.ModuleList()  # 模型列表
    prev_filters = 3  # 输入通道初始
    filters = 0  # 输出通道定义一下
    output_filters = []  # 记录通道数，在route层起作用，通过这个知道通道数是多少
    for i, block in enumerate(blocks[1:]):
        module = nn.Sequential()
        if block['type'] == "convolutional":
            # 如果为卷积模块，则向模块中加入卷积，BN层，激活函数
            # 处理卷积块的参数，主要是将str转换成int
            try:  # 检查是否有bn层
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            if block['pad']:  # 检查是否需要pad
                pad = (int(block['size']) - 1) // 2
            else:
                pad = 0
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            # 设置层并加入到Sequential中
            conv = nn.Conv2d(prev_filters, filters, kernel_size=kernel_size, stride=stride,  # 设置卷积层
                             padding=pad, bias=bias)
            module.add_module(f'Conv_{i}', conv)  # 加入卷积层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)  # 设置BN层
                module.add_module(f'bn_{i}', bn)  # 加入BN层
            if block['activation'] == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)  # 设置激活函数
                module.add_module(f'LeakyReLU_{i}', activn)  # 加入激活函数

        elif block['type'] == "shortcut":
            short_cut = EmptyLayer()  # 仅仅是占位记录作用
            module.add_module(f'Shortcut_{i}', short_cut)

        elif block['type'] == "yolo":
            # 提取出mask指定的anchor,012小目标，345中目标，678大目标
            mask = block['mask'].split(',')
            mask = [int(x) for x in mask]
            anchor = block['anchors'].split(',')
            anchor = [int(x) for x in anchor]
            anchor = [(anchor[i], anchor[i + 1]) for i in range(0, len(anchor), 2)]
            anchor = [anchor[i] for i in mask]

            detection = DetectionLayer(anchor)
            module.add_module(f'Detection_{i}', detection)  # 记录了anchor

        elif block['type'] == "route":
            route = EmptyLayer()
            module.add_module(f'Route_{i}', route)  # route也是占位记录的空层，但是会改变输出通道
            # 获取参数值
            start = int(block['layers'].split(',')[0])
            try:
                end = int(block['layers'].split(',')[1])
            except:
                end = 0
            # 将起始点的绝对位置，变换成相对于当前层i的相对位置
            if start > 0:
                start = start - i
            if end > 0:
                end = end - i
            # 输出通道数
            if end < 0:  # 如果有end
                filters = output_filters[i + start] + output_filters[i + end]
            else:
                filters = output_filters[i + start]

        elif block['type'] == "upsample":
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            module.add_module(f'Upsample_{i}', upsample)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)  # 记录本层通道数

    # 返回网络信息，模型列表
    return blocks, module_list


class Darknet(nn.Module):
    def __init__(self, model_cfg=r'D:\SNAL\config\yolov3.cfg'):
        super(Darknet, self).__init__()
        # 加载cfg中的网络模型
        self.blocks, self.module_list = model_load(model_cfg)
        self.net_info = self.blocks[0]

    def forward(self, x):
        outputs = {}  # 记录每层的输出
        write = 0
        # 按顺序使用模型处理x，返回输出
        for i, module in enumerate(self.blocks[1:]):
            if module['type'] == 'convolutional':  # 进行卷积计算
                x = self.module_list[i](x)
                # get_feature_map(x)
                if i == 73:
                    last_conv = x

            elif module['type'] == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]  # 残差连接结构

            elif module['type'] == 'route':  # 用于引入和拼接
                layers = module['layers'].split(',')
                layers = [int(x) for x in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:  # 如果只有一个数，则route只起引入作用
                    x = outputs[i + layers[0]]
                else:  # 如果有多个数（对于v3来说是两个），则进行拼接
                    if layers[1] > 0:
                        layers[1] -= i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module['type'] == 'upsample':  # 进行上采样计算
                x = self.module_list[i](x)

            elif module['type'] == 'yolo':
                x = x.data
                # 将所有边界框的（85个）属性作为行罗列出来
                x = predict_transform(x, self.net_info['width'], self.module_list[i][0].anchors, module['classes'])
                if write == 0:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return last_conv, detections  # 返回YOLO的需要输入到后续的输出，作为后面网络结构的输入

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# YOLO部分调试程序
# net = Darknet().cuda()
# print(net)
# X = torch.rand(size=(1, 3, 416, 416)).cuda()
# Y = net(X)
# print('输出1：', Y[0], '\n', '输出2:', Y[1])


class MLP_decision(nn.Module):
    def __init__(self):
        super(MLP_decision, self).__init__()
        self.decision_maker = nn.Sequential(
            nn.Linear(173056, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.decision_maker(x)
        return output


# MLP部分调试程序
# net = MLP_decision().cuda()
# print(net)
# X = Y[0]
# decision = net(X)
# print('决策结果', decision)

# 求两个边界框的IOU
def bbox_iou(bbox1, bbox2):
    # 取出四个角的坐标值
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]
    # 计算重合区域的面积
    inter_x1, inter_y1, inter_x2, inter_y2 = torch.max(b1_x1, b2_x1), torch.max(b1_y1, b2_y1), \
                                             torch.min(b1_x2, b2_x2), torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    # 计算并集的面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x1 - b2_x1) * (b2_y2 - b2_y1)
    u_area = b2_area + b1_area - inter_area
    # 计算IOU
    iou = inter_area / u_area
    return iou


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def output_handle(output, num_class, conf_thre, nms_thre):
    write = False  # 一个控制拼接的变量
    # 保留置信度大于阈值的output，其余的归零
    conf_mask = (output[:, :, 4] > conf_thre).float().unsqueeze(2)
    output = output * conf_mask
    # 将output的（x,y,w,h）换成（x1,y1,x2,y2）的方式
    output_tmp = output.new(output.shape)
    output_tmp[:, :, 0] = output[:, :, 0] - output[:, :, 2] / 2
    output_tmp[:, :, 1] = output[:, :, 1] - output[:, :, 3] / 2
    output_tmp[:, :, 2] = output[:, :, 0] + output[:, :, 2] / 2
    output_tmp[:, :, 3] = output[:, :, 1] + output[:, :, 3] / 2
    output[:, :, :4] = output_tmp[:, :, :4]
    # NMS
    output_tmp2 = output.squeeze(0)  # 处理一下，去掉batch
    # 最大置信度的类的ID及其置信度
    max_cls, conf_max_cls = torch.max(output_tmp2[:, 5:5 + num_class], 1)
    # 拼接出（x1,y1,x2,y2,confidence,class,confidence of class）的形式
    max_cls = max_cls.float().unsqueeze(1)
    conf_max_cls = conf_max_cls.float().unsqueeze(1)
    output_tmp2 = torch.cat((output_tmp2[:, :5], max_cls, conf_max_cls), 1)  # 沿着行的方向拼接
    # 取出有效的锚框坐标
    available_box_ind = torch.nonzero(output_tmp2[:, 4])
    # 获取有效锚框的信息
    try:
        available_box = output_tmp2[available_box_ind.squeeze(), :].reshape(-1, 7)
    except:  # 如果没有有效锚框
        return 0
    if available_box.shape[0] == 0:
        return 0
    # 获取图片中包含的物体类别
    cls_img = unique(available_box[:, -1])
    for cls in cls_img:
        # 对于每个类别，取出所有对应于这个类别的锚框
        cls_mask = (available_box[:, -1] == cls).float().unsqueeze(1)
        cls_box_tmp = available_box * cls_mask
        cls_box_ind = torch.nonzero(cls_box_tmp[:, -1]).squeeze()
        cls_box = cls_box_tmp[cls_box_ind].view(-1, 7)
        # 按照confidence的大小排序
        conf_sort = torch.sort(cls_box[:, 4], descending=True)[1]  # 只看下标的排序结果
        cls_box = cls_box[conf_sort]  # 按顺序排序，为NMS取值做准备
        for i in range(cls_box.size(0)):
            try:
                iou = bbox_iou(cls_box[i].unsqueeze(0), cls_box[i + 1:])
            except:
                break
            iou_mask = (iou < nms_thre).float().unsqueeze(1)
            cls_box[i + 1:] = cls_box[i + 1:] * iou_mask
            non_zero_ind = torch.nonzero(cls_box[:, 4]).squeeze()
            cls_box = cls_box[non_zero_ind].view(-1, 7)
        # ind = cls_box.new(cls_box.size(0), 1).fill_(0)
        # seq = ind, cls_box
        if not write:
            box_frame = cls_box
            write = True
        else:
            box_frame = torch.cat((box_frame, cls_box))
    try:
        return box_frame
    except:
        return 0
