import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from tensorboardX import SummaryWriter
import hiddenlayer as hl


def frame_handle(frame, input_dim):
    # 缩放到416*416
    # frame = frame.squeeze()
    # frame = frame.transpose((2, 1, 0))
    # frame = frame.transpose((1, 2, 0))
    w, h = input_dim, input_dim
    new_w = int(frame.shape[1] * min(w / frame.shape[1], h / frame.shape[0]))
    new_h = int(frame.shape[0] * min(w / frame.shape[1], h / frame.shape[0]))
    new_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((input_dim, input_dim, 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = new_frame
    # H*W*C->C*H*W->B*C*H*W
    canvas = canvas[:, :, ::-1].transpose((2, 0, 1)).copy()  # 先转换成RGB格式，再把C放到第一个位置，为转换成tensor做准备
    canvas = torch.from_numpy(canvas).float().div(255.0).unsqueeze(0)  # 每个元素除以255以归一化,添加一个batch维
    return canvas


def frame_handle_batch4(frame, input_dim):
    # 缩放到416*416
    canvas_all = []
    write = 0
    frame = frame.transpose((0, 2, 3, 1))
    for i in range(len(frame)):
        # frame[i] = frame[i].transpose((2, 1, 0))
        w, h = input_dim, input_dim
        new_w = int(frame[i].shape[1] * min(w / frame[i].shape[1], h / frame[i].shape[0]))
        new_h = int(frame[i].shape[0] * min(w / frame[i].shape[1], h / frame[i].shape[0]))
        new_frame = cv2.resize(frame[i], (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((input_dim, input_dim, 3), 128)
        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = new_frame
        # H*W*C->C*H*W->B*C*H*W
        canvas = canvas[:, :, ::-1].transpose((2, 0, 1)).copy()  # 先转换成RGB格式，再把C放到第一个位置，为转换成tensor做准备
        canvas = torch.from_numpy(canvas).float().div(255.0).unsqueeze(0)  # 每个元素除以255以归一化,添加一个batch维
        canvas_all.append(canvas)
        if write == 0:
            result = canvas
            write += 1
        else:
            result = torch.cat((result, canvas), 0)
    # result = torch.cat((canvas_all[0], canvas_all[1], canvas_all[2], canvas_all[3]), 0)
    return result


# 显示训练进度条的加载函数
def process_tqdm(obj, *args, **kwargs):
    try:
        return tqdm(obj, *args, **kwargs)
    except:
        return obj


# 显示模型的GPU使用信息
# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32
def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


# 神经网络训练可视化tensorboard实现
def train_visual_tb(loss=0, acc=0, global_step=0, log_path='./visual/runs/exp'):
    # 实例化
    sumwriter = SummaryWriter(logdir=log_path)
    # 损失值的变化
    sumwriter.add_scalar("train_loss", scalar_value=loss, global_step=global_step)
    # 精度的变化
    sumwriter.add_scalar("train_acc", scalar_value=acc, global_step=global_step)
    # 网络参数分布直方图

    ...


# 使用hiddenlayer可视化训练过程
def train_visual_hl(loss, acc, epoch, step):
    history1 = hl.History()
    canvas1 = hl.Canvas()
    history1.log((epoch, step), train_loss=loss, train_acc=acc)
    with canvas1:
        canvas1.draw_plot(history1["train_loss"])
        canvas1.draw_plot(history1["train_acc"])


# 分类精度计算
def acc_cul(y_hat, y):
    return 0
