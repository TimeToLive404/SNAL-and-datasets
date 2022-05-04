# 对于CNN使用YOLO卷积头有效性的train代码
import numpy as np
import torch
from network.vit import ViT
from data.truck_car_load import data_load
import argparse
from torchsummary import summary
from util.utils import frame_handle, frame_handle_batch4, process_tqdm, modelsize, acc_cul, train_visual_tb, \
    train_visual_hl
import os
import time
import hiddenlayer as hl
from tqdm import tqdm
import cv2
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as T


def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='config/c53.cfg', type=str)
    parser.add_argument('--cifar_dataset', dest='cifar_datafile', help='数据集地址', default='data/cifar-10-batches-py',
                        type=str)
    parser.add_argument('--truck_car_dataset', dest='tc_datafile', help='数据集地址', default='D:/data/random_train_type/',
                        type=str)  # tc_train
    parser.add_argument('--learning_rate', dest='lr', help='学习率', default='0.0001')
    parser.add_argument('--save_path', dest='savefile', help='模型参数保存地址', default='D:/save_for_vit', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='224', type=str)
    parser.add_argument('--epoch', dest='epochs', help='模型训练轮数', default='10', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)  # hl、tb、None

    return parser.parse_args()


def train(net, traindata, loss_function, optimizer, epochs, visual=None):
    net.train()
    summary(net, (3, 224, 224))

    if visual == 'hl':
        history1 = hl.History()
        canvas1 = hl.Canvas()

    for epoch in range(epochs):
        y = [1 for _ in range(250)]
        y[120] = 0
        y[121] = 0
        y[122] = 0
        # traindata = zip(traindata, y)
        process = tqdm(y)
        # process = process_tqdm(traindata)
        for i, y in enumerate(process):  # (x, y)
            process.set_description(f'第{epoch + 1}轮训练')
            if isinstance(y, int):
                y = torch.tensor([y])
            i_name = str(i + 1).zfill(4)
            x = cv2.imread(f'D:/data/seq_train/2/{i_name}.png')
            # x = np.array(x) * 255
            x = frame_handle(x, int(args.inp_dim))
            x = x.cuda()
            y = y.cuda()
            predict = net(x)

            loss = loss_function(predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 以下为模型监测部分
            # modelsize(net, x)  # GPU监测
            global_step = epoch * len(process) + i + 1
            if global_step % 10 == 0:
                if visual == 'hl':
                    acc = acc_cul(predict, y)
                    history1.log((epoch, i), train_loss=loss, train_acc=acc)
                    with canvas1:
                        canvas1.draw_plot(history1["train_loss"])
                        # canvas1.draw_plot(history1["train_acc"])
        canvas1.save('D:/MyViT/result/canvas.png')


def save_model():
    savename = f'model_{time.time()}.pth'
    savepath = os.path.join(args.savefile, savename)
    torch.save(net.state_dict(), savepath)


if __name__ == '__main__':
    args = arg_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available()

    net = ViT(in_channel=3, patch_size=16, emb_size=768, img_size=int(args.inp_dim), depth=6, n_class=10)
    # net = My_CNN_test()
    net.cuda()
    traindata = data_load(args.tc_datafile)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    epochs = int(args.epochs)
    visual = args.visual

    train(net, traindata, loss_function, optimizer, epochs, visual)
    save_model()
