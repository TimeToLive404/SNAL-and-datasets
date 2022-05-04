# 对于CNN使用YOLO卷积头有效性的train代码
import numpy as np
import torch
from network.vit import ViT
from network.cnn import Darknet
from data.truck_car_load import data_load
import argparse
from torchsummary import summary
from util.utils import frame_handle, frame_handle_batch4, process_tqdm, modelsize, acc_cul, train_visual_tb, \
    train_visual_hl
import os
import time
import hiddenlayer as hl
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as T


def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='D:/MyViT/config/c53.cfg', type=str)
    parser.add_argument('--cifar_dataset', dest='cifar_datafile', help='数据集地址', default='data/cifar-10-batches-py',
                        type=str)
    parser.add_argument('--truck_car_dataset', dest='tc_datafile', help='数据集地址', default='D:/data/tc_train/',
                        type=str)
    parser.add_argument('--learning_rate', dest='lr', help='学习率', default='0.0001')
    parser.add_argument('--save_path', dest='savefile', help='模型参数保存地址', default='D:/save_for_vit', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='416', type=str)
    parser.add_argument('--epoch', dest='epochs', help='模型训练轮数', default='1', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)  # hl、tb、None
    parser.add_argument("--weights", dest='weightsfile', help="模型权重文件", default="D:/MyViT/yolov3.weights", type=str)

    return parser.parse_args()


def train(cnn_head, net, traindata, loss_function, optimizer, epochs, visual=None):
    net.train()
    # summary(net, (3, 416, 416))

    if visual == 'hl':
        history1 = hl.History()
        canvas1 = hl.Canvas()

    for epoch in range(epochs):
        process = process_tqdm(traindata)
        for i, (x, y) in enumerate(process):
            process.set_description(f'第{epoch + 1}轮训练')
            x = np.array(x) * 255
            x = frame_handle(x, int(args.inp_dim))
            x = x.cuda()
            y = y.cuda()
            cnn_output = cnn_head(x)
            cnn_output = cnn_output[0].reshape(-1, 1024, 13, 13)
            predict = net(cnn_output)

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

    net = ViT(in_channel=1024, patch_size=1, emb_size=1, img_size=13, depth=6, n_class=10)
    cnn_head = Darknet(args.cfgfile)
    cnn_head.load_weights(args.weightsfile)
    # net = My_CNN_test()
    net.cuda()
    cnn_head.cuda()
    traindata = data_load(args.tc_datafile)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    epochs = int(args.epochs)
    visual = args.visual

    train(cnn_head, net, traindata, loss_function, optimizer, epochs, visual)
    # save_model()
