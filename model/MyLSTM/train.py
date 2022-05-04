# 对于CNN使用YOLO卷积头有效性的train代码
import numpy as np
import torch
from network.rnn import Rnn
from network.cnn import Darknet
from data.tc_load import data_load
import argparse
from util.utils import frame_handle, frame_handle_batch4, process_tqdm, modelsize, acc_cul, train_visual_tb, \
    train_visual_hl, gray
import os
import cv2
import time
import hiddenlayer as hl
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image


def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='config/yolov3.cfg', type=str)
    parser.add_argument('--truck_car_dataset', dest='tc_datafile', help='数据集地址', default='D:/data/random_train_type/',
                        type=str)  # tc_train
    parser.add_argument('--learning_rate', dest='lr', help='学习率', default='0.001')
    parser.add_argument('--save_path', dest='savefile', help='模型参数保存地址', default='D:/save_for_myrnn', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='416', type=str)
    parser.add_argument('--epoch', dest='epochs', help='模型训练轮数', default='6', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)  # hl、tb、None
    parser.add_argument("--weights", dest='weightsfile', help="模型权重文件", default="D:/MyRNN/yolov3.weights", type=str)

    return parser.parse_args()


def train(cnn, rnn, traindata, loss_function, optimizer, epochs, h_prev, visual=None, random=True):
    rnn.train()

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
            if isinstance(y, int):
                y = torch.tensor([y])
            i_name = str(i + 1).zfill(4)
            x = cv2.imread(f'D:/data/seq_train/2/{i_name}.png')
            if random:
                h_prev = torch.zeros(1, 1, 32).cuda()  # (layer?,batch_size,hidden_size) 如果图片不存在时间关系，就每张图初始化状态
            process.set_description(f'第{epoch + 1}轮训练')
            # x = (np.array(x) * 255)[0].transpose((1, 2, 0))
            # x = x.copy()
            x = frame_handle(x, int(args.inp_dim))
            x = x.cuda()
            y = y.cuda()
            cnn_output = cnn(x)
            rnn_input = cnn_output[0].reshape(-1, 1024, 13 * 13)
            predict = rnn(rnn_input, h_prev)
            h_prev = h_prev.detach().cuda()
            result = predict[0][0].view(1, 10)
            loss = loss_function(result, y)
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
        canvas1.save('D:/MyRnn/result/canvas.png')


def save_model():
    savename = f'model_{time.time()}.pth'
    savepath = os.path.join(args.savefile, savename)
    torch.save(rnn.state_dict(), savepath)


if __name__ == '__main__':
    args = arg_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available()

    rnn = Rnn()
    cnn = Darknet(args.cfgfile)
    cnn.cuda()
    rnn.cuda()
    cnn.load_weights(args.weightsfile)  # 加载yolo预训练模型
    traindata = data_load(args.tc_datafile)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=float(args.lr))
    epochs = int(args.epochs)
    visual = args.visual
    h_prev = torch.zeros(1, 3, 20)  # (layer?,batch_size,hidden_size)

    train(cnn, rnn, traindata, loss_function, optimizer, epochs, h_prev, visual)
    save_model()
