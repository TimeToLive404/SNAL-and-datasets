# 对于CNN使用YOLO卷积头有效性的train代码
import numpy as np
import torch
from network.snal import MLP_decision, Snal, Darknet
from data.truck_car_load import data_load, test_load
import argparse
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
import cv2


def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='D:/SNAL/config/yolov3.cfg', type=str)
    parser.add_argument('--cifar_dataset', dest='cifar_datafile', help='数据集地址', default='data/cifar-10-batches-py',
                        type=str)
    parser.add_argument('--train_dataset', dest='train_datafile', help='数据集地址', default='D:/data/tc_train/',
                        type=str)
    parser.add_argument('--test_dataset', dest='test_datafile', help='数据集地址', default='D:/data/tc_test/',
                        type=str)
    parser.add_argument('--learning_rate', dest='lr', help='学习率', default='0.0001')
    parser.add_argument('--save_path', dest='savefile', help='模型参数保存地址', default='D:/saveformlpinsnal', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='416', type=str)
    parser.add_argument('--epoch', dest='epochs', help='模型训练轮数', default='1', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)  # hl、tb、None
    parser.add_argument('--model_state', dest='statefile', help='模型参数保存地址',
                        default='D:/saveformlpinsnal/model_1647941841.320632.pth', type=str)
    parser.add_argument("--weights", dest='weightsfile', help="模型权重文件", default="D:/SNAL/yolov3.weights", type=str)

    return parser.parse_args()


def train(yolo, net, traindata, loss_function, optimizer, epochs, visual=None):
    net.train()

    if visual == 'hl':
        history1 = hl.History()
        canvas1 = hl.Canvas()

    for epoch in range(epochs):
        process = process_tqdm(traindata)
        for i, (x, y) in enumerate(process):
            # darknet部分，每批一张图片
            process.set_description(f'第{epoch + 1}轮训练')
            x = (np.array(x) * 255)[0].transpose((1, 2, 0))
            x = x.copy()
            x = frame_handle(x, int(args.inp_dim))
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                output = yolo(x)

            # mlp部分，需要训练的部分
            predict = net(output[0])
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
        canvas1.save('D:/SNAL/result/canvas2.png')


def save_model():
    savename = f'model_{time.time()}.pth'
    savepath = os.path.join(args.savefile, savename)
    torch.save(decision_maker.state_dict(), savepath)


def test(yolo, net, testdata):
    net.eval()
    predict_list = []
    li_tmp = []
    process = process_tqdm(testdata)
    for i, (x, y) in enumerate(process):
        process.set_description(f'测试进度')
        x = (np.array(x) * 255)[0].transpose((1, 2, 0))
        x = x.copy()
        x = frame_handle(x, int(args.inp_dim))
        x = x.cuda()
        with torch.no_grad():
            output = yolo(x)

        predict = net(output[0])
        li_tmp.append(predict)
        predict = torch.argmax(predict, 1)
        predict_list.append([predict, y])
    accuracy = test_acc(predict_list)

    print(f'acc:{accuracy}')


def test_acc(predict_list):
    right = 0
    for result in predict_list:
        predict = np.array(result[0].cpu())
        label = np.array(result[1])
        if label == predict:
            right += 1
    acc = right / len(predict_list)
    return acc


if __name__ == '__main__':
    args = arg_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available()
    operation = input("输入0进行训练，输入1进行测试：")

    yolo = Darknet(args.cfgfile)
    decision_maker = MLP_decision()

    if operation == '0':
        yolo = yolo.cuda()
        yolo.load_weights(args.weightsfile)
        decision_maker.cuda()
        traindata = data_load(args.train_datafile)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(decision_maker.parameters(), lr=float(args.lr))
        epochs = int(args.epochs)
        visual = args.visual

        train(yolo, decision_maker, traindata, loss_function, optimizer, epochs, visual)
        save_model()

    elif operation == '1':
        yolo = yolo.cuda()
        yolo.load_weights(args.weightsfile)
        decision_maker.cuda()
        state_dict = torch.load(args.statefile)
        decision_maker.load_state_dict(state_dict)

        testdata = test_load(args.test_datafile)
        test(yolo, decision_maker, testdata)
