# 对于CNN使用YOLO卷积头有效性的train代码
import numpy as np
import torch
from network.snal import MLP_decision, Snal, Darknet, scene_eigenvalue, Mineuron
from data.truck_car_load import data_load, test_load, seq_train_dataload, seq_test_dataload
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
import math
from einops import rearrange
import warnings
from copy import deepcopy
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("D:/SNAL/result/log_min.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='D:/SNAL/config/yolov3.cfg', type=str)
    parser.add_argument('--cifar_dataset', dest='cifar_datafile', help='数据集地址', default='data/cifar-10-batches-py',
                        type=str)
    parser.add_argument('--train_dataset', dest='train_datafile', help='数据集地址', default='D:/data/random_train_type/',
                        type=str)  # D:/data/tc_train/
    parser.add_argument('--test_dataset', dest='test_datafile', help='数据集地址', default='D:/data/tmp/',
                        type=str)
    parser.add_argument('--learning_rate', dest='lr', help='学习率', default='0.0001')
    parser.add_argument('--save_path', dest='savefile', help='模型参数保存地址', default='D:/saveformlpinsnal', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='416', type=str)
    parser.add_argument('--num_class', dest='num_class', help='要区别的目标种类数', default='80', type=str)
    parser.add_argument('--confidence', dest='conf_thre', help='置信度阈值', default='0.5', type=str)
    parser.add_argument('--NMS', dest='nms_thre', help='NMS阈值', default='0.4', type=str)
    parser.add_argument('--epoch', dest='epochs', help='模型训练轮数', default='2', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)  # hl、tb、None
    parser.add_argument('--model_state', dest='statefile', help='模型参数保存地址',
                        default='D:/saveformlpinsnal/circle_model_1651042274.8159363_0.885.pth', type=str)
    parser.add_argument("--weights", dest='weightsfile', help="模型权重文件", default="D:/SNAL/yolov3.weights", type=str)
    parser.add_argument("--traindata_len", dest='trainlen', help='训练集长度', default=250, type=int)

    return parser.parse_args()


def train(yolo, net, traindata, loss_function, optimizer, epochs, num_class, conf_thre, nms_thre, visual=None,
          pattern='train'):
    # yolo.eval()
    net.train()

    if visual == 'hl':
        history1 = hl.History()
        canvas1 = hl.Canvas()

    for epoch in range(epochs):
        # # 按顺序设置标签
        # y = [1 for _ in range(250)]
        # y[120] = 0
        # y[121] = 0
        # y[122] = 0
        # traindata = zip(traindata, y)
        process = tqdm(traindata)
        for i, (img, y) in enumerate(process):  # i, ((img, _), y)
            # if isinstance(y, int):
            #     y = torch.tensor([y])
            # i_name = str(i + 1).zfill(4)
            # img = cv2.imread(f'D:/data/seq_train/2/{i_name}.png')
            # darknet部分，每批一张图片
            process.set_description(f'第{epoch + 1}轮训练')
            img = (np.array(img) * 255)[0].transpose((1, 2, 0))
            img = img.copy()
            x = frame_handle(img, int(args.inp_dim))
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                output = yolo(x)

            L = scene_eigenvalue(output[1], num_class, conf_thre, nms_thre, img, pattern=pattern,
                                 common_evalue=47332, i=i)  #
            A = L - 200000  # 20000
            if A >= 0:
                Iapp = 520 / (1 + math.exp(-A)) - 220
            else:
                Iapp = 40
            MIN_output = Mineuron(Iapp, i)  # 中间神经元电位处理后的输出
            # logging.info(MIN_output)
            # MIN_output = rearrange(torch.tensor(MIN_output), 'o -> b o', b=output[0][0])
            MIN_output = torch.tensor([MIN_output]).unsqueeze(0)#.unsqueeze(0)*10
            MLP_input = torch.cat(
                (torch.tensor(MIN_output).cuda(), output[0].reshape(-1, 1024 * 13 * 13)),  # / 100
                1)  # 合在一起作为MLP输入
            # mlp部分，需要训练的部分
            predict = net(MLP_input)
            loss = loss_function(predict, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 以下为模型监测部分
            # modelsize(net, x)  # GPU监测
            global_step = epoch * args.trainlen + i + 1  # epoch * len(process) + i + 1
            if global_step % 10 == 0:
                if visual == 'hl':
                    acc = acc_cul(predict, y)
                    history1.log((epoch, i), train_loss=loss, train_acc=acc)
                    with canvas1:
                        canvas1.draw_plot(history1["train_loss"])
                        # canvas1.draw_plot(history1["train_acc"])
        canvas1.save('D:/SNAL/result/canvas3.png')


def save_model():
    savename = f'circle_model_{time.time()}.pth'
    savepath = os.path.join(args.savefile, savename)
    torch.save(decision_maker.state_dict(), savepath)


def test(yolo, net, testdata, num_class, conf_thre, nms_thre, pattern='train'):
    net.eval()
    predict_list = []
    li_tmp = []

    y = [1 for _ in range(200)]
    y[0:21] = [0 for _ in range(21)]
    y[115:138] = [0 for _ in range(23)]
    y[197:200] = [0 for _ in range(3)]
    # testdata = zip(testdata, y)
    process = process_tqdm(y)  # testdata
    for i, y in enumerate(process):  # (img, y)((img, _), y)
        if isinstance(y, int):
            y = torch.tensor([y])
        process.set_description(f'测试进度')
        img = cv2.imread(f'D:/data/final_datasets/2/{i + 1}.png')
        # img = (np.array(img) * 255)[0].transpose((1, 2, 0))
        # img = img.copy()
        x = frame_handle(img, int(args.inp_dim))
        x = x.cuda()
        y = y.cuda()  # 以上为数据与标签处理
        with torch.no_grad():
            output = yolo(x)  # yolo输出两个值，一个给CNN，一个给MIN

        L = scene_eigenvalue(output[1], num_class, conf_thre, nms_thre, img, pattern=pattern,
                             common_evalue=47332, i=i)
        A = L - 100000  # L:当前特征值，20000为平均特征值
        logger.info(L)
        if A >= 0:
            Iapp = 520 / (1 + math.exp(-A)) - 220
        else:
            Iapp = 40
        MIN_output = Mineuron(Iapp, i)  # 中间神经元电位处理后的输出
        # logger.info(MIN_output)
        # MIN_output = torch.tensor(MIN_output).unsqueeze(0).unsqueeze(0)
        MIN_output = torch.tensor([MIN_output]*10).unsqueeze(0)#.unsqueeze(0)
        # MIN_output = torch.tensor([0] * 10).unsqueeze(0)  # .unsqueeze(0)
        MLP_input = torch.cat(
            (torch.tensor(MIN_output ).cuda(), output[0].reshape(-1, 1024 * 13 * 13)),  # / 100
            1)  # 合在一起作为MLP输入
        # mlp部分，需要训练的部分
        # logger.info(MLP_input)
        predict = net(MLP_input)
        # logger.info(f'{predict.detach().cpu().numpy()[0][1]}, {y.cpu().numpy()}')
        li_tmp.append(predict)
        predict = torch.argmax(predict, 1)
        predict_list.append([predict, y])

        # logger.info(f'L,Iapp,pre,y: {L}, {Iapp}, {predict.cpu().numpy()}, {y.cpu().numpy()}')
    accuracy = test_acc(predict_list)

    print(f'acc:{accuracy}')


def test_acc(predict_list):
    right = 0
    for result in predict_list:
        predict = np.array(result[0].cpu())
        label = np.array(result[1].cpu())
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
        traindata = data_load(args.train_datafile)  # seq_train_dataload(args.train_datafile)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(decision_maker.parameters(), lr=float(args.lr))
        epochs = int(args.epochs)
        visual = args.visual
        train(yolo, decision_maker, traindata, loss_function, optimizer, epochs, args.num_class, args.conf_thre,
              args.nms_thre, visual=visual, pattern='test')
        save_model()

    elif operation == '1':
        yolo = yolo.cuda()
        yolo.load_weights(args.weightsfile)
        decision_maker.cuda()
        state_dict = torch.load(args.statefile)
        decision_maker.load_state_dict(state_dict)
        testdata = test_load(args.test_datafile)  # seq_test_dataload(args.test_datafile)
        test(yolo, decision_maker, testdata, args.num_class, args.conf_thre, args.nms_thre, pattern='test')
