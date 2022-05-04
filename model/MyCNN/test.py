# 对train.py的训练参数的测试代码
import torch
import argparse
import numpy as np
from my_cnn53 import MyCnn
from my_cnn_test import My_CNN_test
from data.truck_car_load import test_load
from util.utils import process_tqdm, frame_handle
from visual.feature_map import GetDeconvFmap, ZFNetFmapVisual
import logging
import warnings
import cv2

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("D:/MyCNN/result/log_cnn.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='config/c53.cfg', type=str)
    parser.add_argument('--truck_car_dataset', dest='tc_datafile', help='数据集地址', default='D:/data/finaldataset_type/',
                        type=str)
    parser.add_argument('--model_state', dest='statefile', help='模型参数保存地址',
                        default='D:/save_for_mycnn/model_1650001305.6539807_10_0.795.pth', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='416', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)

    return parser.parse_args()


def test_acc(predict_list):
    right = 0
    for result in predict_list:
        predict = np.array(result[0].cpu())
        label = np.array(result[1].cpu())
        if label == predict:
            right += 1
    acc = right / len(predict_list)
    return acc


def test(net, testdata):
    net.eval()
    get_feature = 0
    predict_list = []
    li_tmp = []
    y = [1 for _ in range(200)]
    y[0:21] = [0 for _ in range(21)]
    y[115:138] = [0 for _ in range(23)]
    y[197:200] = [0 for _ in range(3)]
    # testdata = zip(testdata, y)
    process = process_tqdm(y)  # testdata
    # process = process_tqdm(testdata)
    for i, y in enumerate(process):  # (x, y)
        process.set_description(f'测试进度')
        if isinstance(y, int):
            y = torch.tensor([y])
        process.set_description(f'测试进度')
        x = cv2.imread(f'D:/data/final_datasets/2/{i + 1}.png')
        # x = np.array(x) * 255
        x = frame_handle(x, int(args.inp_dim))
        x = x.cuda()
        y=y.cuda()
        with torch.no_grad():
            predict = net(x, get_feature)  # 参数暂时搁置
        logger.info(f'{predict.detach().cpu().numpy()[0][1]}, {y.cpu().numpy()}')
        if get_feature == 0:
            get_feature += 1
        li_tmp.append(predict)
        predict = torch.argmax(predict, 1)
        predict_list.append([predict, y])
        # logger.info(f'{predict.cpu().numpy()}, {y.cpu().numpy()}')
    accuracy = test_acc(predict_list)

    print(f'acc:{accuracy}')


if __name__ == '__main__':
    args = arg_init()

    # 加载训练好的模型
    net = MyCnn(args.cfgfile)
    # net = My_CNN_test()
    state_dict = torch.load(args.statefile)
    net.load_state_dict(state_dict)
    # visual_fmap = GetDeconvFmap(net)
    # deconv_fmap = ZFNetFmapVisual(net, args.cfgfile)
    net.cuda()
    testdata = test_load(args.tc_datafile)
    test(net, testdata)
