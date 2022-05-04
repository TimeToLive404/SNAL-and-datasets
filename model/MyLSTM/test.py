# 对train.py的训练参数的测试代码
import torch
import argparse
import numpy as np
from network.rnn import Rnn
from network.cnn import Darknet
from data.tc_load import test_load
from util.utils import process_tqdm, frame_handle
import cv2
import logging

# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("D:/MyLSTM/result/log_rnn.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='config/yolov3.cfg', type=str)
    parser.add_argument('--truck_car_dataset', dest='tc_datafile', help='数据集地址', default='D:/data/final_datasets/',
                        type=str)
    parser.add_argument('--model_state', dest='statefile', help='模型参数保存地址',
                        default='D:/save_for_myrnn/model_1650027647.4625022.pth', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='416', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)
    parser.add_argument("--weights", dest='weightsfile', help="模型权重文件", default="D:/MyRNN/yolov3.weights", type=str)

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


def test(cnn, rnn, testdata, h_prev, random=True):
    rnn.eval()
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
        if random:
            h_prev = torch.zeros(1, 1, 32).cuda()
        if isinstance(y, int):
            y = torch.tensor([y])
        process.set_description(f'测试进度')
        x = cv2.imread(f'D:/data/final_datasets/2/{i + 1}.png')
        # x = (np.array(x) * 255)[0].transpose((1, 2, 0))
        # x = x.copy()
        x = frame_handle(x, int(args.inp_dim))
        x = x.cuda()
        y = y.cuda()
        cnn_output = cnn(x)
        rnn_input = cnn_output[0].reshape(-1, 1024, 13 * 13)
        with torch.no_grad():
            predict = rnn(rnn_input, h_prev)
        logger.info(f'{predict[0].detach().cpu().numpy()[0][1]}, {y.cpu().numpy()}')
        h_prev = h_prev.detach().cuda()
        result = predict[0][0].view(1, 10)
        li_tmp.append(result)
        predict = torch.argmax(result, 1)
        predict_list.append([predict, y])
    accuracy = test_acc(predict_list)

    print(f'acc:{accuracy}')


if __name__ == '__main__':
    args = arg_init()

    # 加载训练好的模型
    rnn = Rnn()
    cnn = Darknet(args.cfgfile)
    cnn.cuda()
    rnn.cuda()
    cnn.load_weights(args.weightsfile)  # 加载yolo预训练模型
    state_dict = torch.load(args.statefile)
    rnn.load_state_dict(state_dict)
    h_prev = torch.zeros(1, 1024, 20)  # (layer?,batch_size,hidden_size)

    testdata = test_load(args.tc_datafile)
    test(cnn, rnn, testdata, h_prev)
