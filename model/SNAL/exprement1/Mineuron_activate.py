# 对于CNN使用YOLO卷积头有效性的train代码
import numpy as np
import torch
from network.snal import MLP_decision, Snal, Darknet, scene_eigenvalue, Mineuron
from data.truck_car_load import data_load, test_load
import argparse
from util.utils import frame_handle, frame_handle_batch4, process_tqdm, modelsize, acc_cul, train_visual_tb, \
    train_visual_hl
import math


def arg_init():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_cfg', dest='cfgfile', help='网络模型配置文件地址', default='D:/SNAL/config/yolov3.cfg', type=str)
    parser.add_argument('--cifar_dataset', dest='cifar_datafile', help='数据集地址', default='data/cifar-10-batches-py',
                        type=str)
    parser.add_argument('--test_dataset', dest='test_datafile', help='数据集地址', default=r'D:\data\specialcase/',
                        type=str)
    parser.add_argument('--learning_rate', dest='lr', help='学习率', default='0.0001')
    parser.add_argument('--save_path', dest='savefile', help='模型参数保存地址', default='D:/saveformlpinsnal', type=str)
    parser.add_argument('--inp_dim', dest='inp_dim', help='网络要处理的图片尺寸', default='416', type=str)
    parser.add_argument('--num_class', dest='num_class', help='要区别的目标种类数', default='80', type=str)
    parser.add_argument('--confidence', dest='conf_thre', help='置信度阈值', default='0.5', type=str)
    parser.add_argument('--NMS', dest='nms_thre', help='NMS阈值', default='0.4', type=str)
    parser.add_argument('--epoch', dest='epochs', help='模型训练轮数', default='1', type=str)
    parser.add_argument('--visual', dest='visual', help='可视化方式', default='hl', type=str)  # hl、tb、None
    parser.add_argument('--model_state', dest='statefile', help='模型参数保存地址',
                        default='D:/saveformlpinsnal/model_1647341383.6291401.pth', type=str)
    parser.add_argument("--weights", dest='weightsfile', help="模型权重文件", default="D:/SNAL/yolov3.weights", type=str)
    parser.add_argument('--picture', dest='picfile', help='图片文件', default='D:/SNAL/data/125.png', type=str)

    return parser.parse_args()


def expremention(yolo, testdata, num_class, conf_thre, nms_thre, pattern='train'):
    print(f'下面开始{pattern}')
    yolo.eval()
    process = process_tqdm(testdata)
    result = (0, 0)
    for i, (img, y) in enumerate(process):
        # img = cv2.imread(f'D:/data/specialcase/0/9.png')
        process.set_description(f'测试进度')
        img = (np.array(img) * 255)[0].transpose((1, 2, 0))
        img = img.copy()
        x = frame_handle(img, int(args.inp_dim))
        x = x.cuda()
        with torch.no_grad():
            output = yolo(x)
        if pattern == 'train':
            result = scene_eigenvalue(output[1], num_class, conf_thre, nms_thre, img, pattern=pattern,
                                      common_evalue=result, i=i)
        elif pattern == 'test':
            result = scene_eigenvalue(output[1], num_class, conf_thre, nms_thre, img, pattern=pattern,
                                      common_evalue=47332, i=i)
        A = result - 200000
        if A >= 0:
            Iapp = 520 / (1 + math.exp(-A)) - 220
        else:
            Iapp = 40
        MIN_output = Mineuron(Iapp, i)  # 中间神经元电位处理后的输出
        MIN_output = torch.tensor(MIN_output).unsqueeze(0).unsqueeze(0)
        print(result, MIN_output)
    print(f'last_result:{result},MIN:{MIN_output}')


if __name__ == '__main__':
    args = arg_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available()
    # operation = input("输入0进行训练，输入1进行测试：")

    yolo = Darknet(args.cfgfile)
    yolo = yolo.cuda()
    yolo.load_weights(args.weightsfile)
    print('预训练模型加载完成')
    testdata = test_load(args.test_datafile)
    expremention(yolo, testdata, args.num_class, args.conf_thre, args.nms_thre, pattern='test')
