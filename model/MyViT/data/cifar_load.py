# 加载CIFAR-10数据集
import pickle as pkl
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import numpy as np


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pkl.load(f, encoding='bytes')
    return dict


# 本地文件的方式加载数据集
def data_load(files):
    x_train = np.empty(shape=[0, 3072])
    y_train = []
    for file in files:
        train_dict = unpickle(file)
        train_data = train_dict.get(b'data', 'no such key in train_dict.')
        labels = train_dict.get(b'labels', 'no such key in train_dict.')
        x_train = np.append(x_train, train_data, axis=0)
        y_train = np.append(y_train, labels)
    return zip(x_train, y_train)


# dataloader的方式加载数据集
cifar10 = torchvision.datasets.CIFAR10(
    root=r'D:\MyCNN\data',
    train=True,
    download=True
)
# train_data_loader = DataLoader(cifar10, batch_size=16, shuffle=False, num_workers=4)

# 测试代码
'''
filepath = r'D:\MyCNN\data\cifar-10-batches-py\data_batch_1'
res = unpickle(filepath)
print(res.keys())

print(res[b'data'][0])
# res[b'data'][0] 3*32*32
img_arr = res[b'data'][0].reshape((3, 32, 32))
img_arr = img_arr.transpose((1, 2, 0))
print(img_arr)
img = Image.fromarray(img_arr)
img.show()
'''
files = [r'D:\MyCNN\data\cifar-10-batches-py\data_batch_1']
train_data_loader = data_load(files)
print('train_data_loader:', train_data_loader)
for i, (x, y) in enumerate(train_data_loader):
    # x是3072的arroy
    print("x:", x)
    img_arr = x.reshape((3, 32, 32))
    img_arr = img_arr.transpose((1, 2, 0))
    img = Image.fromarray(np.uint8(img_arr))
    img.show()
    break
