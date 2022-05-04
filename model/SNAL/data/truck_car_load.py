# 加载truck_car数据集
from torchvision.datasets import ImageFolder
import torchvision
import torch.utils.data as Data


# 训练集加载
def data_load(file):
    # 定义图像增广方式
    train_transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomSizedCrop((1920, 1080), scale=(0.7, 1), ratio=(0.5, 2)),
        torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.1, 0.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 加载为数据集
    train_data = ImageFolder(file, transform=train_transforms)

    train_loader = Data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

    return train_loader


# 测试集加载
def test_load(file):
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = ImageFolder(file, transform=test_transforms)

    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    return test_loader


def add_y_train(func):
    # 按顺序设置标签
    y = [1 for _ in range(250)]
    y[120] = 0
    y[121] = 0
    y[122] = 0

    def inter(file):
        x = func(file)
        xy = zip(x, y)
        return xy

    return inter


@add_y_train
def seq_train_dataload(file):  # 加载顺序数据集，一个文件夹中的所有图片，以及它们的类别列表
    # 使用dataloader加载所有图片，但是不用其标签
    train_transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomSizedCrop((1920, 1080), scale=(0.7, 1), ratio=(0.5, 2)),
        torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.1, 0.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = ImageFolder(file, transform=train_transforms)
    train_loader = Data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
    return train_loader


def add_y_test(func):
    y = [1 for _ in range(200)]
    y[0:21] = [0 for _ in range(21)]
    y[115:138] = [0 for _ in range(23)]
    y[197:200] = [0 for _ in range(3)]

    def inter(file):
        x = func(file)
        xy = zip(x, y)
        return xy

    return inter


@add_y_test
def seq_test_dataload(file):  # 加载顺序数据集，一个文件夹中的所有图片，以及它们的类别列表
    # 使用dataloader加载所有图片，但是不用其标签
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = ImageFolder(file, transform=test_transforms)
    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    return test_loader


# 测试代码
if __name__ == '__main__':
    res = seq_train_dataload(r'D:\data\seq_train/')
    for i, ((x, _), y) in enumerate(res):
        print(i, x, y)
        break

# filepath = r'D:\data\truck_car'
# traindata = data_load(filepath)
# for i, (x, y) in enumerate(traindata):
#     print(x)
#     print(y)
#     break
