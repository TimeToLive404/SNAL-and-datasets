# 加载truck_car数据集
from torchvision.datasets import ImageFolder
import torchvision
import torch.utils.data as Data


# 训练集加载
def data_load(file):
    # 定义图像增广方式
    train_transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomSizedCrop((1920, 1080), scale=(0.7, 1), ratio=(0.5, 2)),
        torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.1,0.1)),
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
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = ImageFolder(file, transform=test_transforms)

    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    return test_loader

# 测试代码
# filepath = r'D:\data\truck_car'
# traindata = data_load(filepath)
# for i, (x, y) in enumerate(traindata):
#     print(x)
#     print(y)
#     break
