from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision

file = 'D:/data/truck_car3/'
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomSizedCrop((1920, 1080), scale=(0.7, 1), ratio=(0.5, 2)),
    torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.1, 0.1)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_data = ImageFolder(file, transform=train_transforms)

to_img = T.ToPILImage()
# 0.2和0.4是标准差和均值的近似
a = to_img(train_data[0][0])
# a = to_img(train_data[0][0])
plt.imshow(a)
plt.axis('off')
plt.show()
