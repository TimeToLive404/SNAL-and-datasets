import torch, torchvision

model = torchvision.models.vgg
model = torchvision.models.vgg16()
from torchsummary import summary

summary(model.cuda(), (3, 224, 224))

