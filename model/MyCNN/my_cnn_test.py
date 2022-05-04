import torch
import torch.nn as nn


class My_CNN_test(nn.Module):
    def __init__(self):
        super(My_CNN_test, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(36864, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
