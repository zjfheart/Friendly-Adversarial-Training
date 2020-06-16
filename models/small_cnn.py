from collections import OrderedDict
import torch.nn as nn
import torch
from torch.autograd import Variable

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block1_pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.batchnorm1_2 = nn.BatchNorm2d(64)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.block2_pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.batchnorm2_2 = nn.BatchNorm2d(128)

        self.block3_conv1 = nn.Conv2d(128, 196, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(196, 196, 3, padding=1)
        self.block3_pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm3_1 = nn.BatchNorm2d(196)
        self.batchnorm3_2 = nn.BatchNorm2d(196)

        self.activ = nn.ReLU()

        self.fc1 = nn.Linear(196*4*4,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        #block1
        x = self.block1_conv1(x)
        x = self.batchnorm1_1(x)
        x = self.activ(x)
        x = self.block1_conv2(x)
        x = self.batchnorm1_2(x)
        x = self.activ(x)
        x = self.block1_pool1(x)

        #block2
        x = self.block2_conv1(x)
        x = self.batchnorm2_1(x)
        x = self.activ(x)
        x = self.block2_conv2(x)
        x = self.batchnorm2_2(x)
        x = self.activ(x)
        x = self.block2_pool1(x)
        #block3
        x = self.block3_conv1(x)
        x = self.batchnorm3_1(x)
        x = self.activ(x)
        x = self.block3_conv2(x)
        x = self.batchnorm3_2(x)
        x = self.activ(x)
        x = self.block3_pool1(x)

        x = x.view(-1,196*4*4)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)

        return x

def small_cnn():
    return SmallCNN()
def test():
    net = small_cnn()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
    print(net)