import torch
import torch.nn as nn

# Conv + batch norm + leakyReLU
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=0, group=1,relu=True):
        super.__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False, groups=group)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

# short-cut Connection
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, g=1):
        super(Bottleneck, self).__init__()
        self.conv1 = CBR2d(in_channels, in_channels // 2, stride=1, kernel=1)
        self.conv2 = CBR2d(in_channels // 2, out_channels, stride=1, kernel=3, padding=1, group=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

# Spatial Pyramid Pooling
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(5, 9, 13)):
        super(SPP, self).__init__()
        self.conv1 = CBR2d(in_channels, out_channels // 2, kernel=1, stride=1)
        self.max = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel])
        self.conv2 = CBR2d((out_channels // 2) * (len(kernel) + 1), out_channels, kernel=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.cv2(torch.cat([x] + [maxpool(x) for maxpool in self.max], 1))

        return x
