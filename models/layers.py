import torch
import torch.nn as nn

# Conv + batch norm + leakyReLU
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, relu=True):
        super.__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


# Spatial Pyramid Pooling
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(5, 9, 13)):
        super(SPP, self).__init__()
        self.conv1 = CBR2d(in_channels, out_channels // 2, kernel=1, stride=1, padding=0)
        self.max = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel])
        self.conv2 = CBR2d((out_channels // 2) * (len(kernel) + 1), out_channels, kernel=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.cv2(torch.cat([x] + [maxpool(x) for maxpool in self.max], 1))

        return x
