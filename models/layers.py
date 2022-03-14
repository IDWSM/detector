import torch
import torch.nn as nn

# Conv + batch norm + leakyReLU
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=0, group=1, relu=True):
        super(CBR2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                              padding=padding, bias=False, groups=group)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

# short-cut Connection
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, g=1, e=0.5):
        """
        :param in_channels: input channels of number
        :param out_channels: output channels of number
        :param shortcut: connection x
        :param g: groups
        :param e: expansion
        """
        super(Bottleneck, self).__init__()
        channels = int(out_channels * e)
        self.conv1 = CBR2d(in_channels, channels, kernel=1, stride=1)
        self.conv2 = CBR2d(channels, out_channels, kernel=3, stride=1, padding=1, group=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self, in_channels, out_channels, num=1, shortcut=True, g=1, e=0.5):
        """
        :param in_channels: input channels of number
        :param out_channels: output channels of number
        :param num: number of repeat
        :param shortcut: using shortcut connection
        :param g: groups
        :param e: expansion
        """
        super(BottleneckCSP, self).__init__()
        channels = int(out_channels * e)
        self.conv1 = CBR2d(in_channels, channels, kernel=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.conv4 = CBR2d(2 * channels, out_channels, kernel=1, stride=1)
        self.bn = nn.BatchNorm2d(2 * channels)
        self.bn.eps = 1e-4
        self.bn.momentum = 0.03
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(channels, channels, shortcut, g, e=1.0) for i in range(num)])

    def forward(self, x):
        x1 = self.conv3(self.m(self.conv1(x)))
        x2 = self.conv2(x)
        return self.conv4(self.lrelu(self.bn(torch.cat((x1, x2), dim=1))))

# Spatial Pyramid Pooling
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(5, 9, 13)):
        super(SPP, self).__init__()
        self.conv1 = CBR2d(in_channels, out_channels // 2, kernel=1, stride=1)
        self.max = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel])
        self.conv2 = CBR2d((out_channels // 2) * (len(kernel) + 1), out_channels, kernel=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(torch.cat([x] + [maxpool(x) for maxpool in self.max], 1))

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padd=None, g=1, relu=True):
        super(Focus, self).__init__()
        self.conv = CBR2d(in_channels * 4, out_channels, kernel=kernel, stride=stride, padding=padd, group=g, relu=relu)

    def forward(self, x): # x(batch, channels, width, height) -> y(batch, 4*channels, width / 2, height / 2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
