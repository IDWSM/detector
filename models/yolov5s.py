import math
import torch
import torch.nn as nn

from models.layers import CBR2d, Focus, SPP, BottleneckCSP, Concat
from utils.model_utils import check_anchor_order

anchor = [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]

class Detect(nn.Module):
    def __init__(self, nc=3, anchors=()):
        super(Detect, self).__init__()
        self.stride = None
        self.nc = nc  # number of classes
        self.no = nc + 5  # anchor당 출력 수
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.export = False

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            batch, _, ny, nx = x[i].shape
            x[i] = x[i].view(batch, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(batch, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=len(anchor[0])//2):
        super(Model, self).__init__()
        self.save = list()
        # backbone
        self.focus = Focus(in_channels, 32, 3, 1, 1)  # P1
        self.conv_1 = CBR2d(32, 64, 3, 2, 1)  # P2
        self.btneckCSP_1 = BottleneckCSP(64, 64, 1)
        self.conv_2 = CBR2d(64, 128, 3, 2, 1)  # P3
        self.btneckCSP_2 = BottleneckCSP(128, 128, 3)
        self.conv_3 = CBR2d(128, 256, 3, 2, 1)  # P5
        self.btneckCSP_3 = BottleneckCSP(256, 256, 3)
        self.conv_4 = CBR2d(256, 512, 3, 2, 1)  # P7
        self.spp = SPP(512, 512, [5, 9, 13])
        # head
        self.btneckCSP_4 = BottleneckCSP(512, 512, 1, shortcut=False)

        self.conv_5 = CBR2d(512, 256, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.concat = Concat(1)  # backbone P4
        self.btneckCSP_5 = BottleneckCSP(512, 256, 1, shortcut=False)

        self.conv_6 = CBR2d(256, 128, 1, 1)
        # upsample
        # concat backbone P3
        self.btneckCSP_6 = BottleneckCSP(256, 128, 1, shortcut=False)
        self.conv2d_1 = nn.Conv2d(128, (num_anchors * (num_classes + 5)), 1, 1)  # P3

        self.conv_7 = CBR2d(128, 128, 3, 2, 1)
        # concat head P4
        self.btneckCSP_7 = BottleneckCSP(256, 256, 1, shortcut=False)
        self.conv2d_2 = nn.Conv2d(256, (num_anchors * (num_classes + 5)), 1, 1)  # P4

        self.conv_8 = CBR2d(256, 256, 3, 2, 1)
        # concat head P5
        self.btneckCSP_8 = BottleneckCSP(512, 512, 1, shortcut=False)
        self.conv2d_3 = nn.Conv2d(512, (num_anchors * (num_classes + 5)), 1, 1)  # P5

        self.detect = Detect(nc=3, anchors=anchor)
        self.detect.stride = torch.tensor([128 / x.shape[-2] for x in
                                           self.forward(torch.zeros(1, in_channels, 128, 128))])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        check_anchor_order(self.detect)
        self.stride = self.detect.stride
        self._initialize_biases()

    def _initialize_biases(self, cf=None):
        _from = [self.conv2d_1, self.conv2d_2, self.conv2d_3]
        for mi, s in zip(_from, self.detect.stride):
            b = mi.bias.view(self.detect.na, -1)  # conv2d.bias(255) to (3, 85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.detect.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        x = self.focus(x)
        x = self.conv_1(x)
        x = self.btneckCSP_1(x)
        x = self.conv_2(x)
        x = self.btneckCSP_2(x)
        self.save.append(x)
        x = self.conv_3(x)
        x = self.btneckCSP_3(x)
        self.save.append(x)
        x = self.conv_4(x)
        x = self.spp(x)

        x = self.btneckCSP_4(x)
        x = self.conv_5(x)
        self.save.append(x)
        x = self.upsample(x)
        x = self.concat((x, self.save[-2]))
        x = self.btneckCSP_5(x)

        x = self.conv_6(x)
        self.save.append(x)
        x = self.upsample(x)
        x = self.concat((x, self.save[-4]))
        x = self.btneckCSP_6(x)
        p1 = self.conv2d_1(x)

        x = self.conv_7(x)
        x = self.concat((x, self.save[-1]))
        x = self.btneckCSP_7(x)
        p2 = self.conv2d_2(x)

        x = self.conv_8(x)
        x = self.concat((x, self.save[-2]))
        x = self.btneckCSP_8(x)
        p3 = self.conv2d_3(x)

        return self.detect([p1, p2, p3])

model = Model(num_classes=3, in_channels=3)

y = model(torch.rand(1, 3, 640, 640))