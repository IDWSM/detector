import torch
import torch.nn as nn

from layers import CBR2d, Focus, SPP, BottleneckCSP, Concat

class Detect(nn.Module):
    def __init__(self, nc=2, anchor=()):
        super(Detect, self).__init__()
        self.stride = None
        self.nc = nc  # number of classes
        self.no = nc + 5  # anchor당 출력 수
        self.nl = len(anchor)  # number of detection layers
        self.na = len(anchor[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchor).float().view(self.nl, -1, 2)
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
    def __init__(self, in_channels, out_channels, num_classes, num_anchors):
        super(Model, self).__init__()
        # backbone
        self.focus = Focus(in_channels, 32, 3, 1, 1) # P1
        self.conv_1 = CBR2d(32, 64, 3, 2, 1) # P2
        self.btneckCSP_1 = BottleneckCSP(64, 64, 1)
        self.conv_2 = CBR2d(64, 128, 3, 2, 1) # P3
        self.btneckCSP_2 = BottleneckCSP(128, 128, 3)
        self.conv_3 = CBR2d(128, 256, 3, 2, 1) # P5
        self.btneckCSP_3 = BottleneckCSP(256, 256, 3)
        self.conv_4 = CBR2d(256, 512, 3, 2, 1) # P7
        self.spp = SPP(512, 512, [5, 9, 13])
        # head
        self.btneckCSP_4 = BottleneckCSP(512, 512, 1, shortcut=False)

        self.conv_5 = CBR2d(512, 256, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.concat = Concat(1) # backbone P4
        self.btneckCSP_5 = BottleneckCSP(512, 256, 1, shortcut=False)

        self.conv_6 = CBR2d(256, 128, 1, 1)
        # upsample
        # concat backbone P3
        self.btneckCSP_6 = BottleneckCSP(256, 128, 1, shortcut=False)
        self.conv2d_1 = nn.Conv2d(128, (num_anchors * (num_classes + 5)), 1, 1) # P3

        self.conv_7 = CBR2d(128, 128, 3, 2, 1)
        # concat head P4
        self.btneckCSP_7 = BottleneckCSP(256, 256, 1, shortcut=False)
        self.conv2d_2 = nn.Conv2d(256, (num_anchors * (num_classes + 5)), 1, 1) # P4

        self.conv_8 = CBR2d(256, 256, 3, 2, 1)
        # concat head P5
        self.btneckCSP_8 = BottleneckCSP(512, 512, 1, shortcut=False)
        self.conv2d_3 = nn.Conv2d(512, (num_anchors * (num_classes + 5)), 1, 1) # P5

    def forward(self, x):
        x = self.focus(x)
        x = self.conv_1(x)
        x = self.btneckCSP_1(x)
        x = self.conv_2(x)
        c1 = self.btneckCSP_2(x)
        x = self.conv_3(c1)
        c2 = self.btneckCSP_3(x)
        x = self.conv_4(c2)
        x = self.spp(x)

        x = self.btneckCSP_4(x)
        c3 = self.conv_5(x)
        x = self.upsample(c3)
        x = self.concat((x, c2))
        x = self.btneckCSP_5(x)

        c4 = self.conv_6(x)
        x = self.upsample(c4)
        x = self.concat((x, c1))
        x = self.btneckCSP_6(x)
        p1 = self.conv2d_1(x)

        x = self.conv_7(x)
        x = self.concat(x, c4)
        x = self.btneckCSP_7(x)
        p2 = self.conv2d_2(x)

        x = self.conv_8(x)
        x = self.concat(x, c3)
        x = self.btneckCSP_8(x)
        p3 = self.conv2d_3(x)



