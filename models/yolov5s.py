import torch
import torch.nn as nn

from layers import CBR2d, Focus, SPP, BottleneckCSP, Bottleneck

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(Model, self).__init__()
        # backbone
        self.focus = Focus(in_channels, 32, 3, 1, 1)
        self.conv1 = CBR2d(32, 64, 3, 2, 1)
        self.btneckCSP1 = BottleneckCSP(64, 64, 1)
        self.conv2 = CBR2d(64, 128, 3, 2, 1)
        self.btneckCSP2 = BottleneckCSP(128, 128, 3)
        self.conv3 = CBR2d(128, 256, 3, 2, 1)
        self.btneckCSP3 = BottleneckCSP(256, 256, 3)
        self.conv4 = CBR2d(256, 512, 3, 2, 1)
        self.spp = SPP(512, 512, [5, 9, 13])

    # def forward(self, x):
