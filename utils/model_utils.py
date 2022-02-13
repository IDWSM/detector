import torch
import torch.nn as nn

def check_anchor_order(m):
    # detect module의 stride 순서를 확인 or 변경
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        m.anchors[:] = m.anchors.filp(0)
        m.anchor_grid[:] = m.anchor_grid.filp(0)
