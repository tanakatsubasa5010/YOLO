import torch
from torch import nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention Mechanism. https://arxiv.org/pdf/2305.13563"""

    def __init__(self, c1, g=32):
        """Initialize EMA with given input channel (c1) and group size."""
        super().__init__()
        assert c1 % g == 0, "c1 must be divisible by g"
        self.g = g
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c1 // self.g, c1 // self.g)
        self.conv1x1 = nn.Conv2d(c1 // self.g, c1 // self.g, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(c1 // self.g, c1 // self.g, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        N, C, H, W = x.shape
        group_x = x.reshape(N * self.g, -1, H, W)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [H, W], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(N * self.g, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(N * self.g, C // self.g, -1)
        x21 = self.softmax(self.agp(x2).reshape(N * self.g, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(N * self.g, C // self.g, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(N * self.g, 1, H, W)
        return (group_x * weights.sigmoid()).reshape(N, C, H, W)