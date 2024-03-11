# Miguel Hernández University of Elche
# Institute for Engineering Research of Elche (I3E)
# Automation, Robotics and Computer Vision lab (ARCV)
# Author: Juan José Cabrera Mora

import torch.nn as nn
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F

class MinkNeXtBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(MinkNeXtBlock, self).__init__()
        assert dimension > 0

        self.conv = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, dimension=dimension)
        self.norm = LayerNorm(planes, eps=1e-6)
        self.pwconv1 = ME.MinkowskiConvolution(planes, 4 * planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm1 = LayerNorm(4*planes, eps=1e-6)
        self.pwconv2 = ME.MinkowskiConvolution(4 * planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = LayerNorm(planes, eps=1e-6)
        self.gelu = ME.MinkowskiGELU()
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.norm(out)
        out = self.gelu(out)

        out = self.pwconv1(out)
        out = self.norm1(out)
        out = self.gelu(out)

        out = self.pwconv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.gelu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, batch):
        x = batch.F
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        out = ME.SparseTensor(x, coordinate_map_key=batch.coordinate_map_key,
                              coordinate_manager=batch.coordinate_manager)
        return out
