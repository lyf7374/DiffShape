import torch
import torch.nn as nn


def normalization(planes, norm="bn"):
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(4, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0, norm="bn", first=False, padding=0):
        super().__init__()
        self.first = first
        self.dropout = dropout
        self.maxpool = nn.MaxPool3d(2, 2, padding=padding)
        self.relu = nn.LeakyReLU(0.2, inplace=False)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = normalization(planes, norm)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = normalization(planes, norm)
        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        if self.dropout > 0:
            x = torch.nn.functional.dropout3d(x, self.dropout)
        y = self.relu(self.bn2(self.conv2(x)))
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)
