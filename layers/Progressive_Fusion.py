import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from utils.tools import Spatial_Dropout
from math import sqrt
from layers.att_layer import AttentionLayer, SENet, CBAM


def BasicConv(filter_in, filter_out, kernel_size, stride=1, padding=None):
    if not padding:
        padding = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        padding = padding
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv1d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
        ("bn", nn.BatchNorm1d(filter_out)),
        ("tanh", nn.Tanh()),
        # ("dropout", Spatial_Dropout(0.05))
    ]))

class ASFF_2(nn.Module):
    def __init__(self, args, inter_dim=512, level=0, channel=[128, 256], len=[48, 24]):
        super(ASFF_2, self).__init__()
        self.att = SENet(inter_dim*2, inter_dim)
        self.upsample = Upsample(channel[1], channel[0], len[0], len[1])
        self.downsample = Downsample(channel[0], channel[1], len[1], len[0])
        self.level = level

    def forward(self, x):
        input1, input2 = x
        # print('input1.shape:', input1.shape)
        # print('input2.shape:', input2.shape)
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)
        x = torch.cat((input1, input2), 1)
        # print('input1.shape:', input1.shape)
        # print("before att x.shape", x.shape)
        out = self.att(x)

        # print('input2.shape:', input2.shape)
        # print(out.shape)

        return out.permute(0, 2, 1)


class ASFF_3(nn.Module):
    def __init__(self, args, inter_dim=512, level=0, channel=[64, 128, 256], len=[48,24,12]):
        super(ASFF_3, self).__init__()

        self.att = SENet(inter_dim*3, inter_dim)

        self.level = level
        if self.level == 0:
            self.upsample4_0 = Upsample(channel[2], channel[0], len[0], len[2])
            self.upsample2_0 = Upsample(channel[1], channel[0], len[0], len[1])
        elif self.level == 1:
            self.upsample2_1 = Upsample(channel[2], channel[1], len[1], len[2])
            self.downsample2_1 = Downsample(channel[0], channel[1], len[1], len[0])
        elif self.level == 2:
            self.downsample2_2 = Downsample(channel[1], channel[2], len[2], len[1])
            self.downsample4_2 = Downsample(channel[0], channel[2], len[2], len[0])

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2_0(input2)
            input3 = self.upsample4_0(input3)
        elif self.level == 1:
            input3 = self.upsample2_1(input3)
            input1 = self.downsample2_1(input1)
        elif self.level == 2:
            input1 = self.downsample4_2(input1)
            input2 = self.downsample2_2(input2)
        x = torch.cat((input1, input2, input3), 1)
        out = self.att(x)
        return out.permute(0, 2, 1)


class ASFF_4(nn.Module):
    def __init__(self, args, inter_dim=512, level=0, channel=[16, 64, 128, 256], len=[48,24,12,6]):
        super(ASFF_4, self).__init__()

        self.att = SENet(inter_dim * 4, inter_dim)

        self.level = level
        if self.level == 0:
            self.upsample2_0 = Upsample(channel[1], channel[0], len[0], len[1])
            self.upsample4_0 = Upsample(channel[2], channel[0], len[0], len[2])
            self.upsample8_0 = Upsample(channel[3], channel[0], len[0], len[3])
        elif self.level == 1:
            self.downsample2_1 = Downsample(channel[0], channel[1], len[1], len[0])
            self.upsample2_1 = Upsample(channel[2], channel[1], len[1], len[2])
            self.upsample4_1 = Upsample(channel[3], channel[1], len[1], len[3])
        elif self.level == 2:
            self.downsample4_2 = Downsample(channel[0], channel[2], len[2], len[0])
            self.downsample2_2 = Downsample(channel[1], channel[2], len[2], len[1])
            self.upsample2_2 = Upsample(channel[3], channel[2], len[2], len[3])
        elif self.level == 3:
            self.downsample8_3 = Downsample(channel[0], channel[3], len[3], len[0])
            self.downsample4_3 = Downsample(channel[1], channel[3], len[3], len[1])
            self.downsample2_3 = Downsample(channel[2], channel[3], len[3], len[2])

    def forward(self, x):
        input1, input2, input3, input4 = x
        if self.level == 0:
            input2 = self.upsample2_0(input2)
            input3 = self.upsample4_0(input3)
            input4 = self.upsample8_0(input4)
        elif self.level == 1:
            input1 = self.downsample2_1(input1)
            input3 = self.upsample2_1(input3)
            input4 = self.upsample4_1(input4)
        elif self.level == 2:
            input1 = self.downsample4_2(input1)
            input2 = self.downsample2_2(input2)
            input4 = self.upsample2_2(input4)
        elif self.level == 3:
            input1 = self.downsample8_3(input1)
            input2 = self.downsample4_3(input2)
            input3 = self.downsample2_3(input3)

        x = torch.cat((input1, input2, input3, input4), 1)
        out = self.att(x)

        return out.permute(0, 2, 1)

class ASFF_5(nn.Module):
    def __init__(self, args, inter_dim=512, level=0, channel=[8, 16, 64, 128, 256], len=[96, 48,24,12,6]):
        super(ASFF_5, self).__init__()

        self.att = SENet(inter_dim * 5, inter_dim)

        self.level = level
        if self.level == 0:
            self.upsample2_0 = Upsample(channel[1], channel[0], len[0], len[1])
            self.upsample4_0 = Upsample(channel[2], channel[0], len[0], len[2])
            self.upsample8_0 = Upsample(channel[3], channel[0], len[0], len[3])
            self.upsample16_0 = Upsample(channel[4], channel[0], len[0], len[4])
        elif self.level == 1:
            self.downsample2_1 = Downsample(channel[0], channel[1], len[1], len[0])
            self.upsample2_1 = Upsample(channel[2], channel[1], len[1], len[2])
            self.upsample4_1 = Upsample(channel[3], channel[1], len[1], len[3])
            self.upsample8_1 = Upsample(channel[4], channel[1], len[1], len[4])
        elif self.level == 2:
            self.downsample4_2 = Downsample(channel[0], channel[2], len[2], len[0])
            self.downsample2_2 = Downsample(channel[1], channel[2], len[2], len[1])
            self.upsample2_2 = Upsample(channel[3], channel[2], len[2], len[3])
            self.upsample4_2 = Upsample(channel[4], channel[2], len[2], len[4])
        elif self.level == 3:
            self.downsample8_3 = Downsample(channel[0], channel[3], len[3], len[0])
            self.downsample4_3 = Downsample(channel[1], channel[3], len[3], len[1])
            self.downsample2_3 = Downsample(channel[2], channel[3], len[3], len[2])
            self.upsample2_3 = Upsample(channel[4], channel[3], len[3], len[4])
        elif self.level == 4:
            self.downsample16_4 = Downsample(channel[0], channel[4], len[4], len[0])
            self.downsample8_4 = Downsample(channel[1], channel[4], len[4], len[1])
            self.downsample4_4 = Downsample(channel[2], channel[4], len[4], len[2])
            self.downsample2_4 = Downsample(channel[3], channel[4], len[4], len[3])


    def forward(self, x):
        input1, input2, input3, input4, input5 = x
        if self.level == 0:
            input2 = self.upsample2_0(input2)
            input3 = self.upsample4_0(input3)
            input4 = self.upsample8_0(input4)
            input5 = self.upsample16_0(input5)
        elif self.level == 1:
            input1 = self.downsample2_1(input1)
            input3 = self.upsample2_1(input3)
            input4 = self.upsample4_1(input4)
            input5 = self.upsample8_1(input5)
        elif self.level == 2:
            input1 = self.downsample4_2(input1)
            input2 = self.downsample2_2(input2)
            input4 = self.upsample2_2(input4)
            input5 = self.upsample4_2(input5)
        elif self.level == 3:
            input1 = self.downsample8_3(input1)
            input2 = self.downsample4_3(input2)
            input3 = self.downsample2_3(input3)
            input5 = self.upsample2_3(input5)
        elif self.level == 4:
            input1 = self.downsample16_4(input1)
            input2 = self.downsample8_4(input2)
            input3 = self.downsample4_4(input3)
            input4 = self.downsample2_4(input4)

        x = torch.cat((input1, input2, input3, input4, input5), 1)
        out = self.att(x)

        return out.permute(0, 2, 1)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, len1, len2):
        super(Upsample, self).__init__()
        self.conv = BasicConv(in_channels, out_channels, kernel_size=1)
        self.layer = nn.Linear(len2, len1)
        self.relu = nn.ReLU()
        self.dropout = Spatial_Dropout(0.05)
        self.initialize_weight(self.layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, len1, len2):
        super(Downsample, self).__init__()
        self.conv = BasicConv(in_channels, out_channels, kernel_size=1)
        self.layer = nn.Linear(len2, len1)
        self.relu = nn.ReLU()
        self.dropout = Spatial_Dropout(0.05)
        # self.layer2 = nn.Linear(filter_size, hidden_size)
        # self.initialize_weight(self.layer1)
        # self.initialize_weight(self.layer2)
        self.initialize_weight(self.layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        return x
    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)