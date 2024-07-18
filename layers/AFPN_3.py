import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from utils.tools import Spatial_Dropout

def BasicConv(filter_in, filter_out, kernel_size, stride=1, padding=None):
    if not padding:
        padding = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        padding = padding
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv1d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
        ("bn", nn.BatchNorm1d(filter_out)),
        ("tanh", nn.Tanh()),
    ]))


class ASFF_2(nn.Module):
    def __init__(self, args, inter_dim=512, level=0, channel=[128, 256], len=[48, 24]):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        self.compress_c = args.compress_c

        self.weight_level_1 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)
        self.weight_level_2 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)

        self.weight_levels = nn.Conv1d(self.compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, kernel_size=3, stride=1, padding=1)
        self.upsample = Upsample(channel[1], channel[0], len[0], len[1])
        self.downsample = Downsample(channel[0], channel[1], len[1], len[0])
        self.level = level

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :] + \
                            input2 * levels_weight[:, 1:2, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, args, inter_dim=512, level=0, channel=[64, 128, 256], len=[48,24,12]):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        self.compress_c = args.compress_c

        self.weight_level_1 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)
        self.weight_level_2 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)
        self.weight_level_3 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)

        self.weight_levels = nn.Conv1d(self.compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, kernel_size=3, stride=1, padding=1)

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
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :] + \
                            input2 * levels_weight[:, 1:2, :] + \
                            input3 * levels_weight[:, 2:, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_4(nn.Module):
    def __init__(self, args, inter_dim=512, level=0, channel=[16, 64, 128, 256], len=[48,24,12,6]):
        super(ASFF_4, self).__init__()

        self.inter_dim = inter_dim
        self.compress_c = args.compress_c

        self.weight_level_1 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)
        self.weight_level_2 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)
        self.weight_level_3 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)
        self.weight_level_4 = BasicConv(self.inter_dim, self.compress_c, kernel_size=1, stride=1)

        self.weight_levels = nn.Conv1d(self.compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, kernel_size=3, stride=1, padding=1)

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

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        level_4_weight_v = self.weight_level_4(input4)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :] + \
                            input2 * levels_weight[:, 1:2, :] + \
                            input3 * levels_weight[:, 2:3, :] + \
                            input4 * levels_weight[:, 3:, :]

        out = self.conv(fused_out_reduced)

        return out



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

