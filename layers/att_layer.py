import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from utils.tools import Spatial_Dropout
from math import sqrt
class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, dropout=0.05):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = Spatial_Dropout(dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, out_d, n_heads, dropout, d_keys=None, d_values=None, mix=True):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, dropout=0.05)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, out_d)
        self.n_heads = n_heads
        self.mix = mix
        self.initialize_weight(self.query_projection)
        self.initialize_weight(self.key_projection)
        self.initialize_weight(self.value_projection)
        self.initialize_weight(self.out_projection)


    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out)

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2): #16
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print("y.shape:", y.shape)
        y = self.fc(y).view(b, c, 1)
        # print("y.shape:", y.shape)
        return x * y


class SENet(nn.Module):
    def __init__(self, in_channels, num_classes=1000, reduction=2): #16
        super(SENet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(64, reduction=reduction)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # print("conv1(x).shape:", x.shape)
        x = self.bn1(x)
        # print("bn1(x).shape:", x.shape)
        x = self.relu(x)
        # print("relu(x).shape:", x.shape)
        x = self.se(x)
        # print("se(x).shape:", x.shape)
        x = x.view(x.size(0), -1, x.size(1))
        # print("(x).shape:", x.shape)
        x = self.fc(x)
        return x


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#         return x * self.sigmoid(y)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         y = torch.cat([max_pool, avg_pool], dim=1)
#         y = self.conv(y)
#         return x * self.sigmoid(y)
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
#         self.spatial_attention = SpatialAttention()
#
#     def forward(self, x):
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x

#空间注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(x.shape)
        y = self.avg_pool(x).view(b, c)
        # print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
