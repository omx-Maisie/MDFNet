# 原序列和分解分量分别编码、预测，最后相加 直接融合
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.embed import DataEmbedding
from layers.ResNet50 import ResNet50_backbone
from layers.Progressive_Fusion import ASFF_2, ASFF_3, ASFF_4, Upsample
from utils.tools import Spatial_Dropout
from layers.att_layer import AttentionLayer, SENet, CBAM

class Model(nn.Module):
    def __init__(self, args, conv_kernel=[12, 6], isometric_kernel=[9, 17], dropout=0.05):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.feature_size = args.feature_size
        self.conv_kernel = args.conv_kernel
        self.isometric_kernel = args.isometric_kernel
        self.Channel = args.channel
        self.c_out = args.c_out
        self.dropout = args.dropout
        self.len = args.len

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # PART:编码
        self.enc_embedding = DataEmbedding(self.feature_size, args.d_model, args.embed, args.freq, args.dropout).to(self.device)

        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels=args.d_model, out_channels=k, kernel_size=i, padding=i//2, stride=i)
            for i, k in zip(self.conv_kernel, self.Channel)]).to(self.device)

        # isometric convolution
        self.isometric_conv = nn.ModuleList([
            nn.Conv1d(in_channels=k, out_channels=k, kernel_size=i, padding=0, stride=1)
            for i, k in zip(self.isometric_kernel, self.Channel)]).to(self.device)

        # upsampling convolution
        self.conv_trans = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=k, out_channels=args.d_model, kernel_size=i, padding=0, stride=i)
            for i, k in zip(self.conv_kernel, self.Channel)]).to(self.device)

        self.norm_list = nn.ModuleList([
            torch.nn.LayerNorm(k)
            for k in self.Channel]).to(self.device)

        self.merge = torch.nn.Conv2d(in_channels=args.d_model, out_channels=args.d_model, kernel_size=(len(self.conv_kernel), 1)).to(self.device)

        self.fnn = FeedForwardNetwork(args.d_model, args.d_model*4, dropout).to(self.device)
        self.fnn_norm = torch.nn.LayerNorm(args.d_model).to(self.device)

        self.norm = torch.nn.LayerNorm(args.d_model).to(self.device)
        self.norm_imf = torch.nn.LayerNorm(1).to(self.device)
        self.act = torch.nn.Tanh().to(self.device)
        self.drop = torch.nn.Dropout(self.dropout).to(self.device)
        self.spatial_drop = Spatial_Dropout(self.dropout).to(self.device)
        self.projection = nn.Linear(args.d_model, self.c_out).to(self.device)
        self.attn = AttentionLayer(args.d_model, self.c_out, n_heads=args.n_heads, dropout=args.dropout).to(self.device)
        # self.att = CBAM(args.d_model).to(self.device)
    def conv_isometric(self, input, conv1d, isometric, norm):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.spatial_drop(self.act(conv1d(x))).to(self.device)
        # x = x1.to(self.device)
        # print("conv1d:", x.shape)
        # # isometric convolution
        # zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1)).to(self.device)
        # # print("zeros:", zeros.shape)
        # x = torch.cat((zeros, x), dim=-1).to(self.device)
        # # print("isometric前:", x.shape)
        # x = self.spatial_drop(self.act(isometric(x))).to(self.device)
        # # print("isometric后:", x.shape)
        # x = norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1).to(self.device)
        x = norm(x1.permute(0, 2, 1)).permute(0, 2, 1).to(self.device)
        # print("norm后:", x.shape)
        return x
    def conv_trans_conv(self, input, conv1d_trans, x0):
        # print("x0.shape:", x0.shape)
        len = x0.shape[1]
        x = input
        # upsampling convolution
        x = self.spatial_drop(self.act(conv1d_trans(x))).to(self.device)
        x = x[:, :, :len].to(self.device)  # truncate
        x = self.norm(x.permute(0, 2, 1) + x0).to(self.device)
        return x

    def forward(self, batch_x, batch_x_mark):
        # batch_x = batch_x[:, :, 1:]
        # SECTION:编码
        zeros = torch.zeros([batch_x.shape[0], self.pred_len, batch_x.shape[2]]).to(self.device)
        batch_x_enc = torch.cat([batch_x[:, -self.seq_len:, :], zeros], dim=1)
        zeros_0 = torch.zeros([batch_x_mark.shape[0], self.pred_len, batch_x_mark.shape[2]]).to(self.device)
        batch_x_mark_enc = torch.cat([batch_x_mark, zeros_0], dim=1)
        x_emb = self.enc_embedding(batch_x_enc, batch_x_mark_enc)

        # SECTION: 多尺度特征提取
        multi = []
        for i in range(len(self.conv_kernel)):
            temp_out = self.conv_isometric(x_emb, self.conv[i], self.isometric_conv[i], self.norm_list[i])
            # print("temp_out.shape:", temp_out.shape)
            multi.append(temp_out)

        x_1 = multi[0]
        x_2 = multi[1]
        x_3 = multi[2]
        x_4 = multi[3]

        # SECTION: 上采样
        x_1 = self.conv_trans_conv(x_1, self.conv_trans[0], x_emb).to(self.device)
        x_2 = self.conv_trans_conv(x_2, self.conv_trans[1], x_emb).to(self.device)
        x_3 = self.conv_trans_conv(x_3, self.conv_trans[2], x_emb).to(self.device)
        x_4 = self.conv_trans_conv(x_4, self.conv_trans[3], x_emb).to(self.device)

        x_new = []
        x_new.append(x_1)
        x_new.append(x_2)
        x_new.append(x_3)
        x_new.append(x_4)
        # print(x_1.shape)
        # print(x_2.shape)
        # print(x_3.shape)
        # print(x_4.shape)
        mg = torch.tensor([]).to(self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, x_new[i].unsqueeze(1)), dim=1)

        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
        mg = self.attn(mg, mg,mg)
        return mg[:, -self.pred_len:, :]

        # # print(mg.shape)
        # mg = self.att(mg.permute(0, 3, 1, 2))
        # # print(mg.shape)
        # mg = self.merge(mg).squeeze(-2).permute(0, 2, 1)
        # mg_out = self.fnn_norm(mg + self.fnn(mg))
        # proj_out = self.projection(mg_out)
        # # print("proj_out.shape:", proj_out.shape)
        # sum_out = torch.sum(proj_out, dim=2)
        # sum_out = sum_out.unsqueeze(dim=2)
        #
        # return sum_out[:, -self.pred_len:, :]

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight).to(self.device)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0).to(self.device)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer1 = nn.Linear(hidden_size, filter_size).to(self.device)
        self.relu = nn.ReLU().to(self.device)

        # self.dropout = nn.Dropout(dropout_rate).to(self.device)
        self.spatial_dropout = Spatial_Dropout(dropout_rate).to(self.device)
        self.layer2 = nn.Linear(filter_size, hidden_size).to(self.device)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.spatial_dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight).to(self.device)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0).to(self.device)