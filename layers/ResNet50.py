import torch
import torch.nn as nn
from torch.nn import functional as F
from layers.Progressive_Fusion import ASFF_2, ASFF_3, ASFF_4, Upsample, Downsample


class ResNet50BasicBlock(nn.Module):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
        super(ResNet50BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm1d(outs[0])
        self.conv2 = nn.Conv1d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
        self.bn2 = nn.BatchNorm1d(outs[1])
        self.conv3 = nn.Conv1d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
        self.bn3 = nn.BatchNorm1d(outs[2])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + x)


class ResNet50DownBlock(nn.Module):
    def __init__(self, in_channel, outs, kernel_size, stride, padding):
        super(ResNet50DownBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm1d(outs[0])
        self.conv2 = nn.Conv1d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm1d(outs[1])
        self.conv3 = nn.Conv1d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm1d(outs[2])

        self.extra = nn.Sequential(
            nn.Conv1d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
            nn.BatchNorm1d(outs[2])
        )

    def forward(self, x):
        x_shortcut = self.extra(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(x_shortcut + out)


class ResNet50_backbone(nn.Module):
    def __init__(self, C):
        super(ResNet50_backbone, self).__init__()
        self.conv1 = nn.Conv1d(C, 64, kernel_size=7, stride=1, padding=3)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.layer1 = nn.Sequential(
            ResNet50DownBlock(64, outs=[64, 64, 128], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(128, outs=[64, 64, 128], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(128, outs=[64, 64, 128], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        )

        self.layer2 = nn.Sequential(
            ResNet50DownBlock(128, outs=[128, 128, 256], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[128, 128, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[128, 128, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50DownBlock(256, outs=[128, 128, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )

        self.layer3 = nn.Sequential(
            ResNet50DownBlock(256, outs=[256, 256, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[256, 256, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[256, 256, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            ResNet50DownBlock(512, outs=[256, 256, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(512, outs=[256, 256, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(512, outs=[256, 256, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        self.layer4 = nn.Sequential(
            ResNet50DownBlock(512, outs=[512, 512, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[512, 512, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            ResNet50DownBlock(1024, outs=[512, 512, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1, ceil_mode=False)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(1024, 10)
        # 使用卷积代替全连接
        self.conv11 = nn.Conv1d(1024, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out0 = out
        out = self.layer1(out)
        out1 = out
        out = self.layer2(out)
        out2 = out
        out = self.layer3(out)
        out3 = out
        out = self.layer4(out)
        out4 = out
        # out = self.avgpool(out)
        # out = self.conv11(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out0, out1, out2, out3, out4


if __name__ == '__main__':
    x = torch.randn(128, 4, 192)
    net = ResNet50_backbone(4)
    out0, out1, out2, out3, out4 = net(x)
    print('out0.shape: ', out0.shape)
    print('out1.shape: ', out1.shape)
    print('out2.shape: ', out2.shape)
    print('out3.shape: ', out3.shape)
    print('out4.shape: ', out4.shape)
    # print('out.shape: ', out.shape)
    # print(out)
    # SECTION:渐进融合
    x_1 = out1
    x_2 = out2
    x_3 = out3
    x_4 = out4
    # PART 1:x_1、x_2
    Fusion_1 = [x_1, x_2]
    x_1 = ASFF_2(inter_dim=x_1.shape[1], level=0, channel=[x_1.shape[1], x_2.shape[1]])(Fusion_1)
    print(x_1.shape)
    x_2 = ASFF_2(inter_dim=x_2.shape[1], level=1, channel=[x_1.shape[1], x_2.shape[1]])(Fusion_1)
    print(x_2.shape)

    # PART 2:X_1、x_2、x_3
    Fusion_2 = [x_1, x_2, x_3]
    x_1 = ASFF_3(inter_dim=x_1.shape[1], level=0, channel=[x_1.shape[1], x_2.shape[1], x_3.shape[1]])(Fusion_2)
    print(x_1.shape)
    x_2 = ASFF_3(inter_dim=x_2.shape[1], level=1, channel=[x_1.shape[1], x_2.shape[1], x_3.shape[1]])(Fusion_2)
    print(x_2.shape)
    x_3 = ASFF_3(inter_dim=x_3.shape[1], level=2, channel=[x_1.shape[1], x_2.shape[1], x_3.shape[1]])(Fusion_2)
    print(x_3.shape)
    # PART 3:X_1、x_2、x_3、x_4
    Fusion_3 = [x_1, x_2, x_3, x_4]
    x_1 = ASFF_4(inter_dim=x_1.shape[1], level=0, channel=[x_1.shape[1], x_2.shape[1], x_3.shape[1], x_4.shape[1]])(Fusion_3)
    print(x_1.shape)
    x_2 = ASFF_4(inter_dim=x_2.shape[1], level=1, channel=[x_1.shape[1], x_2.shape[1], x_3.shape[1], x_4.shape[1]])(Fusion_3)
    print(x_2.shape)
    x_3 = ASFF_4(inter_dim=x_3.shape[1], level=2, channel=[x_1.shape[1], x_2.shape[1], x_3.shape[1], x_4.shape[1]])(Fusion_3)
    print(x_3.shape)
    x_4 = ASFF_4(inter_dim=x_4.shape[1], level=3, channel=[x_1.shape[1], x_2.shape[1], x_3.shape[1], x_4.shape[1]])(Fusion_3)
    print(x_4.shape)


    x_1 = nn.Conv1d(x_1.shape[1], 4, kernel_size=1)(x_1)
    x_2 = Upsample(x_2.shape[1], x_1.shape[1], scale_factor=2)(x_2)
    x_3 = Upsample(x_3.shape[1], x_1.shape[1], scale_factor=4)(x_3)
    x_4 = Upsample(x_4.shape[1], x_1.shape[1], scale_factor=8)(x_4)
    print(x_1.shape)
    print(x_2.shape)
    print(x_3.shape)
    print(x_4.shape)