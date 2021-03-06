# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/2003.14299.pdf
# - https://arxiv.org/pdf/1807.08865.pdf
# - https://xjqi.github.io/geonet.pdf
# - https://github.com/meteorshowers/StereoNet-ActiveStereoNet
# - https://github.com/xjqi/GeoNet
# - https://openaccess.thecvf.com/content_ECCV_2018/papers/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.pdf
# - https://github.com/EPFL-VILAB/XTConsistency/blob/master/modules/unet.py
# - https://arxiv.org/pdf/1606.00373.pdf
# - https://arxiv.org/pdf/2010.06626v2.pdf
# - https://github.com/dontLoveBugs/FCRN_pytorch/blob/master/criteria.py#L37

import torch
import torch.nn as nn
import torch.nn.functional as F



class UNetBlock(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2) if down_size else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        return x


class UNetBlockT(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super(UNetBlockT, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if up_sample else nn.Identity()
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNetFeature(nn.Module):
    def __init__(self, in_channels=3):
        super(UNetFeature, self).__init__()
        self.in_channels = in_channels
        self.down_block1 = UNetBlock(in_channels, 16, False)
        self.down_block2 = UNetBlock(16, 32, True)
        self.down_block3 = UNetBlock(32, 64, True)
        self.down_block4 = UNetBlock(64, 128, True)
        self.down_block5 = UNetBlock(128, 256, True)
        self.down_block6 = UNetBlock(256, 512, True)
        self.down_block7 = UNetBlock(512, 1024, True)

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, 1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, 1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, 1024)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        x7 = self.down_block7(x6)

        x7 = self.relu(self.bn1(self.mid_conv1(x7)))
        x7 = self.relu(self.bn2(self.mid_conv2(x7)))
        x7 = self.relu(self.bn3(self.mid_conv3(x7)))

        return x1, x2, x3, x4, x5, x6, x7


class UNetFCN(nn.Module):
    def __init__(self, out_channels=3):
        super(UNetFCN, self).__init__()
        self.up_block1 = UNetBlockT(1024, 2048, 512)
        self.up_block2 = UNetBlockT(512, 512, 256)
        self.up_block3 = UNetBlockT(256, 256, 128)
        self.up_block4 = UNetBlockT(128, 128, 64)
        self.up_block5 = UNetBlockT(64, 64, 32)
        self.up_block6 = UNetBlockT(32, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        x = self.up_block1(x6, x7)
        x = self.up_block2(x5, x)
        x = self.up_block3(x4, x)
        x = self.up_block4(x3, x)
        x = self.up_block5(x2, x)
        x = self.up_block6(x1, x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = UNetFeature()
        self.predict = UNetFCN(out_channels=3)

    def forward(self, left_image, right_image):
        featureL = self.feature(left_image)
        featureR = self.feature(right_image)

        feature = list(map(lambda x: torch.cat(x, dim=1), zip(featureL, featureR)))
        norm = self.predict(*feature)

        return norm


class LossFunction(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(LossFunction, self).__init__()
        self.normal_loss = nn.L1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()

        self.alpha = alpha
        self.beta = beta

        self.normal_loss_val = 0
        self.cos_loss_val = 0

    def forward(self, predictions, targets):
        normal_p = predictions
        normal_gt = targets

        normal_p_n = F.normalize(normal_p, p=2, dim=1)
        normal_gt_n = F.normalize(normal_gt, p=2, dim=1)

        normal_loss = self.normal_loss(normal_p, normal_gt) * self.alpha
        cos_loss = self.cos_loss(normal_p_n, normal_gt_n, torch.tensor(1.0, device=normal_p.device)) * self.beta

        self.normal_loss_val = normal_loss.item()
        self.cos_loss_val = cos_loss.item()

        normal = normal_loss + cos_loss

        return normal

    def show(self):
        loss = self.normal_loss_val + self.cos_loss_val
        return f'(total:{loss:.4f} l1:{self.normal_loss_val} 1-cos:{self.cos_loss_val})'


if __name__ == "__main__":
    left = torch.rand((4, 3, 256, 256))
    right = torch.rand((4, 3, 256, 256))
    model = Model()
    pred = model(left, right)
    assert pred.shape == (4, 3, 256, 256), "Model"

    print("model ok")
