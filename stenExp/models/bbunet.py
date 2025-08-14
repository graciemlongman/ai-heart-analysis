'''
Created on Mar 18, 2020

@author: eljurros
'''
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class DownConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x

class BBConv(Module):
    def __init__(self, in_feat, out_feat, pool_ratio, no_grad_state):
        super(BBConv, self).__init__()
        self.mp = nn.MaxPool2d(pool_ratio)
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        if no_grad_state is True:
            self.conv1.requires_grad = False
        else:
            self.conv1.requires_grad = True
    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = F.sigmoid(x)
        return x
 

class BB_Unet(Module):
    """A reference U-Net model.
    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, drop_rate=0.6, bn_momentum=0.1, no_grad_state=False, BB_boxes = 1, partition='train'):
        super(BB_Unet, self).__init__()

        self.train_flag = True if partition == 'train' else False
        
        #Downsampling path
        self.conv1 = DownConv(3, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottle neck
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)
        # bounding box encoder path:
        self.b1 = BBConv(BB_boxes, 256, 4, no_grad_state)
        self.b2 = BBConv(BB_boxes, 128, 2, no_grad_state)
        self.b3 = BBConv(BB_boxes, 64, 1, no_grad_state)
        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, padding=1)


    def forward(self, x, bb):
        
        x1 = self.conv1(x)
        p1 = self.mp1(x1)

        x2 = self.conv2(p1)
        p2 = self.mp2(x2)

        x3 = self.conv3(p2)
        p3 = self.mp3(x3)

        # Bottle neck
        x4 = self.conv4(p3)
        
        # bbox encoder
        if self.train_flag: #i.e. dont have bboxes when we run inference
            x3 = x3*self.b1(bb)
            # x2 = x2*self.b2(bb)
            # x1 = x1*self.b3(bb)

        # Up-sampling
        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)

        x5 = self.conv9(u3)
        
        return x5

if __name__ == '__main__':
    model = BB_Unet(partition='train')

