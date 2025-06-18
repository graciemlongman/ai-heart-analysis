
## ADAPTED FROM https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
## Removed 5th conv block in decoder and replaced with ASPP block
## Replace 1-4th conv blocks with bottleneck resblocks


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class bottleneck(nn.Module):
    def __init__(self, ch_in, ch_out, base_width=16):
        super(bottleneck,self).__init__()
        self.Downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        
        width = int(ch_out * base_width/64)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(ch_in, width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
            )
        
        self.Conv2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
            )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(width, ch_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            )
        
        self.Relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_d = self.Downsample(x)
        
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        
        x = x + x_d
        x = self.Relu(x)

        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class ASPP(nn.Module):
    def __init__(self, ch_in, ch_out, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(ch_out)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(ch_out)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(ch_out)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(ch_out)
        )

        self.c5 = nn.Conv2d(ch_out, ch_out, kernel_size=1, padding=0)
    
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x_cat = x1 + x2 + x3 + x4
        x = self.c5(x_cat)
        return x

class AttU_Net3(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net3,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Encoder1 = bottleneck(ch_in=img_ch,ch_out=64)
        self.Encoder2 = bottleneck(ch_in=64,ch_out=128)
        self.Encoder3 = bottleneck(ch_in=128,ch_out=256)
        self.Encoder4 = bottleneck(ch_in=256,ch_out=512)

        self.bridge = ASPP(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Encoder1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Encoder2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Encoder3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Encoder4(x4)

        # bridge
        b1 = self.Maxpool(x4)
        b1 = self.bridge(b1)

        # decoding + concat path
        d5 = self.Up5(b1)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1