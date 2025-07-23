
import torchvision
import torch
from torch import nn
from torchvision.models.resnet import Bottleneck
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3


class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class SE_block(Bottleneck): #SE-Pre from the paper
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__(inplanes, planes, stride, downsample, dilation=dilation)

        self.SE_block = Squeeze_Excitation(inplanes)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels*self.expansion, kernel_size=3, stride=1, padding=dilation, dilation=dilation)

    def forward(self, x):
        
        xSE = self.SE_block(x)
        
        x1 = self.conv1(xSE)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        if self.downsample is not None:
            x=self.downsample(x)
        x3=x3+x
        x3=self.relu(x3)

        return x3
    
import numpy as np
class BB_block(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__(inplanes, planes, stride, downsample, dilation=dilation)

        self.BBconv = nn.Conv2d(3, self.conv2.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels*self.expansion, kernel_size=3, stride=1, padding=dilation, dilation=dilation)

    def forward(self, x, b):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)


        blank0 = torch.zeros((b.shape[0],1,256,256)).to(torch.device('cuda'))
        blank1 = torch.ones((b.shape[0],1,256,256)).to(torch.device('cuda'))
        b = torch.cat((b,blank0,blank1), 1)

        b = F.interpolate(b, size=x2.shape[2:], mode='nearest')
        b = self.BBconv(b)
        x2 = b + x2
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        if self.downsample is not None:
            x=self.downsample(x)
        x3=x3+x
        x3=self.relu(x3)

        return x3
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dilation=1

        # Initial convolution and max pooling layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the layers dynamically based on the input configuration
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=True)

        # Final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1, dilate=False):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *=stride
            stride=1
        
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = [block(inplanes=self.in_channels,
                        planes=out_channels,
                        stride=stride,
                        downsample=downsample,
                        dilation=previous_dilation)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.ModuleList(layers)
    
    def _forward_layer(self, layers, x, b=None):
        for block in layers:
            if b!= None:
                x = block(x, b)
            else:
                x=block(x)
        return x

    def forward(self, x, b=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if b != None:
            x = self._forward_layer(self.layer1, x, b)
            x = self._forward_layer(self.layer2, x, b)
            x = self._forward_layer(self.layer3, x, b)
            x = self._forward_layer(self.layer4, x, b)
        else:
            x = self._forward_layer(self.layer1, x)
            x = self._forward_layer(self.layer2, x)
            x = self._forward_layer(self.layer3, x)
            x = self._forward_layer(self.layer4, x)
        
        # x = self.avgpool(x)
        # x = torch.flatten(x,1)
        # x = self.fc(x)

        return {'out':x}

class DeepLabV3_BB(DeepLabV3):
    def __init__(self, backbone=ResNet(BB_block, [3,4,23,3]), classifier=DeepLabHead(2048, 1)):
        super().__init__(backbone, classifier)

        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, b=None):
        input_shape = x.shape[-2:]

        features = self.backbone(x, b)
        x = self.classifier(features['out'])

        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
class DeepLabV3_SE(DeepLabV3):
    def __init__(self, backbone=ResNet(SE_block, [3,4,23,3]), classifier=DeepLabHead(2048, 1)):
        super().__init__(backbone, classifier)

        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, b=None):
        input_shape = x.shape[-2:]

        features = self.backbone(x, b)
        x = self.classifier(features['out'])

        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class nomod(DeepLabV3):
    def __init__(self, backbone=ResNet(Bottleneck, [3,4,23,3]), classifier=DeepLabHead(2048, 1)):
        super().__init__(backbone, classifier)

        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, b=None):
        input_shape = x.shape[-2:]

        features = self.backbone(x, b)
        x = self.classifier(features['out'])

        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
def _load_weights(model, pretrained_model=None):
    pretrained_model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT', 
                                                progress=True, aux_loss=None)
    pretrained_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    pretrained_dict = pretrained_model.state_dict()
    
    model_dict = model.state_dict()
    
    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    
    model_dict.update(filtered_dict)

    model.load_state_dict(model_dict)

    return model

if __name__ == '__main__': 
    
    model = _load_weights(DeepLabV3_BB())
    x = torch.randn(8, 3, 256, 256)
    b = torch.randn(8, 1, 256, 256)

    with torch.no_grad():
        output = model(x,b)
    print(output.shape) 