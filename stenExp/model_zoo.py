from models.resunetplusplus import build_resunetplusplus
from models.aunet import AttU_Net
from models.aunet1 import AttU_Net1
from models.aunet2 import AttU_Net2
from models.aunet3 import AttU_Net3
from models.transunet import TransUNet
from torch import nn
import torchvision
import sys
import numpy as np


def ModelZoo(choice):
    if choice == 'resunetplusplus':
        return build_resunetplusplus()
    elif choice == 'deeplabv3resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT', 
                                                progress=True, aux_loss=None)
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        return model
    elif choice == 'attentionunet':
        return AttU_Net()
    elif choice == 'transunet':
        return TransUNet(img_dim=256,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)
    elif choice == 'aunet1':
        return AttU_Net1()
    elif choice == 'aunet2':
        return AttU_Net2()
    elif choice == 'aunet3':
        return AttU_Net3()
    else:
        raise ValueError(f"Model choice '{choice}' is not supported.")
    

if __name__=='__main__':
    model=ModelZoo('attentionunet')