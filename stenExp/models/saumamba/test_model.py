import torch
from vmamba import VSSM

input_channels=3 
num_classes=1
depths=[2, 2, 9, 2]
depths_decoder=[2, 9, 2, 2]
drop_path_rate=0.2

vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )

model_dict = vmunet.state_dict()
check_point=torch.load("pre_trained_weights/vmamba_small_e238_ema.pth")

pretrained_dict = modelCheckpoint['model']


new_dict = {k: v for k, v in pretrained_dict.items()}
print(new_dict)