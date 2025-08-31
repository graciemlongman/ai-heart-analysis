# Stenosis Experiments
The goal of the stenosis experiments was to improve on the StenUNet architecture guven it had open source code so would be a good starting point.
Hence each model followed the StenUNet pipeline; using the same preprocessing, data augmentations and post processing methods. The DL architecture was changed to see if any improvement could be observed on the nn-UNet originally used.

Five DL architectures were applied to the ARCADE stenosis dataset:

1. DeepLabV3ResNet101
2. Attention UNet
3. YOLOv8x-seg
4. ResUNet++
5. TransUNet

The main metrics used to assess performance were Dice score and mean IoU.

To further probe results, two more mini experiments were conducted:
1. Post processing threshold study
2. Optimiser study; Adam, RMSprop, SGD, for the top three models

Following this different modifications to various architectures were investigated:
1. DeepLabV3ResNet101
    - Squeeze/excitation blocks
    - Deformable convolutions
    - CBAM blocks in the resnet backbone or placed before the deeplab head classifier
2. Attention UNet
    - Deformable convolutions
    - ASPP bridge
    - Residual blocks
    - Bottleneck blocks
3. U-Mamba
    - Attention gates
    - CBAM block as the last step
4. LKM-UNET
    - LKM-UNet with mamba blocks placed only in the bottleneck between enc/dec

Finally Bounding-Box informed models were investigated.


# Citations
- DeepLabV3ResNet101 architecture https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html
- Resnet architecture - https://github.com/a-martyn/resnet/blob/master/resnet.py, https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html
- Attention UNet architecture https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
- YOLOv8x-seg architecture https://docs.ultralytics.com/
- ResUNet++ architecture https://github.com/DebeshJha/ResUNetplusplus-PyTorch-
- TransUNet architecture https://github.com/mkara44/transunet_pytorch/tree/main/utils
- U-Mamba https://github.com/bowang-lab/U-Mamba
- LKM-UNet https://github.com/wjh892521292/LKM-UNet
- BB-UNet https://github.com/rosanajurdi/BB-UNet_UNet_with_bounding_box_prior
- StenUNet pre and post processing https://github.com/HuiLin0220/StenUNet