# Stenosis Experiments
The goal of the stenosis experiments was to improve on the StenUNet architecture guven it had open source code so would be a good starting point.
Hence each model followed the StenUNet pipeline; using the same preprocessing, data augmentations and post processing methods. The DL architecture was changed to see if any improvement could be observed on the nn-UNet originally used.

Five DL architectures were applied to the ARCADE stenosis dataset:

1. DeepLabV3ResNet101
2. Attention UNet
3. YOLOv8x-seg
4. ResUNet++
5. TransUNet

**Results here**

The main metrics used to assess performance were mean F1 score and mean IoU.

To further probe results, three more mini experiments were conducted:
1. Post processing threshold study
2. Optimiser study; Adam, RMSprop, SGD, for the top three models
3. Modifications to AttentionUNet architecture

**Results here**

Following from these experiments..

# Folder structure

# Citations
- DeepLabV3ResNet101 architecture https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html
- Attention UNet architecture https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
- YOLOv8x-seg architecture https://docs.ultralytics.com/
- ResUNet++ architecture https://github.com/DebeshJha/ResUNetplusplus-PyTorch-
- TransUNet architecture https://github.com/mkara44/transunet_pytorch/tree/main/utils
- StenUNet pre and post processing https://github.com/HuiLin0220/StenUNet