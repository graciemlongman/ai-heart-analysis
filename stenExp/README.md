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

# Use
To train a model, enter the `run.py` file. In the hyperparaemter section, specify the model you would like to use, the opimiser, whether you are resuming training or using bounding boxes. From the command line simply use `python run.py`. Results will automatically be stored in the model_runs folder.

To evaluate the model, again specify the model you have just trained in the `test.py` file, then simply `python test.py`. Results will automatically be stored in the model_runs folder.

You can modify architectures - place the pytorch code in the models folder. You must add an import in the utils folder for the `ModelZoo` function to init the model class, which is how the `run.py` file takes the models.

For the Mamba models, these are run on the nnUNet framework. See the nnUNet docs or the citations below for use. I placed the modified architectures in the nets folder and wrote a new trainer class for teh experiments I made. 

(more detail in report if needed)

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