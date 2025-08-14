import nibabel as nib
from stenExp.utils.metrics import *
from stenExp.utils.utils import *
from stenExp.postprocess import *
from PIL import Image
import numpy as np
import os
from operator import add

# Load the NIfTI image - adjust for json
def nifti2png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        #print(output_folder)
        file_exists_print_and_exit()

    for file in os.listdir(input_folder):
        print(file)
        if file.endswith('.nii.gz'):
            basename=file[:11]
            nii_img = nib.load(input_folder+file)
            img_data = nii_img.get_fdata().astype(np.uint8)

            # Ensure the shape is correct
            if img_data.shape[-1] == 3:
                img = (img_data * 255)
                img = Image.fromarray(img, 'RGB')
                img.save(f'{output_folder}{basename}.png')
            else:
                raise ValueError(f"Unexpected image shape: {img_data.shape}")

if __name__ == '__main__':
    destination_path = 'stenExp/model_runs/LKM-UNet/three/'

    input_folder = 'LKM-UNet/data/nnUNet_results/Dataset112_ArcadeXCA/nnUNetTrainerLKMUNet__nnUNetPlans__2d/inference_check/'
    out_preds = f'{destination_path}results/mask/'

    labels_folder = 'LKM-UNet/data/nnUNet_raw/Dataset112_ArcadeXCA/labels_test/'
    masks = f'{destination_path}labels_test/'

    input_images_folder = 'LKM-UNet/data/nnUNet_raw/Dataset112_ArcadeXCA/images_test/'
    images = f'{destination_path}images_test/'

    nifti2png(input_folder, out_preds)
    nifti2png(labels_folder, masks)
    nifti2png(input_images_folder, images)

    save_path = f'{destination_path}results/'
    results_path = f'{destination_path}results.txt'
    size=(256,256)
    
    metrics_on_preds(images, masks, out_preds, size, save_path, results_path)