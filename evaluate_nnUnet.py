import nibabel as nib
from stenExp.utils.metrics import *
from stenExp.utils.utils import *
from stenExp.utils.postprocess import *
from PIL import Image
import numpy as np
import os
from operator import add

# script which converts the predicted images to png and then
# calculates the metrics on the preds, storing them in the 
# correct folder in model_runs 

def nifti2png(input_folder, output_folder):
    if not os.path.exists(output_folder): # Create folder to store png outputs
        os.makedirs(output_folder)
    else:
        file_exists_print_and_exit()

    for file in os.listdir(input_folder):
        if file.endswith('.nii.gz'):
            basename=file[:11]
            nii_img = nib.load(input_folder+file) # Load image
            img_data = nii_img.get_fdata().astype(np.uint8)

            # Ensure the shape is correct
            if img_data.shape[-1] == 3:
                img = (img_data * 255)
                img = Image.fromarray(img, 'RGB')
                img.save(f'{output_folder}{basename}.png') #Save image
            else:
                raise ValueError(f"Unexpected image shape: {img_data.shape}")

if __name__ == '__main__':
    destination_path = 'stenExp/model_runs/nnU-MambaBot1_final/one/'

    input_folder = 'U-Mamba/data/nnUNet_results/Dataset112_ArcadeXCA/nnUNetTrainerattUMambaBot1__nnUNetPlans__2d/inference/'
    out_preds = f'{destination_path}results/mask/'

    labels_folder = 'U-Mamba/data/nnUNet_raw/Dataset112_ArcadeXCA/labels_test/'
    masks = f'{destination_path}labels_test/'

    input_images_folder = 'U-Mamba/data/nnUNet_raw/Dataset112_ArcadeXCA/images_test/'
    images = f'{destination_path}images_test/'

    nifti2png(input_folder, out_preds)
    nifti2png(labels_folder, masks)
    nifti2png(input_images_folder, images)

    save_path = f'{destination_path}results/'
    results_path = f'{destination_path}results.txt'
    size=(256,256)
    
    metrics_on_preds(images, masks, out_preds, size, save_path, results_path)