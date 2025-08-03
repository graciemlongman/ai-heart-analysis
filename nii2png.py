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

def calc_metrics(images, masks, preds, size, save_path, results_path):
    for item in ["mask", "joint", "procd_mask"]:
        if not os.path.exists(f"{save_path}/{item}"):
            os.makedirs(f"{save_path}/{item}")
        else:
            file_exists_print_and_exit()

    metrics_score, post_metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for (x, y, z) in tqdm(zip(os.listdir(masks), os.listdir(preds), os.listdir(images)), total=len(masks)):

        name = x[:11] + '.png'

        img = cv2.imread(images+z, cv2.IMREAD_COLOR) # (512,512)
        img = cv2.resize(img, size)

        mask = cv2.imread(masks+x, cv2.IMREAD_GRAYSCALE)
        mask= cv2.resize(mask, size)
        save_mask = np.stack([mask,mask,mask],axis=-1)

        y_pred = cv2.imread(preds+y, cv2.IMREAD_GRAYSCALE)
        y_pred = cv2.resize(y_pred, size)
        save_y_pred = np.stack([y_pred,y_pred,y_pred], axis=-1)
        
        score = calculate_metrics(mask, y_pred,y_true_proc='evaluate', y_pred_proc='postprocessed', size=size)
        metrics_score = list(map(add, metrics_score, score))

        y_post_pred = binary_remove_small_segments(cv2.resize(y_pred, (512,512)), 50)
        save_y_post_pred = np.stack((cv2.resize(y_post_pred,size),) * 3, axis=-1)*255
        
        post_score=calculate_metrics(mask, y_post_pred, y_true_proc='evaluate', y_pred_proc='postprocessed', size=size)
        post_metrics_score=list(map(add, post_metrics_score, post_score))

        plot_true_vs_preds_to_file(size, save_path, name, img, save_mask, save_y_pred, save_y_post_pred)
        overlay_results(size, save_path, name, img, save_mask, save_y_pred, save_y_post_pred)

    metrics = mean_score(metrics_score, 300)
    post_metrics = mean_score(post_metrics_score, 300)

    save_test_results_to_file(results_path, metrics, post_metrics)

if __name__ == '__main__':
    input_folder = 'U-Mamba/data/nnUNet_results/Dataset112_ArcadeXCA/nnUNetTrainerattUMambaBot__nnUNetPlans__2d/inference/'
    out_preds = 'stenExp/model_runs/nnU-attUMambaBot_2_final/one/preds/'

    labels_folder = 'U-Mamba/data/nnUNet_raw/Dataset112_ArcadeXCA/labels_test/'
    masks = 'stenExp/model_runs/nnU-attUMambaBot_2_final/one/labels_test/'

    input_images_folder = 'U-Mamba/data/nnUNet_raw/Dataset112_ArcadeXCA/images_test/'
    images = 'stenExp/model_runs/nnU-attUMambaBot_2_final/one/images_test/'

    nifti2png(input_folder, out_preds)
    nifti2png(labels_folder, masks)
    nifti2png(input_images_folder, images)

    save_path = 'stenExp/model_runs/nnU-attUMambaBot_2_final/one/results/'
    results_path = 'stenExp/model_runs/nnU-attUMambaBot_2_final/one/results/results.txt'
    size=(256,256)
    calc_metrics(images, masks, out_preds, size, save_path, results_path)