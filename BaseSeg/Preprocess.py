from skimage import io, exposure, util, morphology
from skimage.color import rgb2gray
import cv2 as cv
import numpy as np 

def preprocess(img):

    image = img.astype(np.float32) / 255.0 
    
    if image.ndim == 3 and image.shape[2] == 3:
        image = rgb2gray(image)
        
    # First, contrast limited adaptive histogram equalisation (CLAHE)
    clahe_image = exposure.equalize_adapthist(image)

    # invert image
    inv_clahe_image = util.invert(clahe_image)

    # White top-hat transform (large SE, exact size not specified)
    w_tophat_image = morphology.white_tophat(inv_clahe_image, morphology.disk(20))

    # Original - top hat
    orig_tophat = image - w_tophat_image

    # Non-negative thresholding, threshold not specified
    thresh_image = orig_tophat #> 0

    # CLAHE
    clahe_thresh_image = exposure.equalize_adapthist(thresh_image)
    preprocess_img = clahe_thresh_image

    return (preprocess_img * 255).astype(np.uint8)

def preprocess_inplace(img_path):
    img = io.imread(img_path)
    image = preprocess(img)
    io.imsave(img_path, image)