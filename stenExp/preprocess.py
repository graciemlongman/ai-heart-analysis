


#### TAKEN FROM https://github.com/HuiLin0220/StenUNet/blob/master/pre_process/preprocess.py ####

import skimage
from scipy.ndimage import gaussian_filter
from skimage import io, exposure

from utils_StenUNet.DDFB import *
from utils_StenUNet.guided_filter import *
from utils_StenUNet.homomorphic_filter import *
from utils_StenUNet.multiscale_tophat import *

def preprocess(img):

    homo = HomomorphicFilter()
    radius = 8
    eps = 0.2
    alpha = 2

    # Define the scale parameter sigma
    sigma = 2

    image = img.copy()

    if len(image.shape) > 2:
        # p('image different shape: {}'.format(image.shape))
        image = image.mean(axis=np.argmin(image.shape))
        # p('image new shape: {}'.format(image.shape))

    # 1 homomorphic transform
    homomorphic_img = homo.apply_filter(image.copy())

    # 2 Normalize
    nml_homomorphic_img = normalize(homomorphic_img)
    
    '''
    # 3 multiscale
    multi_scale_img = multi_scale(nml_homomorphic_img.copy())

    # 4 Adaptive Equalization
    img_adapteq = exposure.equalize_hist(multi_scale_img)

    # 5 Apply Gaussian smoothing to the input image
    smoothed_img = gaussian_filter(img_adapteq.copy(), sigma)
    adaptive_thresh = skimage.filters.threshold_local(smoothed_img, 7, 'gaussian')

    # 6 Guided_filter
    GF = FastGuidedFilter(image, radius, eps, alpha)
    fgf_image = GF.filter(adjust_scale(adaptive_thresh))

    # 7 multiscale
    filtered_fgf = adjust_scale(fgf_image) + multi_scale_img

    # 8
    multi_scale_img_only = multi_scale(img.copy())

    #9 ddfb
    high_pass_spatial_mean = remove_spatial_mean(nml_homomorphic_img)
    ddfb_vessels = ddfb(high_pass_spatial_mean)
    enhanced_vessels = adjust_scale(multi_scale_img + ddfb_vessels)
    '''
    #return all images

    ### added by me ###
    # convert back to 3 channel image for resunet++
    nml_homomorphic_img = np.stack([nml_homomorphic_img]*3, axis=-1).astype(np.uint8)

    return nml_homomorphic_img

### added by me ###
def preprocess_inplace(img_path):
    img = io.imread(img_path)
    image = preprocess(img)
    io.imsave(img_path, image)