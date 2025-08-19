
import os, time, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from tqdm import tqdm
import torch
import collections
from utils.utils import *
from preparedata import load_data


# https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/test.py
def process_mask(y_pred):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

def predict(out_preds, test_x, test_b, size, bbox):
    if not os.path.exists(out_preds):
        os.makedirs(out_preds)
    else:
        file_exists_print_and_exit()
    
    for i, (x, y, b) in tqdm(enumerate(zip(test_x, test_y, test_b)), total=len(test_x)):
        name = x.split("/")
        name = f"{name[-1]}"
        
        image = cv2.resize(cv2.imread(x, cv2.IMREAD_COLOR), size)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)/255.0
        image = torch.from_numpy(image.astype(np.float32)).to(device)
        
        box = cv2.resize(cv2.imread(b, cv2.IMREAD_GRAYSCALE), size)
        box = np.expand_dims(box, axis=0)/255.0
        box = np.expand_dims(box, axis=0) # for rn101
        box = torch.from_numpy(box.astype(np.float32)).to(device)
        #print(box.shape)

        with torch.no_grad():
            if bbox:
                y_pred = model(image, box)
            else:
                y_pred = model(image)

            if isinstance(y_pred, collections.OrderedDict):
                y_pred=y_pred['out']
            y_pred = torch.sigmoid(y_pred)
            save = process_mask(y_pred)
            cv2.imwrite(f'{out_preds}/{name}', save)


if __name__ == "__main__":
    """ Seeding """
    seeding(42)
    
    # pp_threshold = [25, 50, 75, 100, 125, 150]
    # for thresh in pp_threshold:

    """ Vars """
    model_choice = 'deeplabv3resnet101_cbam_class'
    optim_choice = 'RMSprop'
    bbox=False

    """ Directories and chkpt path """
    images='stenExp/datasets/arcade/stenosis/test/images/'
    masks='stenExp/datasets/arcade/stenosis/test/annotations/'
    
    folder =f'{model_choice}/{optim_choice}'
    out_preds=f'stenExp/model_runs/{folder}/results/mask/'
    checkpoint_path = f"stenExp/model_runs/{folder}/checkpoint.pth"

    save_path = f"stenExp/model_runs/{folder}/results/"
    results_path = f'{save_path}results.txt'

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelZoo(choice=model_choice, partition='train').to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    (train_x, train_y, train_b), (valid_x, valid_y, valid_b), (test_x, test_y, test_b) = load_data(bbox=True)

    size = (256, 256)
    
    predict(out_preds, test_x, test_b, size, bbox)
    metrics_on_preds(images, masks, out_preds, size, save_path, results_path)
    
    
