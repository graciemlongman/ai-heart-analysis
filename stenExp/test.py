
import os, time, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
import collections
from utils.utils import *
from preparedata import load_data
from postprocess import *

def format_mask_for_post_processing(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = (y_pred > 0.5).astype(np.uint8)
    y_pred=cv2.resize(y_pred, (512,512))
    return y_pred

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

def evaluate(model, save_path, results_path, test_x, test_y, test_b, size, bbox, pp_threshold=50):
    metrics_score, post_metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y, b) in tqdm(enumerate(zip(test_x, test_y, test_b)), total=len(test_x)):
        name = x.split("/")
        name = f"{name[-3]}_{name[-1]}"
        
        """ Image """
        image = cv2.resize(cv2.imread(x, cv2.IMREAD_COLOR), size)
        save_img = image
        
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)/255.0
        image = torch.from_numpy(image.astype(np.float32)).to(device)

        """ Mask """
        mask = cv2.resize(cv2.imread(y, cv2.IMREAD_GRAYSCALE) , size)
        save_mask = np.expand_dims(mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)

        """ Box """
        box = cv2.resize(cv2.imread(b, cv2.IMREAD_GRAYSCALE), size)
        box = np.expand_dims(box, axis=0)/255.0
        box = np.expand_dims(box, axis=0)
        box = torch.from_numpy(box.astype(np.float32)).to(device)
        #print(box.shape)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            if bbox:
                y_pred = model(image, box)
            else:
                y_pred = model(image)

            if isinstance(y_pred, collections.OrderedDict):
                y_pred=y_pred['out']
            y_pred = torch.sigmoid(y_pred)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score = calculate_metrics(mask, y_pred, y_true_proc='evaluate')
            metrics_score = list(map(add, metrics_score, score))

            """ Predicted Mask for img display """
            y_pred_3d=process_mask(y_pred)

            """ Post process prediction"""
            y_pred_2d = format_mask_for_post_processing(y_pred)
            y_post_pred = binary_remove_small_segments(y_pred_2d, pp_threshold)
            y_post_pred_3d = np.stack((cv2.resize(y_post_pred,size),) * 3, axis=-1)*255

            """Evaluation metrics for post processed"""
            post_score=calculate_metrics(mask, y_post_pred, y_true_proc='evaluate', y_pred_proc='postprocessed', size=size)
            post_metrics_score=list(map(add, post_metrics_score, post_score))

        """ Save the image - mask - pred """
        plot_true_vs_preds_to_file(size, save_path, name, save_img, save_mask, y_pred_3d, y_post_pred_3d)
        overlay_results(size, save_path, name, save_img, save_mask, y_pred_3d, y_post_pred_3d)

    """ Calc metrics """
    metrics = mean_score(metrics_score, len(test_x))
    post_metrics = mean_score(post_metrics_score, len(test_x))
    mean_time_taken = np.mean(time_taken)

    """ Save to file """
    save_test_results_to_file(results_path, metrics, post_metrics, mean_time_taken, len(test_x))


if __name__ == "__main__":
    """ Seeding """
    seeding(42)
    
    # pp_threshold = [25, 50, 75, 100, 125, 150]
    # for thresh in pp_threshold:

    """ Vars """
    model_choice = 'deeplabv3resnet101_nomod'
    optim_choice = 'Adam'
    bbox=False

    """ Directories and chkpt path """
    folder =f'{model_choice}/{optim_choice}'
    checkpoint_path = f"stenExp/model_runs/{folder}/checkpoint.pth"
    save_path = f"stenExp/model_runs/{folder}/results/"
    results_path = f'{save_path}results.txt'

    for item in ["mask", "joint", "procd_mask"]:
        if not os.path.exists(f"{save_path}/{item}"):
            os.makedirs(f"{save_path}/{item}")
        else:
            file_exists_print_and_exit()

    create_file(results_path)


    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelZoo(choice=model_choice, partition='train').to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    (train_x, train_y, train_b), (valid_x, valid_y, valid_b), (test_x, test_y, test_b) = load_data(bbox=True)

    size = (256, 256)
    evaluate(model, save_path, results_path, test_x, test_y, test_b, size, bbox)
