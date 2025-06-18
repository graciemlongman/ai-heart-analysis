
import os, time, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
import collections
from utils import *
from preparedata import load_data
from postprocess import *
from model_zoo import ModelZoo

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

def evaluate(model, save_path, results_path, test_x, test_y, size, pp_threshold=50):
    metrics_score, post_metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = x.split("/")
        name = f"{name[-3]}_{name[-1]}"
        
        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0) 
        image = image/255.0

        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

    #     """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE) 
        mask = cv2.resize(mask, size)
        save_mask = np.expand_dims(mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
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
    model_choice = 'aunet3'
    optim_choice = 'Adam'

    """ Directories and chkpt path """
    folder =f'{model_choice}/{optim_choice}'
    checkpoint_path = f"stenExp/model_runs/{folder}/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelZoo(choice=model_choice).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    # only need test here
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()

    save_path = f"stenExp/model_runs/{folder}/results/"
    for item in ["mask", "joint", "procd_mask"]:
        if not os.path.exists(f"{save_path}/{item}"):
            os.makedirs(f"{save_path}/{item}")
        else:
            file_exists_print_and_exit()

    results_path = f'{save_path}results.txt'
    create_file(results_path)

    size = (256, 256)
    evaluate(model, save_path, results_path, test_x, test_y, size)
