
import os, sys
import random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from operator import add
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from utils.metrics import precision, recall, F2, dice_score, jac_score, hd_dist
    from utils.postprocess import *
except:
    pass

try:
    from stenExp.utils.metrics import precision, recall, F2, dice_score, jac_score, hd_dist
    from stenExp.utils.postprocess import *
except:
    pass

## https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/utils.py
""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
######

#Adapted
def print_and_save(file_path, data_str, print_=True):
    if print_:
        print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")
#####

def file_exists_print_and_exit():
    print("Log file exists")
    print('Exiting process - check your directories :)')
    sys.exit()

def create_file(path):
    if os.path.exists(path):
        file_exists_print_and_exit()
    else:
        train_log = open(path, "w")
        train_log.write("\n")
        train_log.close()

## Adapted from  https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/utils.py
def calculate_metrics(y_true, y_pred, y_true_proc=True, y_pred_proc=True,size=None):
    if y_pred_proc=='postprocessed':
        y_pred = cv2.resize(y_pred, size)
        y_pred=np.expand_dims(y_pred, axis=0)
        y_pred=np.expand_dims(y_pred, axis=0)
    else:
        ## Tensor processing
        y_pred = y_pred.detach().cpu().numpy()

    if y_true_proc=='evaluate':
        y_true = np.expand_dims(y_true, axis=0)
        y_true = np.expand_dims(y_true, axis=0)
    else:
        ## Tensor processing
        y_true = y_true.detach().cpu().numpy()
    
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    ## HD
    if len(y_true.shape) == 3:
        score_hd = hd_dist(y_true[0], y_pred[0])
    elif len(y_true.shape) == 4:
        score_hd = hd_dist(y_true[0,0], y_pred[0,0])

    ## Reshape
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta, score_hd]

## adapted from https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/utils.py
def mean_score(metrics_score, num_imgs, print_=False):
    jaccard = metrics_score[0]/num_imgs
    f1 = metrics_score[1]/num_imgs
    recall = metrics_score[2]/num_imgs
    precision = metrics_score[3]/num_imgs
    acc = metrics_score[4]/num_imgs
    f2 = metrics_score[5]/num_imgs
    hd = metrics_score[6]/num_imgs
    
    if print_:
        print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - HD: {hd:1.4f}")
    
    return [jaccard, f1, recall, precision, acc, f2, hd]

def metrics_on_preds(images, masks, preds, size, save_path, results_path, pp_threshold=50):
    for item in ["joint", "procd_mask", 'overlay']:
        if not os.path.exists(f"{save_path}/{item}"):
            os.makedirs(f"{save_path}/{item}")
        else:
            file_exists_print_and_exit()

    metrics_score, post_metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for (x, y, z) in tqdm(zip(os.listdir(masks), os.listdir(preds), os.listdir(images)), total=len(masks)):
        name = x.split('/')[-1]

        img = cv2.imread(images+z, cv2.IMREAD_COLOR) # (512,512)
        img = cv2.resize(img, size)

        mask = cv2.imread(masks+x, cv2.IMREAD_GRAYSCALE)
        mask= cv2.resize(mask, size)
        save_mask = np.stack([mask,mask,mask],axis=-1)

        y_pred = cv2.imread(preds+y, cv2.IMREAD_GRAYSCALE)
        y_pred = cv2.resize(y_pred, size)
        save_y_pred = np.stack([y_pred,y_pred,y_pred], axis=-1)
        
        score = calculate_metrics(mask, y_pred, y_true_proc='evaluate', y_pred_proc='postprocessed', size=size)
        metrics_score = list(map(add, metrics_score, score))

        y_post_pred = binary_remove_small_segments(cv2.resize(y_pred, (512,512)), pp_threshold)
        save_y_post_pred = np.stack((cv2.resize(y_post_pred,size),) * 3, axis=-1)*255
        
        post_score=calculate_metrics(mask, y_post_pred, y_true_proc='evaluate', y_pred_proc='postprocessed', size=size)
        post_metrics_score=list(map(add, post_metrics_score, post_score))

        plot_true_vs_preds_to_file(size, save_path, name, img, save_mask, save_y_pred, save_y_post_pred)

    metrics = mean_score(metrics_score, 300)
    post_metrics = mean_score(post_metrics_score, 300)

    save_test_results_to_file(results_path, metrics, post_metrics)

def save_test_results_to_file(results_path, metrics, post_metrics, mean_time_taken=None, num_imgs=None):
    m_str = f"Jaccard: {metrics[0]:1.4f} - F1: {metrics[1]:1.4f} - Recall: {metrics[2]:1.4f} - Precision: {metrics[3]:1.4f} - Acc: {metrics[4]:1.4f} - F2: {metrics[5]:1.4f} - HD: {metrics[6]:1.4f} \n"
    pm_str = f"Jaccard: {post_metrics[0]:1.4f} - F1: {post_metrics[1]:1.4f} - Recall: {post_metrics[2]:1.4f} - Precision: {post_metrics[3]:1.4f} - Acc: {post_metrics[4]:1.4f} - F2: {post_metrics[5]:1.4f} - HD: {post_metrics[6]:1.4f} \n"

    print_and_save(results_path, m_str)
    print_and_save(results_path, pm_str)

    if mean_time_taken is not None:
        mean_fps = 1/mean_time_taken
        mean_spf = mean_time_taken/num_imgs
        time_str = f"Mean FPS: {mean_fps} \nMean SPF: {mean_spf} \n"
        print_and_save(results_path, time_str)

def plot_true_vs_preds_to_file(size, save_path, name, image, y_true, y_pred, y_post_pred):
    line = np.ones((size[0], 10, 3)) * 255
    cat_images = np.concatenate([image, line, y_true, line, y_pred, line, y_post_pred], axis=1)
    cv2.imwrite(f"{save_path}/joint/{name}", cat_images)
    cv2.imwrite(f"{save_path}/procd_mask/{name}", y_post_pred)

    cat_raw_imgs = np.concatenate([image, line, image, line, image, line, image], axis=1)
    overlaid_images = cv2.addWeighted(cat_raw_imgs, 0.5, cat_images, 0.5, 0)
    cv2.imwrite(f"{save_path}/overlay/{name}", overlaid_images)


def OptZoo(choice, model, lr):
    if choice == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif choice == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001) #TransUNet
    elif choice == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Optimiser {choice} is not supported')

try:
    from models.resunetplusplus import build_resunetplusplus
    #from models.transunet.transunet import TransUNet
    from models.saumamba.vmunet import VMUNet

    from models.aunet_and_mods.aunet import AttU_Net
    from models.aunet_and_mods.aunet1 import AttU_Net1 #deformable
    from models.aunet_and_mods.aunet2 import AttU_Net2 #aspp
    from models.aunet_and_mods.aunet3 import AttU_Net3 #residual
    from models.aunet_and_mods.aunet4 import AttU_Net4 #bottleneck

    from models.umamba_and_mods.attUMambaEnc import AttUMambaEnc, InitWeights_He #aunet dec
    from models.umamba_and_mods.attUMambaBot import AttUMambaBot
    from models.umamba_and_mods.attUMambaEnc_2 import UMambaEnc_2 #umamba decoder
    from models.umamba_and_mods.attUMambaBot_2 import UMambaBot_2 
    from models.umamba_and_mods.umambaBot import UMambaBot
    from models.umamba_and_mods.umambaEnc import UMambaEnc

    from models.bbunet import BB_Unet
    from models.bb_aunet import attBB_UNet

    from models.resnet_dlv3 import ResNet101DeepLabV3, ResNet, SE_block, BB_block, DF_Block, CBAM_Block, _load_weights
    from torchvision.models.resnet import Bottleneck
except Exception as e:
    print(f"[IMPORT ERROR] {e}")
    pass
from torch import nn
import torchvision

def ModelZoo(choice, partition=None):
    if choice == 'resunetplusplus':
        return build_resunetplusplus()
    elif choice == 'deeplabv3resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT', 
                                                progress=True, aux_loss=None)
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        return model
    elif choice == 'deeplabv3resnet101_bb':
        return _load_weights(ResNet101DeepLabV3(backbone=ResNet(BB_block, [3,4,23,3])))
    elif choice == 'deeplabv3resnet101_se':
        return _load_weights(ResNet101DeepLabV3(backbone=ResNet(SE_block, [3,4,23,3])))
    elif choice == 'deeplabv3resnet101_df':
        return _load_weights(ResNet101DeepLabV3(backbone=ResNet(Bottleneck, [3,4,23,3], deformable=True)))
    elif choice == 'deeplabv3resnet101_df2':
        return _load_weights(ResNet101DeepLabV3(backbone=ResNet(DF_Block, [3,4,23,3])))
    elif choice == 'deeplabv3resnet101_cbam_block':
        return _load_weights(ResNet101DeepLabV3(backbone=ResNet(CBAM_Block, [3,4,23,3])))
    elif choice == 'deeplabv3resnet101_cbam_class':
        return _load_weights(ResNet101DeepLabV3(backbone=ResNet(Bottleneck, [3,4,23,3], use_cbam_class=True)))
    elif choice == 'deeplabv3resnet101_nomod':
        return _load_weights(ResNet101DeepLabV3())
    elif choice == 'transunet':
        return TransUNet(img_dim=256,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)
    elif choice == 'transunet_weights':
        model = TransUNet(img_dim=256,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)
        model.load_from(weights=np.load('stenExp/models/transunet/pretrained_weights/weights.pt'))
        return model
    elif choice == 'saumamba':
        model = VMUNet(num_classes=1,
                        input_channels=3,
                        depths=[2,2,2,2],
                        depths_decoder=[2,2,2,1],
                        drop_path_rate=0.2,
                        load_ckpt_path='stenExp/models/saumamba/pretrained_weights/vmamba_small_e238_ema.pth')
        model.load_from()
        return model
    elif choice == 'attentionunet':
        return AttU_Net()
    elif choice == 'aunet1':
        return AttU_Net1()
    elif choice == 'aunet2':
        return AttU_Net2()
    elif choice == 'aunet3':
        return AttU_Net3()
    elif choice == 'aunet4':
        return AttU_Net4()
    elif choice == 'bbunet':
        return BB_Unet(partition=partition)
    elif choice == 'bbaunet':
        return attBB_UNet(partition=partition)
    elif choice == 'attumambaEnc':
        model = AttUMambaEnc(input_size=(256,256),
                 n_stages=7,
                 features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                 kernel_sizes=3,
                 strides=(1, 2, 2, 2, 2, 2, 2),
                 n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6))
        return model.apply(InitWeights_He(1e-2))
    elif choice == 'attumambaBot':
        model = AttUMambaBot(input_size=(256,256),
                 n_stages=7,
                 features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                 kernel_sizes=3,
                 strides=(1, 2, 2, 2, 2, 2, 2),
                 n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6))
        return model.apply(InitWeights_He(1e-2))
    elif choice == 'attumambaEnc_2':
        model = UMambaEnc_2(input_size=(256,256), input_channels=3, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_conv_per_stage=[1, 3, 4, 6, 6, 6, 6], num_classes=1,
                              n_conv_per_stage_decoder=[1, 1, 1, 1, 1, 1],
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={},
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False)
        return model.apply(InitWeights_He(1e-2))
    elif choice == 'attumambaBot_2':
        model = UMambaBot_2(input_channels=3, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_conv_per_stage=[1, 3, 4, 6, 6, 6, 6], num_classes=1,
                              n_conv_per_stage_decoder=[1, 1, 1, 1, 1, 1],
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={},
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False)
        return model.apply(InitWeights_He(1e-2))
    elif choice == 'umambaBot':
        model = UMambaBot(input_channels=3, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_conv_per_stage=[1, 3, 4, 6, 6, 6, 6], num_classes=1,
                              n_conv_per_stage_decoder=[1, 1, 1, 1, 1, 1],
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={},
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False)
        return model.apply(InitWeights_He(1e-2))
    elif choice == 'umambaEnc':
        model = UMambaEnc(input_size=(256,256), input_channels=3, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_conv_per_stage=[1, 3, 4, 6, 6, 6, 6], num_classes=1,
                              n_conv_per_stage_decoder=[1, 1, 1, 1, 1, 1],
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={},
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False)
        return model.apply(InitWeights_He(1e-2))
    else:
        raise ValueError(f"Model choice '{choice}' is not supported.")

def plot_training_curve(path_to_log_files, save_img_loc):
    train_loss, val_loss, epochs=[],[],[]
    
    if isinstance(path_to_log_files, str):
        path_to_log_files = [path_to_log_files]

    for log_file in path_to_log_files:
        with open(log_file, 'r') as file:
            for line in file:
                part=line.strip().split()
                if 'Train Loss' in line:
                    train_loss.append(float(part[2]))
                if 'Val. Loss' in line:
                    val_loss.append(float(part[2]))
                if 'Epoch:' in line:
                    epochs.append(int(part[1]))

    epochs = range(1, len(epochs)+1)

    fig, ax=plt.subplots()
    ax.plot(epochs, train_loss, label='Train Loss')
    ax.plot(epochs, val_loss, label='Validation Loss')
    ax.set(title=f'history', ylabel= 'Loss', xlabel='Epochs')
    ax.legend()
    plt.savefig(save_img_loc)
    plt.close()

if __name__ == '__main__':
    model = ModelZoo('deeplabv3resnet101_cbam_block')