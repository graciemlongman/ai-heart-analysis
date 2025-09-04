import os
from utils.utils import create_file, print_and_save

# Script which extracts the outputted metrics from the relevent results.txt 
# files generated after inference

# find . -type f -name "results.txt"

orig_exp_paths = ['./stenExp/model_runs/deeplabv3resnet101/one/results/results.txt',
                  './stenExp/model_runs/attentionunet/one/results/results.txt',
                  './stenExp/model_runs/yolov8x-seg/one/CLI_SUMMARY.txt',
                  './stenExp/model_runs/resunetplusplus/four/results/results.txt',
                  './stenExp/model_runs/transunet/one/results/results.txt']

opt_paths = ['./stenExp/model_runs/deeplabv3resnet101/RMSprop/results/results.txt',
                    './stenExp/model_runs/deeplabv3resnet101/one/results/results.txt',
                    './stenExp/model_runs/deeplabv3resnet101/SGD/results/results.txt',
                    './stenExp/model_runs/attentionunet/RMSprop/results/results.txt',
                    './stenExp/model_runs/attentionunet/one/results/results.txt',
                    './stenExp/model_runs/attentionunet/SGD/results/results.txt',
                    './stenExp/model_runs/yolov8x-seg/RMSprop/CLI_SUMMARY.txt',
                    './stenExp/model_runs/yolov8x-seg/adam/CLI_SUMMARY.txt',
                    './stenExp/model_runs/yolov8x-seg/SGD/CLI_SUMMARY.txt',
                    './stenExp/model_runs/resunetplusplus/RMSprop/results/results.txt',
                    './stenExp/model_runs/resunetplusplus/adam/results/results.txt',
                    './stenExp/model_runs/resunetplusplus/SGD/results/results.txt',
                    './stenExp/model_runs/transunet/RMSprop/results/results.txt',
                    './stenExp/model_runs/transunet/one/results/results.txt',
                    './stenExp/model_runs/transunet/SGD/results/results.txt',]

pp_paths = ['./stenExp/model_runs/attentionunet/postprocexp/thresh_25/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_50/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_75/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_100/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_125/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_150/results/results.txt']

dlv3_paths = ['./stenExp/model_runs/deeplabv3resnet101/RMSprop/results/results.txt',
              './stenExp/model_runs/deeplabv3resnet101_se/RMSprop/results/results.txt',
              './stenExp/model_runs/deeplabv3resnet101_df/RMSprop/results/results.txt',
              './stenExp/model_runs/deeplabv3resnet101_cbam_block/RMSprop/results/results.txt',
              './stenExp/model_runs/deeplabv3resnet101_cbam_class/RMSprop/results/results.txt']

mod_paths = ['./stenExp/model_runs/attentionunet/Adam/results/results.txt',
             './stenExp/model_runs/aunet1/Adam/results/results.txt',
             './stenExp/model_runs/aunet2/Adam/results/results.txt',
             './stenExp/model_runs/aunet3/Adam/results/results.txt',
             './stenExp/model_runs/aunet4/Adam/results/results.txt',]

mamba_paths = ['./stenExp/model_runs/nnU-MambaBot_final/one/results.txt',
                './stenExp/model_runs/nnU-attMambaBot_final/one/results.txt',
                './stenExp/model_runs/nnU-attMambaBot1_final/one/results.txt',

                './stenExp/model_runs/LKM-UNet_final/one/results.txt',
                './stenExp/model_runs/LKMUNetBOT_final/one/results.txt']

bb_paths = ['./stenExp/model_runs/bbunet/Adam/results/results.txt',
            './stenExp/model_runs/bbaunet/Adam/results/results.txt',
            './stenExp/model_runs/bbunet_bb_in_x3_only/Adam/results/results.txt',
            './stenExp/model_runs/deeplabv3resnet101_bb/RMSprop/results/results.txt',]


""" Vars """
path_to_log_files = dlv3_paths
save_path = 'stenExp/scores/dlv3_scores.csv'
headings='Model,Jaccard,F1,Recall,Precision,Acc,F2,HD,MFPS,MSPF'

create_file(save_path)
print_and_save(save_path, headings, print_=False)

for log_file in path_to_log_files:
    splits = log_file.strip().split('/')
    model = splits[3]
    optimiser = 'adam' if splits[4] == 'one' else splits[4]
    threshold = splits[5]
    scores=[]
    with open(log_file, 'r') as file:
        for line in file:
            part=line.strip().split()
            if 'Jaccard' in line:
                scores.append(f'{model}, {part[1]}, {part[4]}, {part[7]}, {part[10]}, {part[13]}, {part[16]}, {part[19]}')
            
            mean_fps = part[2] if 'Mean FPS' in line else None

            mean_spf = part[2] if 'Mean SPF' in line else None

        print_and_save(save_path, f'{scores[0]},{mean_fps},{mean_spf}')
        print_and_save(save_path, f'{scores[1]},{mean_fps},{mean_spf}')
