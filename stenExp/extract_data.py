import os
from utils import create_file, print_and_save

opt_paths = ['./stenExp/model_runs/deeplabv3resnet101/RMSprop/results/results.txt',
                    './stenExp/model_runs/deeplabv3resnet101/one/results/results.txt',
                    './stenExp/model_runs/deeplabv3resnet101/SGD/results/results.txt',
                    './stenExp/model_runs/attentionunet/RMSprop/results/results.txt',
                    './stenExp/model_runs/attentionunet/one/results/results.txt',
                    './stenExp/model_runs/attentionunet/SGD/results/results.txt',
                    './stenExp/model_runs/yolov8x-seg/adam/CLI_SUMMARY.txt',
                    './stenExp/model_runs/yolov8x-seg/RMSprop/CLI_SUMMARY.txt',
                    './stenExp/model_runs/yolov8x-seg/SGD/CLI_SUMMARY.txt']

pp_paths = ['./stenExp/model_runs/attentionunet/postprocexp/thresh_25/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_50/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_75/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_100/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_125/results/results.txt',
            './stenExp/model_runs/attentionunet/postprocexp/thresh_150/results/results.txt']


path_to_log_files = pp_paths
save_path = 'stenExp/model_runs/attentionunet/postprocexp/pp_scores.csv'
headings='Model,Threshold,Jaccard,F1,Recall,Precision,Acc,F2,HD,MFPS,MSPF'

create_file(save_path)
print_and_save(save_path, headings, print_=False)

for log_file in path_to_log_files:
    splits = log_file.strip().split('/')
    model = splits[3]
    optimiser = 'adam' if splits[4] =='one' else splits[4]
    threshold = splits[5]
    scores=[]
    with open(log_file, 'r') as file:
        for line in file:
            part=line.strip().split()
            if 'Jaccard' in line:
                scores.append(f'{model}, {threshold}, {part[1]}, {part[4]}, {part[7]}, {part[10]}, {part[13]}, {part[16]}, {part[19]}')
            if 'Mean FPS' in line:
                mean_fps = part[2]
            if 'Mean SPF' in line:
                mean_spf = part[2]

        print_and_save(save_path, f'{scores[0]},{mean_fps},{mean_spf}')
        print_and_save(save_path, f'{scores[1]},{mean_fps},{mean_spf}')
