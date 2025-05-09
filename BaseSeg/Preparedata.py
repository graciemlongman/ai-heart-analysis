import os
import json
import shutil
from collections import defaultdict
import sys 
sys.path.append(os.path.expanduser('~/PROJECT/'))
from Preprocess import preprocess

def cls_prepare_data(dataset_dir='arcade/syntax/', dataset_new='BaseSeg/syntax1'):
    if dataset_dir is None:
        raise ValueError("No Data")
    
    #load in annotations
    file_store={}
    label_store=defaultdict(list)
    for i in ['train', 'test', 'val']:
        with open(dataset_dir+f'{i}/annotations/{i}.json', encoding="utf-8") as file:
            anns = json.load(file)
            file_store[i]=anns['annotations']
    
    #based on class labels, go through each image and decide if LCA or RCA
    #RCA: 1 2 3 4 16 16a 16b 16c
    for split, file in file_store.items():
        print(split)
        prev='**'
        for ann in file:
            image_id = ann['image_id']
            if prev != image_id:
                if str(ann['category_id']) in ['1','2','3','4','16','16a','16b','16c']:
                    label='RCA'
                else:
                    label='LCA'
                label_store[split,image_id].append(label)
                prev=image_id
            else:
                prev=image_id
                continue
    print(len(label_store))
    
    #make dataset directories
    if os.path.exists(dataset_new)==False:
        os.makedirs(dataset_new)
    for split in ['train', 'test','val']:
        for classi in ['LCA', 'RCA']:
            if os.path.exists(dataset_new+f'/{split}/{classi}/') == False:
                os.makedirs(dataset_new+f'/{split}/{classi}/')
    
    #copy images into appropriate folder
    for (split, image_id), labels in label_store.items():
        label = labels[0]
        if label in ['LCA', 'RCA']:
            src = f'{dataset_dir}/{split}/images/{image_id}.png'
            dst = f'{dataset_new}/{split}/{label}/'
            if os.path.exists(src):
                shutil.copy(src, dst)

if __name__ == '__main__':
    cls_prepare_data()
