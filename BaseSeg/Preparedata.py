import os
import json
import shutil
import numpy as np
from collections import defaultdict
import sys 
sys.path.append(os.path.expanduser('~/PROJECT/'))
from Preprocess import preprocess_inplace

def load_annotations(dataset_dir):
    if dataset_dir is None:
        raise ValueError("No Data")
    
    #load in annotations
    file_store={}
    for i in ['train', 'test', 'val']:
        with open(dataset_dir+f'{i}/annotations/{i}.json', encoding="utf-8") as file:
            anns = json.load(file)
            file_store[i]=anns 
    return file_store

def seg_prepare_data(dataset_dir='arcade/syntax/', dataset_new='BaseSeg/syntax2'):
    file_store = load_annotations(dataset_dir)

    #make dataset directories
    if os.path.exists(dataset_new)==False:
        os.makedirs(dataset_new)
        for type in ['images', 'labels']:
            os.makedirs(f'{dataset_new}/{type}')
            for split in ['train', 'test','val']:
                os.makedirs(f'{dataset_new}/{type}/{split}/')
    
    name_anns=defaultdict(list)
    for split, file in file_store.items():
        name_cls= {img['id']: img['file_name'] for img in file['images']}
        for ann in file['annotations']:
            annotation = np.array(ann["segmentation"][0])
            annotation[0::2] /= file["images"][ann["image_id"]-1]["width"]
            annotation[1::2] /= file["images"][ann["image_id"]-1]["height"]
            name_anns[name_cls[ann["image_id"]]].append(str(ann["category_id"]-1) + " " + str(list(annotation)).replace("[", "").replace("]", "").replace(",", ""))
        # where in the loop should this be ?
        src = f'{dataset_dir}{split}/images/{name_cls[ann["image_id"]]}'
        dst = f'{dataset_new}/images/{split}/'
        if os.path.exists(src):
            shutil.copy(src, dst)
            #preprocess_inplace(f'{dst}{name_cls[ann["image_id"]]}.png')
        
        # No such file or directory: 'labels/train922.txt'
        for k, v in name_anns.items():
            with open(f"labels/{split}" + k.replace("png", "txt"), "w", encoding="utf-8") as file:
                file.write("\n".join(v))
     


def cls_prepare_data(dataset_dir='arcade/syntax/', dataset_new='BaseSeg/syntax1'):

    file_store = load_annotations(dataset_dir)

    #based on class labels, go through each image and decide if LCA or RCA
    #RCA: 1 2 3 4 16 16a 16b 16c
    label_store=defaultdict(list)
    for split, file in file_store.items():
        print(split)
        prev='**'
        for ann in file['annotations']:
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
                preprocess_inplace(f'{dst}{image_id}.png')

if __name__ == '__main__':
    seg_prepare_data()
