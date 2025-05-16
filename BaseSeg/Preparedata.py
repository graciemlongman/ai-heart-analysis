import os
import json
import shutil
import numpy as np
from collections import defaultdict
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
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

def copy_image(src, dst):
    if os.path.exists(src):
        shutil.copy(src, dst)

def classify_l_r(file_store):
    label_store = defaultdict(list)
    rca_ids = {1, 2, 3, 4, 20, 21, 22, 23}
    processed_ids = set()

    for split, file in file_store.items():
        for ann in file['annotations']:
            image_id = ann['image_id']
            if (split, image_id) not in processed_ids:
                if ann['category_id'] in rca_ids:
                    label = 'RCA'
                else:
                    label = 'LCA'
                label_store[split, image_id].append(label)
                processed_ids.add((split, image_id))
    return label_store

def write_yaml(dataset_new_path, denom):
    home_path='/home/lunet/nc0051/PROJECT/ai-heart-analysis'
    if denom=='LCA':
        with open(f'{dataset_new_path}/data.yaml', 'w', encoding='utf-8') as file:
            file.write(f"""train: {home_path}/{dataset_new_path}/images/train
val: {home_path}/{dataset_new_path}/images/val
nc: 17
names: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25]
""")
    else:
        with open(f'{dataset_new_path}/data.yaml', 'w', encoding='utf-8') as file:
            file.write(f"""train: {home_path}/{dataset_new_path}/images/train
val: {home_path}/{dataset_new_path}/images/val
nc: 8
names: [1, 2, 3, 4, 20, 21, 22, 23]
""")


def seg_prepare_data(dataset_dir='arcade/syntax/', preprocess=False):
    
    file_store = load_annotations(dataset_dir)
    label_store = classify_l_r(file_store)
    print(label_store['train',922][0]=='LCA')
    
    dataset_newL= 'BaseSeg/syntaxLCA'
    dataset_newR='BaseSeg/syntaxRCA'
    
    #make dataset directories
    for dataset_new in ['BaseSeg/datasets/syntaxLCA', 'BaseSeg/datsets/syntaxRCA']:
        if os.path.exists(dataset_new)==False:
            os.makedirs(dataset_new)
            write_yaml(dataset_new, denom=str(dataset_new[-3:]))
            for type in ['images', 'labels']:
                os.makedirs(f'{dataset_new}/{type}')
                for split in ['train', 'test','val']:
                    os.makedirs(f'{dataset_new}/{type}/{split}/')
    
    # adapted from https://github.com/cmctec/ARCADE/blob/main/useful%20scripts/coco_to_yolo.ipynb
    for split, file in file_store.items():
        name_anns=defaultdict(list)
        name_cls= {img['id']: img['file_name'] for img in file['images']}
        #format the annotations
        for ann in file['annotations']:
            annotation = np.array(ann["segmentation"][0])
            annotation[0::2] /= file["images"][ann["image_id"]-1]["width"]
            annotation[1::2] /= file["images"][ann["image_id"]-1]["height"]
            name_anns[name_cls[ann["image_id"]]].append(str(ann["category_id"]) + " " + str(list(annotation)).replace("[", "").replace("]", "").replace(",", ""))
        
        for k, v in name_anns.items():
            filename = os.path.splitext(k)[0]
            id = [key for key, val in name_cls.items() if val==k]
            print(split, int(filename), id[0], v[0][0])
            if label_store[str(split), id[0]][0] == 'LCA':
                print('LCA')
                with open(f"{f'{dataset_newL}/labels/{split}/{filename}.txt'}", "w", encoding="utf-8") as file:
                    file.write("\n".join(v))
            else:
                with open(f"{f'{dataset_newR}/labels/{split}/{filename}.txt'}", "w", encoding="utf-8") as file:
                    file.write("\n".join(v))
        # end of adaption

        #copy images into correct file structures
        for img_id, filename in name_cls.items():
            src = f'{dataset_dir}{split}/images/{filename}'
            if label_store[split, img_id][0] == 'LCA':
                dst = f'{dataset_newL}/images/{split}/'
                copy_image(src,dst)
                if preprocess:
                    preprocess_inplace(f'{dst}{filename}')
            else:
                dst = f'{dataset_newR}/images/{split}/'
                copy_image(src,dst)
                if preprocess:
                    preprocess_inplace(f'{dst}{filename}')

        

def cls_prepare_data(dataset_dir='arcade/syntax/', dataset_new='BaseSeg/datasets/syntax1', preprocess=False):

    file_store = load_annotations(dataset_dir)

    #based on class labels, go through each image and decide if LCA or RCA
    #RCA: 1 2 3 4 16 16a 16b 16c
    label_store = classify_l_r(file_store)
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
                if preprocess:
                    preprocess_inplace(f'{dst}{image_id}.png')
    return label_store

if __name__ == '__main__':
    file_store=load_annotations(dataset_dir='arcade/syntax/')
    label_store = classify_l_r(file_store=file_store)
    cls_prepare_data(preprocess=False)

