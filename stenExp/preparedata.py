import os
import json
import numpy as np
from utils import shuffling
import cv2
import shutil
from glob import glob
from collections import defaultdict
from preprocess import *

import torch
from torch.utils.data import Dataset, DataLoader


def copy_file(src, dst):
    if os.path.exists(src):
        shutil.copy(src, dst)

def write_yaml(path):
    home_path = '/home/lunet/nc0051/PROJECT/ai-heart-analysis'
    with open(f'{path}/data.yaml', 'w', encoding='utf-8') as file:
        file.write(f"train: {home_path}/{path}images/train\n")
        file.write(f"val: {home_path}/{path}images/val\n")
        file.write(f"nc: 1\n")
        file.write("names: ['stenosis']")


def make_directories(path_new, version='torch'):
    if version=='torch':
        if os.path.exists(path_new)==False:
            os.makedirs(path_new)
            for split in ['train', 'test','val']:
                for type in ['annotations', 'images']:
                    os.makedirs(f'{path_new}/{split}/{type}/')
    elif version == 'yolo':
        if os.path.exists(path_new)==False:
            os.makedirs(path_new)
            write_yaml(path_new)
            for type in ['images', 'labels']:
                os.makedirs(f'{path_new}/{type}')
                for split in ['train', 'test','val']:
                    os.makedirs(f'{path_new}/{type}/{split}/')
    else:
        raise ValueError(f"Dataset version '{version}' is not supported.")

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

def prepare_data_for_yolo(path='arcade/stenosis/', path_new='stenExp/datasets/arcade/yolo_stenosis/', preprocess=False):
    file_store = load_annotations(path)
    make_directories(path_new, version='yolo')

    # adapted from https://github.com/cmctec/ARCADE/blob/main/useful%20scripts/coco_to_yolo.ipynb
    for split, file in file_store.items():
        name_anns=defaultdict(list)
        name_cls= {img['id']: img['file_name'] for img in file['images']}
        #format the annotations
        for ann in file['annotations']:
            annotation = np.array(ann["segmentation"][0])
            annotation[0::2] /= file["images"][ann["image_id"]-1]["width"]
            annotation[1::2] /= file["images"][ann["image_id"]-1]["height"]
            
            name_anns[name_cls[ann["image_id"]]].append("0" + " " + str(list(annotation)).replace("[", "").replace("]", "").replace(",", ""))
        
        for k, v in name_anns.items():
            filename = os.path.splitext(k)[0]
            with open(f'{path_new}/labels/{split}/{filename}.txt', "w", encoding="utf-8") as file:
                    file.write("\n".join(v))

        # end of adaption
    
        #copy images into correct file structures
        for img_id, filename in name_cls.items():
            src = f'{path}{split}/images/{filename}'
            dst = f'{path_new}images/{split}/'

            print(src)
            print(dst)

            copy_file(src,dst)
            if preprocess:
                preprocess_inplace(f'{dst}{filename}')

def create_dataset(path='arcade/stenosis/', path_new='stenExp/datasets/arcade/stenosis/'):
    make_directories(path_new)
    
    for split in ['train', 'test', 'val']:
        image_paths = sorted(glob(os.path.join(path,split,'images', '*.png')))
        image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        dst_img=path_new+f'{split}/images/'
        json_src = path+f'{split}/annotations/{split}.json'
        json_dst=path_new+f'{split}/annotations/'
        
        for src_img in image_paths:
            print('source',src_img)
            print('dst', dst_img)
            copy_file(src_img, dst_img)
        copy_file(json_src, json_dst)

        for i, mask in enumerate(annotation_to_mask(os.path.join(path,split,f'annotations/{split}.json'), split=split)):
            mask=(mask* 255).astype(np.uint8)
            cv2.imwrite(os.path.join(json_dst,f'{i+1}.png'),mask)
    return path_new

def prepare_data_stenosis(path='stenExp/datasets/arcade/stenosis/', copy_data=False):
    if copy_data:
        create_dataset()
    for split in ['train', 'test', 'val']:
        for image_path in sorted(glob(os.path.join(path,split,'images', '*.png'))):
            preprocess_inplace(image_path)

            file = os.path.basename(image_path)
            if int(os.path.splitext(file)[0])%25==0:
                print(f'{split}/{file}')


def annotation_to_mask(path_to_json, split):
    num_imgs = 300 if split=='test' else 200 if split=='val' else 1000

    with open(path_to_json, encoding="utf-8") as file:
        file_store = json.load(file)
        annotations = file_store['annotations']

    name_cls= {img['id']: img['file_name'][:-4] for img in file_store['images']}
    

    masks = np.zeros((num_imgs, 512, 512), dtype=np.uint8)
    for ann in annotations:
        points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], dtype=np.int32).T
        points = points.reshape((-1, 1, 2))
        tmp = np.zeros((512, 512), dtype=np.uint8)
        cv2.fillPoly(tmp, [points], (1))
        masks[int(name_cls[ann["image_id"]])-1] |= tmp

    return masks 

#https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/train.py
def load_data(path='stenExp/datasets/arcade/stenosis/'):
    def get_data(path, split):
        """returns:
                images as a list of file paths
                labels as a list of file paths """

        images = sorted(glob(os.path.join(path, split, "images", "*.png")))
        labels = sorted(glob(os.path.join(path, split, "annotations", "*.png")))
        
        return images, labels
    
    """ Training data """
    train_x, train_y = get_data(path, split='train')

    """Validation data"""
    valid_x, valid_y = get_data(path, split='val')

    """Testing data"""
    test_x, test_y = get_data(path, split='test')

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

#https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/train.py
class ARCADE_DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]


        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0 

        return image, mask

    def __len__(self):
        return self.n_samples

def check_labels(train_loader, valid_loader, zipped_test):
    #zipped_test = zip(test_x, test_y)
    # PyTorch Example
    f='DEBUG_val'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks) in enumerate(valid_loader):
        # Show first sample in batch
        plt.imshow(images[0][0], cmap='gray')  # assuming shape [B, C, H, W]
        plt.imshow(masks[0][0], alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break

    f='DEBUG_train'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks) in enumerate(train_loader):
        # Show first sample in batch
        plt.imshow(images[0][0], cmap='gray')  # assuming shape [B, C, H, W]
        plt.imshow(masks[0][0], alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break

        # PyTorch Example
    f='DEBUG_test'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks) in enumerate(zipped_test):
        image = cv2.imread(images, cv2.IMREAD_COLOR)
        mask = cv2.imread(masks, cv2.IMREAD_GRAYSCALE)
        
        # Show first sample in batch
        plt.imshow(image, cmap='gray')  # assuming shape [B, C, H, W]
        plt.imshow(mask, alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break


# if __name__ =='__main__':
#     #preprocess(copy_data=True)
#     #annotation_to_mask('arcade/stenosis/train/annotations/train.json', split='train')

#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path='stenExp/datasets1/arcade/stenosis/')
#     train_x, train_y = shuffling(train_x, train_y)

#     data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
#     print(data_str)

#     import albumentations as A
#     transform = A.Compose([
#         A.Affine(scale=(0.7, 1.4), p=0.5),
#         A.Rotate(limit=180, p=0.5),
#         A.GaussNoise(std_range=(0.0, 0.32), p=0.5),
#         A.GaussianBlur(blur_limit=3, sigma_limit=(0.71, 1), p=0.5)
#         ])

#     train_dataset = ARCADE_DATASET(train_x, train_y, size=(256,256), transform=transform)
#     valid_dataset = ARCADE_DATASET(valid_x, valid_y, size=(256,256), transform=None)

#     bs=8
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=bs,
#         shuffle=True,
#         num_workers=2
#     )

#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=bs,
#         shuffle=False,
#         num_workers=2
#     )

#     print(f'Data Loaded. Batch size = {bs}')

#         # path = 'DEBUG2'
#     # os.makedirs(path, exist_ok=True)
#     # for i, arr in enumerate(train_y):
#     #     if len(np.where(arr==1))>0:
#     #         print(i,'yay')
#     #     plt.imshow(arr)
#     #     plt.savefig(os.path.join(f'{path}/{i}.png'))

#     #     arr_cv = (arr * 255).astype(np.uint8)
#     #     cv2.imwrite(os.path.join(path, f'{i}_cv2.png'), arr_cv)

#     # #print(train_y)
#     # import sys
#     # sys.exit()
        