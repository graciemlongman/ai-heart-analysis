import os
import json
import numpy as np
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

def write_json(path):
    home_path = '/home/lunet/nc0051/PROJECT/ai-heart-analysis'
    data={ "channel_names": { 
            "0": "RGB", 
            # "1": "G",
            # "2": "B"
            }, 
            "labels": { 
            "background": 0,
            "stenosis": 1,
            }, 
            "numTraining": 1200, 
            "file_ending": ".nii.gz"
            }
    with open(f'{path}/dataset.json', 'w', encoding='utf-8') as file:
        json.dump(data, file)


def make_directories(path_new, version='torch'):
    if version=='torch':
        if os.path.exists(path_new)==False:
            os.makedirs(path_new)
            for split in ['train', 'test','val']:
                for type in ['annotations', 'images', 'boxes']:
                    os.makedirs(f'{path_new}/{split}/{type}/')
    elif version == 'yolo':
        if os.path.exists(path_new)==False:
            os.makedirs(path_new)
            write_yaml(path_new)
            for type in ['images', 'labels']:
                os.makedirs(f'{path_new}/{type}')
                for split in ['train', 'test','val']:
                    os.makedirs(f'{path_new}/{type}/{split}/')
    elif version == 'nnunet':
        if os.path.exists(path_new)==False:
            os.makedirs(path_new)
            write_json(path_new)
            for folder in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']:
                os.makedirs(f'{path_new}/{folder}')
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

import nibabel as nib
import numpy as np
from PIL import Image
def prepare_data_for_nnunet(path='arcade/stenosis/', path_new='Dataset111_ArcadeXCA/'):
    make_directories(path_new, version='nnunet')

    for split in ['train', 'test', 'val']:
        image_paths = sorted(glob(os.path.join(path,split,'images', '*.png')))
        image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        dst_img = f'{path_new}imagesTs' if split == 'test' else f'{path_new}imagesTr'
        dst_msk = f'{path_new}labelsTs' if split == 'test' else f'{path_new}labelsTr'

        #copy images and save as .nii.gz
        print('Moving images', split)
        for i, src_img in enumerate(image_paths):
            i_str = str(i + 1001) if split == 'val' else str(i + 1)
            num = i_str.zfill(4)
            img_np = np.array(Image.open(src_img).convert('RGB'))
            nib.save(nib.Nifti1Image(img_np.astype(np.uint8), affine=np.eye(4)),f'{dst_img}/arcade_{num}_0000.nii.gz')

        print('Moving Masks', split)
        for i, mask in enumerate(annotation_to_mask(os.path.join(path,split,f'annotations/{split}.json'), split=split)):
            mask=(mask).astype(np.uint8)
            mask=np.stack([mask,mask,mask],axis=-1)
            i_str = str(i + 1001) if split == 'val' else str(i+1)
            num = i_str.zfill(4)
            nib.save(nib.Nifti1Image(mask,affine=np.eye(4)), f'{dst_msk}/arcade_{num}.nii.gz')
    print('Fin')

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

def annotation_to_box(path_to_json, split):
    num_imgs = 300 if split=='test' else 200 if split=='val' else 1000

    with open(path_to_json, encoding="utf-8") as file:
        file_store = json.load(file)
        annotations = file_store['annotations']

    name_cls= {img['id']: img['file_name'][:-4] for img in file_store['images']}
    

    boxes = np.zeros((num_imgs, 512, 512), dtype=np.uint8)
    for ann in annotations:
        p = np.array(ann["bbox"], dtype=np.int32)
        points = np.array([[p[0], p[1]], [p[0]+p[2], p[1]], [p[0]+p[2],p[1]+p[3]], [p[0], p[1]+p[3]]], dtype=np.int32)
        # print(points)
        # sys.exit()
        #points = points.reshape((-1,1,2))
        tmp = np.zeros((512, 512), dtype=np.uint8)
        cv2.fillPoly(tmp, [points], (1))
        boxes[int(name_cls[ann["image_id"]])-1] |= tmp

    return boxes 

def create_dataset(path='arcade/stenosis/', path_new='stenExp/datasets/arcade/stenosis/'):
    make_directories(path_new)
    
    for split in ['train', 'test', 'val']:
        image_paths = sorted(glob(os.path.join(path,split,'images', '*.png')))
        image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        dst_img=path_new+f'{split}/images/'
        json_src = path+f'{split}/annotations/{split}.json'
        json_dst=path_new+f'{split}/annotations/'
        box_dst=path_new+f'{split}/boxes/'
        
        for src_img in image_paths:
            print('source',src_img)
            print('dst', dst_img)
            copy_file(src_img, dst_img)
        copy_file(json_src, json_dst)

        for i, mask in enumerate(annotation_to_mask(os.path.join(path,split,f'annotations/{split}.json'), split=split)):
            mask=(mask* 255).astype(np.uint8)
            cv2.imwrite(os.path.join(json_dst,f'{i+1}.png'),mask)
        
        for i, box in enumerate(annotation_to_box(os.path.join(path,split,f'annotations/{split}.json'), split=split)):
            box=(box* 255).astype(np.uint8)
            cv2.imwrite(os.path.join(box_dst,f'{i+1}.png'),box)

def prepare_data_stenosis(path='stenExp/datasets/arcade/stenosis/', copy_data=False):
    if copy_data:
        create_dataset()
    for split in ['train', 'test', 'val']:
        for image_path in sorted(glob(os.path.join(path,split,'images', '*.png'))):
            preprocess_inplace(image_path)


#https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/train.py
def load_data(path='stenExp/datasets/arcade/stenosis/', bbox=False):
    def get_data(path, split, bbox=False):
        """returns:
                images as a list of file paths
                labels as a list of file paths """

        images = sorted(glob(os.path.join(path, split, "images", "*.png")))
        labels = sorted(glob(os.path.join(path, split, "annotations", "*.png")))
        bboxes = sorted(glob(os.path.join(path, split, "boxes", "*.png")))
        return images, labels, bboxes
    
    train_x, train_y, train_b = get_data(path, split='train', bbox=bbox)
    valid_x, valid_y, valid_b = get_data(path, split='val', bbox=bbox)
    test_x, test_y, test_b = get_data(path, split='test', bbox=bbox)

    if bbox:
        return [(train_x, train_y, train_b), (valid_x, valid_y, valid_b), (test_x, test_y, test_b)]
    else:
        return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

#https://github.com/DebeshJha/ResUNetplusplus-PyTorch-/blob/main/train.py
class ARCADE_DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None, bbox=False, boxes_path=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)
        self.bbox = bbox
        self.boxes_path = boxes_path

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        if self.bbox:
            box = cv2.imread(self.boxes_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            if self.bbox:
                augmentations = self.transform(image=image, mask=mask, box=box)
                image = augmentations["image"]
                mask = augmentations["mask"]
                box = augmentations['box']
            else:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0 

        if self.bbox:
            box = cv2.resize(box, self.size)
            if box.ndim == 2:
                box = np.expand_dims(box, axis=0)
            box=box/255.0
            return image, mask, box
        else:
            return image, mask

    def __len__(self):
        return self.n_samples

from sklearn.utils import shuffle
from utils.utils import print_and_save
def data_loader(train_log_path, bbox, size, transform, batch_size):
    if bbox:
        print('Including bbox')
        (train_x, train_y, train_b), (valid_x, valid_y, valid_b), (test_x, test_y, test_b) = load_data(bbox=True)
        train_x, train_y, train_b = shuffle(train_x, train_y, train_b, random_state=42)
        
        train_dataset = ARCADE_DATASET(train_x, train_y, size, transform=transform, bbox=True, boxes_path=train_b)
        valid_dataset = ARCADE_DATASET(valid_x, valid_y, size, transform=None, bbox=True, boxes_path=valid_b)
    else:
        print('Loading data and initialising dataset and data loader...')
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()
        train_x, train_y = shuffle(train_x, train_y, random_state=42)

        train_dataset = ARCADE_DATASET(train_x, train_y, size, transform=transform)
        valid_dataset = ARCADE_DATASET(valid_x, valid_y, size, transform=None)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, valid_loader

def check_labels(train_loader, valid_loader, zipped_test):
    f='DEBUG_val'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks) in enumerate(valid_loader):

        plt.imshow(images[0][0], cmap='gray')  
        plt.imshow(masks[0][0], alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break

    f='DEBUG_train'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks) in enumerate(train_loader):

        plt.imshow(images[0][0], cmap='gray')
        plt.imshow(masks[0][0], alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break

    f='DEBUG_test'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks) in enumerate(zipped_test):
        image = cv2.imread(images, cv2.IMREAD_COLOR)
        mask = cv2.imread(masks, cv2.IMREAD_GRAYSCALE)

        plt.imshow(image, cmap='gray')
        plt.imshow(mask, alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break

def check_boxes(train_loader, valid_loader):
    f='DEBUG_val'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks, b) in enumerate(valid_loader):

        plt.imshow(images[0][0], cmap='gray')  
        plt.imshow(b[0], alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break

    f='DEBUG_train'
    os.makedirs(f, exist_ok=True)
    for i, (images, masks, b) in enumerate(train_loader):

        plt.imshow(images[0][0], cmap='gray')
        plt.imshow(b[0], alpha=0.4, cmap='Reds')
        plt.title('Batch Sample Overlay valid')
        plt.show()
        plt.savefig(os.path.join(f'{f}/{i}.png'))
        if i==20:
            break

if __name__  == '__main__':
    #prepare_data_for_nnunet()
    create_dataset()