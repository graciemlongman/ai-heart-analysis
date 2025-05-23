import cv2
import json
import numpy as np
from collections import defaultdict
import scipy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def store_images(json_file, path):
    store={}
    for img in json_file["images"]:
        image = cv2.imread(path + img['file_name'])
        store[img["id"]] = np.array(image)
    return store

# Adapted from https://github.com/cmctec/ARCADE/blob/main/useful%20scripts/create_masks.ipynb
def get_points(json_file):
    store = defaultdict(list)
    for ann in json_file['annotations']:
        points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], dtype=np.int32).T
        store[ann["image_id"], ann["category_id"]].append(points)
    return store

def get_masks(json_file, task, split):
    num_imgs = 301 if split=='test' else 201 if split=='val' else 1001
    if task =='syntax':
        masks = np.zeros((num_imgs, 26, 512, 512), dtype=np.uint8)  # Use uint8 for binary mask
    else:
        masks = np.zeros((num_imgs, 512, 512), dtype=np.uint8)

    for ann in json_file["annotations"]:
        points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], dtype=np.int32).T
        points = points.reshape((-1, 1, 2))
        tmp = np.zeros((512, 512), dtype=np.uint8)
        cv2.fillPoly(tmp, [points], (1))
        if task=='syntax':
            masks[ann["image_id"], ann['category_id']] |= tmp
        else:
            masks[ann["image_id"]] |= tmp # binary mask
    return masks
### end of adaptation

def plot_points(img_store, points_store, categories, name=False):
    plt.figure(figsize=(15,10))
    cat_colours = {category: plt.cm.hsv(i / 26) for i, category in enumerate(categories)}
    for idx, image_id in enumerate(range(1,13)):
        ax = plt.subplot(3, 4, idx + 1)
        plt.imshow(img_store[image_id])
        # Draw all polygons for this image
        for (img_id, cat_id), points_list in points_store.items():
            if img_id == image_id:
                for poly in points_list:
                    poly = np.vstack([poly, poly[0]])  # close the polygon
                    plt.plot(poly[:, 0], poly[:, 1], linewidth=2, color=cat_colours[cat_id], label=categories[cat_id])  # outline
                    if cat_id != 26:
                        plt.legend()

        plt.title(f"Image ID {image_id}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    if name:
        plt.savefig(name)

def plot_masks(filename, img_store, masks_store, task, name=False):
    plt.figure(figsize=(15, 10))
    for idx, ann in enumerate(filename["annotations"][:12]):
        ax = plt.subplot(3, 4, idx + 1)
        if task=='syntax':
            plt.imshow(img_store[ann['image_id']]) #plot img
            plt.imshow(masks_store[ann["image_id"], ann["category_id"]], alpha=0.5) #plot annotation(s)
            plt.title(f"Image ID {ann['image_id']} | Category {ann['category_id']}")
        else:
            plt.imshow(img_store[idx+1]) #plot img
            plt.imshow(masks_store[idx+1], alpha=0.5)
            plt.title(f'Image ID {idx+1}')
        
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    if name:
        plt.savefig(name)

def split_into_fg_bg(filename, img_store, masks):
    fimgs,bimgs=[],[]
    for ann in filename["annotations"]:
        image = np.array(img_store[ann['image_id']]) 
        mask = np.array(masks[ann['image_id'], ann["category_id"]])
        fimgs.append(image[mask==1])
        bimgs.append(image[mask==0])

    #flatten imgs and concatenate into one array
    fimgs = np.concatenate([fg.flatten() for fg in fimgs])
    bimgs = np.concatenate([bg.flatten() for bg in bimgs])
    return fimgs, bimgs

def calc_intensity_properties(arr):
    max = np.max(arr)
    mean = np.mean(arr)
    median = np.median(arr)
    min = np.min(arr)
    std = np.std(arr)
    return {'max':max, 'min':min, 'mean':mean, 'median':median, 'std':std}