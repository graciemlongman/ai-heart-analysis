import os
import json
#from Preprocess import preprocess

def cls_prepare_data(dataset_dir='arcade/syntax/'):
    if dataset_dir is None:
        raise ValueError("No Data")
    
    #load in annotations
    file_store=[]
    label_store={}
    for i in ['train', 'test', 'val']:
        with open(dataset_dir+f'{i}/annotations/{i}.json', encoding="utf-8") as file:
            anns = json.load(file)
            file_store.append(anns['annotations'])
    
    #based on class labels, go through each image and decide if LCA or RCA
    #RCA: 1 2 3 4 16 16a 16b 16c
    for file in file_store:
        for ann in file:
            image_id = ann['image_id']
            print(image_id)
            if str(ann['category_id']) in ['1','2','3','4','16','16a','16b','16c']:
                label='RCA'
            else:
                label='LCA'
            label_store[image_id]=[label]
    print(len(file_store))
    print(len(label_store))
    
    # #make directory for train/val
    # for split in ['train', 'val']:
    #     if os.path.exists(dataset_dir+f'{split}/labels/') == False:
    #         os.makedirs(dataset_dir+f'{split}/labels/')
    #         #write labels to file
    #         if split == 'train':
    #             labels=label_store[:1000] 
    #         else:
    #             labels=label_store[1200:1500]
    #         for f, l in enumerate(labels): 
    #             with open(dataset_dir+f'{split}/labels/{f}.txt', 'w', encoding='utf-8') as file:
    #                 file.write(str(l))



    # # preprocess all the images as well

    # # Create dataset.yaml file
    # yaml_content = f"""
    # train: {dataset_dir} + 'train/images'
    # val: {dataset_dir} + 'val/images'
    # test: {dataset_dir} + 'test/images'
    
    # nc: 2
    # names: ['LCA', 'RCA']
    # """
    
    # yaml_path = os.path.join(dataset_dir, "baseseg1_dataset.yaml")
    # with open(yaml_path, 'w') as f:
    #     f.write(yaml_content)
    
    # print(f"Dataset prepared at {dataset_dir}")
    # print(f"Training: {len(train_lca)} LCA, {len(train_rca)} RCA")
    # print(f"Validation: {len(val_lca)} LCA, {len(val_rca)} RCA")
    # print(f"Testing: {len(test_lca)} LCA, {len(test_rca)} RCA")
    
    #return yaml_path

if __name__ == '__main__':
    cls_prepare_data()
