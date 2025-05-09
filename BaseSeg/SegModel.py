from ultralytics import YOLO
import torch
import os

# prepare data
# polynomial learning rate scheduler?
# segment LCA and RCA separately, then ensemble results

class SegModel:
    def __init__(self, data_dir, model_save_dir):
        self.yolo_model=YOLO("yolov8x-seg.pt")
        self.model = self.yolo_model.model
        self.dataset_dir = data_dir
        self.model_save_dir = model_save_dir

    def train_model(self, data, model_version):
        
        patience = 250 if model_version=='LCA' else 200

        #prepare data
        #use coco to yolo for annotations
        #make yaml file

        #includes default augmentations in pipeline
        results = self.model.train(data=data, imgsz=512, device=0, 
                            epochs=600, batch=16, momentum=0.9, lr0=0.000714,
                            patience=patience, dropout=0.5, val=True,
                            project=self.model_save_dir, name='cad_cls')
        return results
    
    def eval_model(self, best_model_path):
        model=YOLO(best_model_path)
        results=model.val(data=test_dir)

        print(f'Test results:{results}')
        return results
    
    def ensemble(self):
        return 


    