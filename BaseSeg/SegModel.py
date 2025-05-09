from ultralytics import YOLO
import os
from Preparedata import seg_prepare_data

# prepare data
# polynomial learning rate scheduler?
# segment LCA and RCA separately, then ensemble results

class SegModels:
    def __init__(self, data_dir, model_save_dir):
        self.yolo_model=YOLO("yolov8x-seg.pt")
        self.model = self.yolo_model.model
        self.dataset_dir = data_dir
        self.model_save_dir = model_save_dir

    def train_model(self, path, name, model_version, prepare_data=False):

        if prepare_data == True:
            path = seg_prepare_data(self.dataset_dir)
        
        patience = 250 if model_version=='LCA' else 200

        #includes default augmentations in pipeline
        results = self.model.train(data=path, imgsz=512, device=0, 
                            epochs=600, batch=16, momentum=0.9, lr0=0.000714,
                            patience=patience, dropout=0.5, val=True,
                            project=self.model_save_dir, name=name)
        return results
    
    def eval_model(self, best_model_path):
        model=YOLO(best_model_path)
        results=model.val(data='BaseSeg/syntax1/test', project=self.model_save_dir, name=name+'val')

        print(f'Test results:{results}')
        return results
    
    def ensemble(self):
        return 


    