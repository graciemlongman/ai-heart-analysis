from ultralytics import YOLO
import os
from Preparedata import cls_prepare_data

class ClsModel:
    def __init__(self, data_dir, model_save_dir):
        self.model = YOLO("yolov8x-cls.pt")
        self.dataset_dir = data_dir
        self.model_save_dir = model_save_dir

    def train_model(self, ):
        data = cls_prepare_data(self.dataset_dir)

#         #includes default augmentations in pipeline
#         results = self.model.train(data=data, imgsz=512, device=0, 
#                             epochs=400, batch=16, cos_lr=True,
#                             patience=50, dropout=0.5, val=True,
#                             project=self.model_save_dir, name='cad_cls')

#         best_model_path = os.path.join(self.model_save_dir, 'cad_cls', 'weights', 'best.pt')
#         print(f"Best model saved at: {best_model_path}")
        
#         return best_model_path
    
#     def eval_model(self, best_model_path):

#         model=YOLO(best_model_path)
#         results=model.val(data=test_dir)

#         print(f'Test results:{results}')
#         return results


#dynamically adjust learning rate - not speced - default? use cos_lr for now?
#weight-decay - not specified - use default at 0.0005?
#momentum - not specifiec - use default at 0.937?
#validation yes
#early stopping yes
#patience yes
#dropout yes
if __name__ == '__main__':
    ClsModel(data_dir='arcade/syntax/', model_save_dir='BaseSeg/models/cls/')
