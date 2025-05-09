from ultralytics import YOLO
import os
from Preparedata import cls_prepare_data

class ClsModel:
    def __init__(self, data_dir, model_save_dir):
        self.model = YOLO("yolov8x-cls.pt")
        self.dataset_dir = data_dir
        self.model_save_dir = model_save_dir

    def train_model(self,path,name, prepare_data=False):
        if prepare_data==True:    
            cls_prepare_data(self.dataset_dir)

        #includes default augmentations in pipeline
        self.model.train(data=path, imgsz=512, device=0, 
                            epochs=400, batch=16, cos_lr=True,
                            patience=50, dropout=0.5, val=True,
                            project=self.model_save_dir, name=name+'train')

        best_model_path = os.path.join(self.model_save_dir, name, 'weights', 'best.pt')
        print(f"Best model saved at: {best_model_path}")
        
        return best_model_path
    
    def eval_model(self, best_model_path, name):

        model=YOLO(best_model_path)
        results=model.val(data='BaseSeg/syntax1/test', project=self.model_save_dir, name=name+'val')

        print(f'Test results:{results}')
        return results

if __name__ == '__main__':
    model = ClsModel(data_dir='arcade/syntax/', model_save_dir='BaseSeg/models/cls/')
    best_model_path = model.train_model(path='BaseSeg/syntax1', name='no_pp', prepare_data=False)
    results = model.eval_model(best_model_path='BaseSeg/models/cls/pp/weights/best.pt', name='pp')
