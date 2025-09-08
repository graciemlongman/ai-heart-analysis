from ultralytics import YOLO
import os
from Preparedata import cls_prepare_data

# written by me

class ClsModel:
    def __init__(self, data_dir, model_save_dir):
        self.model = YOLO("yolov8x-cls.pt")
        self.dataset_dir = data_dir
        self.model_save_dir = model_save_dir

    def train_model(self,path,name, prepare_data=False):
        if prepare_data==True:    
            cls_prepare_data(preprocess=True)

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
        results=model.val(data='BaseSeg/datasets/syntax1/test', project=self.model_save_dir, name=name+'val')

        print(f'Test results:{results}')
        return results

if __name__ == '__main__':
    model = ClsModel(data_dir='arcade/syntax/', model_save_dir='BaseSeg/models/cls/')
    model.train_model(path='BaseSeg/datasets/syntax1', name='second', prepare_data=True)
    best_model = YOLO('BaseSeg/models/cls/firsttrain/weights/best.pt')
    results = best_model('BaseSeg/datasets/syntax1/test/ens', save=True, project='BaseSeg/runs/classify/predict/', name='cls')
