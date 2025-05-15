from ultralytics import YOLO
import os
from Preparedata import seg_prepare_data

class SegModel:
    def __init__(self, model_version, data_dir, model_save_dir):
        self.model=YOLO("yolov8x-seg.pt")
        self.model_version = model_version #ie LCA/RCA
        self.dataset_dir = data_dir
        self.model_save_dir = model_save_dir

    def train_model(self, path, name, prepare_data=False):

        if prepare_data == True:
            seg_prepare_data(self.dataset_dir, preprocess=True)
        
        patience = 250 if self.model_version=='LCA' else 200

        #includes default augmentations in pipeline
        self.model.train(data=path, imgsz=512, device=0, 
                            epochs=600, batch=16, momentum=0.9, lr0=0.000714, lrf=0.001,
                            patience=patience, dropout=0.5, val=True,
                            project=self.model_save_dir, name=name)
        best_model_path = os.path.join(self.model_save_dir, name, 'weights', 'best.pt')
        print(f"Best model saved at: {best_model_path}")
        
        return best_model_path
    
    def eval_model(self, name, best_model_path):
        model=YOLO(best_model_path)
        results=model.val(data='BaseSeg/syntax1/test', project=self.model_save_dir, name=name+'val')

        print(f'Test results:{results}')
        return results
    
    def ensemble(self):
        return 
    
if __name__=='__main__':
    modelLCA=SegModel('LCA', data_dir='arcade/syntax/', model_save_dir='BaseSeg/models/segL/')
    modelRCA=SegModel('RCA', data_dir='arcade/syntax', model_save_dir='BaseSeg/models/segR')

    # modelLCA.train_model(path='BaseSeg/syntaxLCA/data.yaml', name='first', prepare_data=False)
    # modelRCA.train_model(path='BaseSeg/syntaxRCA/data.yaml', name='first', prepare_data=False)

    lca_best_path = 'BaseSeg/models/segL/first/weights/best.pt'
    rca_best_path = 'BaseSeg/models/segR/first/weights/best.pt'

    best_LCAmodel = YOLO(lca_best_path)
    best_RCAmodel = YOLO(rca_best_path)

    #LCAresults = best_LCAmodel('BaseSeg/syntaxLCA/images/test', save=True, project='BaseSeg/runs/segment/predict', name='LCApred')
    #RCAresults = best_RCAmodel('BaseSeg/syntaxRCA/images/test', save=True, project='BaseSeg/runs/segment/predict', name='RCApred')


    