from ultralytics import YOLO
import os,sys
from preparedata import *
from postprocess import *
from utils import *
from operator import add

class SegModel:
    def __init__(self, raw_data_dir, save_path):
        self.model=YOLO("yolov8x-seg.pt")
        self.raw_data = raw_data_dir
        self.model_save_dir = save_path
    
    def prepare_data(self, preprocess=False):
        prepare_data_for_yolo(self.raw_data, preprocess=preprocess)

    def train_model(self, path, name):
        #includes default augmentations in pipeline
        self.model.train(data=path, imgsz=512, device=0, optimizer='RMSprop',
                            epochs=500, batch=8, lr0=0.0001,
                            patience=50, dropout=0.5, val=True,
                            project=self.model_save_dir, name=name)
        best_model_path = os.path.join(self.model_save_dir, name, 'weights', 'best.pt')
        print(f"Best model saved at: {best_model_path}")
        return best_model_path

    def process_results(self, results, save_path):
        
        for item in ["mask", "joint", "procd_mask"]:
            if not os.path.exists(f"{save_path}/{item}"):
                os.makedirs(f"{save_path}/{item}")
            else:
                print('Results folder already exists')
                print('Check your directories :)')
                sys.exit()

        metrics, metrics_post=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        size=(256,256)
        for i, result in enumerate(results):
            ipath=result.path
            name = os.path.basename(ipath)
            mpath= f'stenExp/datasets/arcade/stenosis/test/annotations/{name}'

            image = cv2.resize(cv2.imread(ipath, cv2.IMREAD_COLOR),size)
            y_true=cv2.resize(cv2.imread(mpath, cv2.IMREAD_GRAYSCALE),size)

            if result.masks is None:
                y_pred = np.zeros(size, dtype=np.uint8)
            else:
                y_pred = np.zeros((512,512), dtype=np.uint8)
                for m in result:
                    tmp = np.zeros((512,512), dtype=np.uint8)
                    for points in m.masks.xy:
                        pts = points.astype(np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(tmp, [pts], (1,))
                    y_pred |= tmp

            y_post_pred = binary_remove_small_segments(cv2.resize(y_pred, (512,512)), 50)

            metrics_score = calculate_metrics(y_true, y_pred, y_true_proc='evaluate', y_pred_proc='postprocessed', size=size)
            metrics = list(map(add, metrics, metrics_score))
            
            metrics_score_post = calculate_metrics(y_true,y_post_pred, y_true_proc='evaluate', y_pred_proc='postprocessed', size=size)
            metrics_post = list(map(add, metrics_post, metrics_score_post))
            
            y_post_pred_3d = np.stack((cv2.resize(y_post_pred,size),)*3, axis=-1)*255
            y_pred_3d = np.stack((cv2.resize(y_pred,size),) * 3, axis=-1)*255
            y_true_3d=np.stack((y_true,) * 3, axis=-1)

            plot_true_vs_preds_to_file(size, save_path, name, image, y_true_3d, y_pred_3d, y_post_pred_3d)
        
        mean_score(metrics,num_imgs=300, print_=True)
        mean_score(metrics_post, num_imgs=300, print_=True)

    
if __name__=='__main__':
    model_choice='yolov8x-seg'
    name='RMSprop'
    
    model=SegModel(raw_data_dir='arcade/stenosis/', save_path=f'stenExp/model_runs/{model_choice}')
    # model.prepare_data(preprocess=True)
    model.train_model(path='stenExp/datasets/arcade/yolo_stenosis/data.yaml', name=name)
    
    best=YOLO(f'stenExp/model_runs/{model_choice}/{name}/weights/best.pt')
    results=best('stenExp/datasets/arcade/yolo_stenosis/images/test', save=False, project=f'stenExp/model_runs/{model_choice}/{name}', name='predict1')
    
    save_path=f'stenExp/model_runs/{model_choice}/{name}/results'
    model.process_results(results, save_path=save_path)