import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    model = YOLO(r'cfg/models/mymodel/all/two69.yaml')
    model.train(data=r'ultralytics/cfg/datasets/coco.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,  # 是否是单类别检测                
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',
                #lr0=0.0001,
                amp=False,
                resume=False,
                save_period=1,
               # project='runs',
                name='runs/coco_root/two69',
              #  resume='runs/train/v11s/exp2/weights/last.pt',
                )