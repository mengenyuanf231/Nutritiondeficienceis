import sys  ###
import ultralytics  ###
import os  ###
from ultralytics import YOLO
import ultralytics.models
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #### 
import matplotlib
# 设置matplotlib后端（避免显示问题）
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
# 配置字体，优先使用系统可用字体

model = YOLO("./weights/epoch198.pt")  ####
model.info()  #### 
model.val(    ####
              data='ultralytics/cfg/datasets/coco.yaml', 
              split='test',  
              imgsz=640,    
              batch=8,
                      device='0', ####
              save_json=False,
              name='runs/test/best',  
       )