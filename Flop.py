from ultralytics import YOLO

model = YOLO('cfg/models/mymodel/all/tree6912-all.yaml')

del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.fuse()