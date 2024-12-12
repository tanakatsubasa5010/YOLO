from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    epochs = 1
    dataset = CrossValidationDataset(r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria", 3, 1, 1, augment=['identity'])
    for data, test in dataset:
        torch.cuda.empty_cache()
        model = YOLO('yolov8n.yaml')
        model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8n')
        
        torch.cuda.empty_cache()
        model = YOLO('yolov8s.yaml')
        model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8s')
        
        torch.cuda.empty_cache()
        model = YOLO('yolov8m.yaml')
        model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8m')
        
        torch.cuda.empty_cache()
        model = YOLO('yolov8l.yaml')
        model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8l')
        
        torch.cuda.empty_cache()
        model = YOLO('yolov8x.yaml')
        model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8x')