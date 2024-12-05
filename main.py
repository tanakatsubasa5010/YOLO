from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    dataset = CrossValidationDataset(r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria", 3, 1, 1, augment=['identity'])
    for data, test in dataset:
        torch.cuda.empty_cache()
        model = YOLO('tla-yolo11n.yaml')
        model.train(data=data, batch=4, epochs=500, patience=500, workers=0, name='tla-yolo11n')
        
        torch.cuda.empty_cache()
        model = YOLO('tla-yolo11s.yaml')
        model.train(data=data, batch=4, epochs=500, patience=500, workers=0, name='tla-yolo11s')
        
        torch.cuda.empty_cache()
        model = YOLO('tla-yolo11m.yaml')
        model.train(data=data, batch=4, epochs=500, patience=500, workers=0, name='tla-yolo11m')
        
        torch.cuda.empty_cache()
        model = YOLO('tla-yolo11l.yaml')
        model.train(data=data, batch=4, epochs=500, patience=500, workers=0, name='tla-yolo11l')
        
        torch.cuda.empty_cache()
        model = YOLO('tla-yolo11x.yaml')
        model.train(data=data, batch=4, epochs=500, patience=500, workers=0, name='tla-yolo11x')