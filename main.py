from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    epochs = 1
    dataset = CrossValidationDataset(r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria", 3, 1, 1, augment=['identity'])
    for data, test in dataset:
        torch.cuda.empty_cache()
        model = YOLO('sod-ema-yolo11n.yaml')
        model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11n')
        
        torch.cuda.empty_cache()
        model = YOLO('sod-ema-yolo11s.yaml')
        model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11s')
        
        torch.cuda.empty_cache()
        model = YOLO('sod-ema-yolo11m.yaml')
        model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11m')
        
        torch.cuda.empty_cache()
        model = YOLO('sod-ema-yolo11l.yaml')
        model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11l')
        
        torch.cuda.empty_cache()
        model = YOLO('sod-ema-yolo11x.yaml')
        model.train(data=data, batch=2, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11x')



        torch.cuda.empty_cache()
        model = YOLO('yolo11n.yaml')
        model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11n')
        
        torch.cuda.empty_cache()
        model = YOLO('yolo11s.yaml')
        model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11s')
        
        torch.cuda.empty_cache()
        model = YOLO('yolo11m.yaml')
        model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11m')
        
        torch.cuda.empty_cache()
        model = YOLO('yolo11l.yaml')
        model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11l')
        
        torch.cuda.empty_cache()
        model = YOLO('yolo11x.yaml')
        model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11x')




        