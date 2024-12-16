from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    epochs = 500
    dataset = CrossValidationDataset(r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria", 3, 1, 1, augment=['identity'])
    for data, test in dataset:
        try:
            torch.cuda.empty_cache()
            model = YOLO('sod-ema-yolo11n.yaml')
            model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11n')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('sod-ema-yolo11s.yaml')
            model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11s')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('sod-ema-yolo11m.yaml')
            model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11m')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('sod-ema-yolo11l.yaml')
            model.train(data=data, batch=4, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11l')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('sod-ema-yolo11x.yaml')
            model.train(data=data, batch=2, epochs=epochs, patience=epochs, workers=0, name='sod-ema-yolo11x')
        except:
            pass



        try:
            torch.cuda.empty_cache()
            model = YOLO('yolo11n.yaml')
            model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11n')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('yolo11s.yaml')
            model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11s')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('yolo11m.yaml')
            model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11m')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('yolo11l.yaml')
            model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11l')
        except:
            pass

        try:
            torch.cuda.empty_cache()
            model = YOLO('yolo11x.yaml')
            model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='yolo11x')
        except:
            pass




        