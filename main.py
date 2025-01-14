from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    epochs = 500
    dataset = CrossValidationDataset(r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria", 3, 1, 1, augment=['identity'])
    for data, test in dataset:
        torch.cuda.empty_cache()
        model = YOLO('yolo11n.yaml', learned_section='encoder.pt')
        model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='simsiam-yolo11n')

        torch.cuda.empty_cache()
        model = YOLO('yolo11n.yaml', learned_section='encoder.pt')
        model.train(data=data, freeze=11, batch=8, epochs=epochs, patience=epochs, workers=0, name='freeze-simsiam-yolo11n')
        
        # torch.cuda.empty_cache()
        # model = YOLO('yolov8s.yaml')
        # model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8s')
        
        # torch.cuda.empty_cache()
        # model = YOLO('yolov8m.yaml')
        # model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8m')
        
        # torch.cuda.empty_cache()
        # model = YOLO('yolov8l.yaml')
        # model.train(data=data, batch=8, epochs=500, patience=500, workers=0, name='yeast_only_yolov8l')
        
        # torch.cuda.empty_cache()
        # model = YOLO('yolo11x.yaml', learned_section='encoder.pt')
        # model.train(data=data, batch=8, epochs=epochs, patience=epochs, workers=0, name='simsiam-yolo11')

        # torch.cuda.empty_cache()
        # model = YOLO('yolo11x.yaml', learned_section='encoder.pt')
        # model.train(data=data, freeze=11, batch=8, epochs=epochs, patience=epochs, workers=0, name='freeze-simsiam-yolo11')