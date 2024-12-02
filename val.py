from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':

    data = r"C:\Users\tanaka\dataset\cross_validation\detection\yeast\temp0\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8n\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8s\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8m\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8l\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8x\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\yeast\temp1\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8n2\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8s2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8m2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8l2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8x2\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\yeast\temp2\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8n3\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8s3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8m3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8l3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8x3\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\yeast\temp3\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8n4\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8s4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8m4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8l4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8x4\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\yeast\temp4\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8n5\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8s5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8m5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8l5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yeast\yeast_only_yolov8x5\weights\best.pt')
    model.val(data=data, workers=0)