from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':

    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp0\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11n\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11s\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11m\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11l\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11x\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp1\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11n2\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11s2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11m2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11l2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11x2\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp2\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11n3\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11s3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11m3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11l3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11x3\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp3\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11n4\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11s4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11m4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11l4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11x4\weights\best.pt')
    model.val(data=data, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp4\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11n5\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11s5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11m5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11l5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\3bacteria\tla-yolo11x5\weights\best.pt')
    model.val(data=data, workers=0)