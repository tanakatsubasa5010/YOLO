from cross_validation import *
from ultralytics import YOLO
import torch

if __name__ == '__main__':

    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp0\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11n\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11s\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11m\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11l\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11x\weights\best.pt')
    model.val(data=data, batch=2, workers=0)


    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp1\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11n2\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11s2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11m2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11l2\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11x2\weights\best.pt')
    model.val(data=data, batch=2, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp2\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11n3\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11s3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11m3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11l3\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11x3\weights\best.pt')
    model.val(data=data, batch=2, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp3\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11n4\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11s4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11m4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11l4\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11x4\weights\best.pt')
    model.val(data=data, batch=2, workers=0)



    data = r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp4\test.yaml"

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11n5\weights\best.pt')
    model.val(data=data, workers=0)
    
    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11s5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11m5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11l5\weights\best.pt')
    model.val(data=data, workers=0)

    torch.cuda.empty_cache()
    model = YOLO(r'runs\detect\yolo11x5\weights\best.pt')
    model.val(data=data, batch=2, workers=0)