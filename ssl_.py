import torch
import torch.utils
import torch.utils.data
import torchvision
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np


class Classify(torch.nn.Module):
    def __init__(self, backbone, c, out_c=None, h=2048):
        super(Classify, self).__init__()

        self.c = c
        self.out_c = out_c
        if out_c is None:
            self.out_c = self.c
        self.backbone = backbone
        self.avepool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.c, h), 
            torch.nn.ReLU(inplace=True), 
            torch.nn.Linear(h, self.out_c)
        )

    def forward(self, x):
        return self.sub_forward(x)

    def sub_forward(self, x):               # (N, 3, 640, 640)
        x = self.backbone(x)                # (N, C,  20,  20)
        x = self.avepool(x)                 # (N, C,   1,   1)
        x = torch.reshape(x, (-1, self.c))  # (N, C)
        x = self.mlp(x)                     # (N, out_C)
        return x

    def loss(self, x, t):
        loss = torch.nn.functional.cross_entropy(x, t)
        return loss


class SimSiam(torch.nn.Module):
    def __init__(self, backbone, c, out_c=None, h=2048):
        super(SimSiam, self).__init__()

        self.c = c
        self.out_c = out_c
        if out_c is None:
            self.out_c = self.c
        self.backbone = backbone
        self.avepool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(self.c, h), 
            torch.nn.ReLU(inplace=True), 
            torch.nn.Linear(h, self.c)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self.c, h), 
            torch.nn.ReLU(inplace=True), 
            torch.nn.Linear(h, self.c)
        )

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(640), 
            torchvision.transforms.ColorJitter()])

    def forward(self, x):
        x0 = self.transform(x)
        x1 = self.transform(x)
        return *self.sub_forward(x0), *self.sub_forward(x1)

    def sub_forward(self, x):               # (N, 3, 640, 640)
        x = self.backbone(x)                # (N, C,  20,  20)
        x = self.avepool(x)                 # (N, C,   1,   1)
        x = torch.reshape(x, (-1, self.c))  # (N, C)
        o1 = self.mlp1(x)                   # (N, C)
        o2 = self.mlp2(o1)                  # (N, C)
        return o1.detach(), o2

    def loss(self, x, t):
        o11, o12, o21, o22 = x
        loss1 = - torch.nn.functional.cosine_similarity(o11, o22)
        loss2 = - torch.nn.functional.cosine_similarity(o12, o21)
        return ((loss1 + loss2) / 2).mean()


class LinearEval(torch.nn.Module):
    def __init__(self, backbone):
        super(LinearEval, self).__init__()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = glob(os.path.join(path, 'train', 'images', '*.jpg'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = to_tensor(cv2.imread(self.data[idx]))
        return data
    
def collate_fn(batch):
    images = torch.stack(batch)
    return images, None



class ClassifyDataset(torchvision.datasets.ImageFolder):
    def __init__(self, path):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.RandomResizedCrop(640), 
            torchvision.transforms.ColorJitter()])
        super().__init__(path, transform=transform)


def train(ssl, backbone, c, dataset, out_c=None, epoch=100, device=torch.device('cuda'), collate_fn=None):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    ssl = ssl(backbone, c, out_c).to(device)
    backbone = backbone.to(device)
    optim = torch.optim.AdamW(ssl.parameters())

    for e in range(epoch):
        desc = 'epoch: {: >4} / {:>4}'.format(e + 1, epoch)
        with tqdm(total=len(dataloader)) as pbar:
            loss = 0
            for i, (img, tgt) in enumerate(dataloader):
                img = img.to(device)
                if tgt is not None:
                    tgt = tgt.to(device)

                out = ssl(img)
                criterion = ssl.loss(out, tgt)
                optim.zero_grad()
                criterion.backward()
                optim.step()
                
                loss = (i * loss + criterion.item()) / (1 + i)
                pbar.set_description(desc + '  loss: {:.3f}'.format(loss))
                pbar.update()
    torch.save(backbone.cpu().state_dict(), 'backbone.pt')


# train(ssl=SimSiam , backbone=YOLO('yolo11n.yaml').model.model[:11], c=256, dataset=Dataset(r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp0"), epoch=20, collate_fn=collate_fn)
# train(ssl=Classify, backbone=YOLO('yolo11n.yaml').model.model[:11], c=256, out_c=3, dataset=ClassifyDataset(r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp0\train_classify"), epoch=100)

model = YOLO('yolo11n.yaml')
model.train(data=r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp0\data.yaml", batch=8, epochs=20, patience=500, workers=0, name='sim_siam')
