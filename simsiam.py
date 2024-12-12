# https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py#L159

import torch
from torch import nn
import torch.utils.data
from torchvision import transforms
import torchvision.datasets as datasets
from ultralytics import YOLO
import math
from tqdm import tqdm
from PIL import ImageFilter
import random
from glob import glob
import os
import cv2
from torchvision.io import write_jpeg


class SimSiam(nn.Module):
    def __init__(self, encoder, prev_dim, pred_dim=512, encode_dim=2048):
        super(SimSiam, self).__init__()

        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.predictor1 = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), 

                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), 

                                        nn.Linear(prev_dim, encode_dim, bias=False), 
                                        nn.BatchNorm1d(encode_dim, affine=False))

        self.predictor2 = nn.Sequential(nn.Linear(encode_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(pred_dim, encode_dim))
        
    def forward(self, x1, x2):
        p1, z1 = self.sub_forward(x1)
        p2, z2 = self.sub_forward(x1)

        return p1, p2, z1, z2
    
    def sub_forward(self, x):
        t = self.encoder(x)
        t = self.pool(t).reshape(t.shape[0], -1)
        z = self.predictor1(t)
        p = self.predictor2(z)

        return p, z.detach()


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(640, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        return [x1, x2]
    

def collate_fn(batch):
    imgs1, imgs2= list(zip(*batch))
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    return imgs1, imgs2
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.data = glob(os.path.join(path, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.transform(cv2.imread(self.data[idx]))
        return data
    

def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def train(encoder, prev_dim, path, batch, epochs, device='cuda'):
    simsiam = SimSiam(encoder, prev_dim).to(device)

    init_lr = 0.05 * batch / 256

    criterion = nn.CosineSimilarity(dim=1)

    optimizer = torch.optim.SGD(simsiam.parameters(), init_lr, momentum=0.9, weight_decay=1e-4)

    dataset = Dataset(path, TwoCropsTransform())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    for e in range(epochs):
        adjust_learning_rate(optimizer, init_lr, e, epochs)
        desc = 'epoch: {: >4} / {:>4}'.format(e + 1, epochs)

        with tqdm(total=len(dataloader)) as pbar:
            losses = 0
            for i, img in enumerate(dataloader):
                img[0], img[1] = img[0].to(device), img[1].to(device)

                # write_jpeg(input=(img[0][0] * 255).type(torch.uint8), filename='a.jpg')
                # write_jpeg(input=(img[1][0] * 255).type(torch.uint8), filename='b.jpg')
                # break
                
                p1, p2, z1, z2 = simsiam(img[0], img[1])
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses = (i * losses + loss.item()) / (1 + i)
                pbar.set_description(desc + '  loss: {:.3f}'.format(loss))
                pbar.update()

        torch.save(encoder.cpu().state_dict(), 'encoder{}.pt'.format(e))
        encoder.to(device)


# backbone = YOLO('yolo11n.yaml').model.model[:11]
# train(backbone, 256, r"C:\Users\tanaka\dataset\coco\train2017\train2017", 16, 100)



model = YOLO('yolo11n.yaml', learned_section='backbone39.pt')
# model.train(data=r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp0\data.yaml", batch=8, epochs=100, patience=500, workers=0, name='simsiam_yolo11n')
model.train(data=r"C:\Users\tanaka\dataset\cross_validation\detection\3bacteria\temp0\data.yaml", freeze=11, batch=8, epochs=100, patience=500, workers=0, name='freeze_simsiam_yolo11n')