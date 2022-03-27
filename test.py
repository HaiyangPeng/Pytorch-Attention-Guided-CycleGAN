import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from models import Gen, Dis, Attn
from losses import realTargetLoss, fakeTargetLoss, cycleLoss

from torchutils import toZeroThreshold, weights_init, Plotter, save_checkpoint
import itertools
from PIL import Image

import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='datasets/apple2orange/', help='root of the images')
parser.add_argument('--resume', type=str, default='None', help='file to resume')
parser.add_argument('--saveroot', type=str, default='datasets/apple2orange/', help='root of saving images')

opt = parser.parse_args()

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

# Generators and Discriminators
genA2B = Gen() 
genB2A = Gen()
disA = Dis()
disB = Dis()
# Attention Modules
AttnA = Attn()
AttnB = Attn()

if cudaAvailable:
    genA2B.cuda().eval()
    genB2A.cuda().eval()

    disA.cuda().eval()
    disB.cuda().eval()

    AttnA.cuda().eval()
    AttnB.cuda().eval()

dataroot = opt.dataroot
batchSize = 1
n_cpu = 4
size = 256

transforms_ = [ 
                transforms.Resize(int(size), Image.BICUBIC), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True, mode='test'), 
    batch_size=batchSize, shuffle=False, num_workers=n_cpu)


if opt.resume is not 'None':
    checkpoint = torch.load(opt.resume)
    
    genA2B.load_state_dict(checkpoint['genA2B'])
    genB2A.load_state_dict(checkpoint['genB2A'])
    disA.load_state_dict(checkpoint['disA'])
    disB.load_state_dict(checkpoint['disB'])
    AttnA.load_state_dict(checkpoint['AttnA'])
    AttnB.load_state_dict(checkpoint['AttnB'])

    del(checkpoint)
    
with torch.no_grad():
    
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print(i)
        realA, realB = batch['A'].type(Tensor), batch['B'].type(Tensor)
        
        # A --> A'' 
        attnMapA = toZeroThreshold(AttnA(realA))
        fgA = attnMapA * realA
        bgA = (1 - attnMapA) * realA
        genB = genA2B(fgA) 
        fakeB = (attnMapA * genB) + bgA
        fakeBcopy = fakeB.clone()
        attnMapfakeB = toZeroThreshold(AttnB(fakeB))
        fgfakeB = attnMapfakeB * fakeB
        bgfakeB = (1 - attnMapfakeB) * fakeB
        genA_ = genB2A(fgfakeB)
        A_ = (attnMapfakeB * genA_) + bgfakeB
        
        realA_numpy = ((realA + 1.0) / 2.0 * 255)[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].astype('uint8')
        fakeB_numpy = ((fakeB + 1.0) / 2.0 * 255)[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].astype('uint8')     
        realA_A_ = np.concatenate([realA_numpy, fakeB_numpy], 1)
        cv2.imwrite(opt.saveroot + str(i) + 'fakeB.png', realA_A_)
        
        # B --> B''
        attnMapB = toZeroThreshold(AttnB(realB))
        fgB = attnMapB * realB
        bgB = (1 - attnMapB) * realB
        genA = genB2A(fgB) 
        fakeA = (attnMapB * genA) + bgB
        fakeAcopy = fakeA.clone()
        attnMapfakeA = toZeroThreshold(AttnA(fakeA))
        fgfakeA = attnMapfakeA * fakeA
        bgfakeA = (1 - attnMapfakeA) * fakeA
        genB_ = genA2B(fgfakeA)
        B_ = (attnMapfakeA * genB_) + bgfakeA
        
        realB_numpy = ((realB + 1.0) / 2.0 * 255)[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].astype('uint8')
        fakeA_numpy = ((fakeA+ 1.0) / 2.0 * 255)[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].astype('uint8')     
        realB_B_ = np.concatenate([realB_numpy, fakeA_numpy], 1)
        cv2.imwrite(opt.saveroot + str(i) + 'fakeA.png', realB_B_)
        
        

        

        