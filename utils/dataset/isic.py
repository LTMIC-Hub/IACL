#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from .randaugment import rand_augment_transform
import random
from PIL import Image, ImageFilter




IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# IMAGENET_MEAN = [0.466, 0.471, 0.380]
# IMAGENET_STD  = [0.195, 0.194, 0.192]


augmentation_rand = transforms.Compose(
                                    [transforms.Resize((224,224)),   
                                    transforms.RandomHorizontalFlip(p=0.5), 
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                    ])

augmentation_sim = transforms.Compose(
    [transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
        ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    )
rgb_mean = (0.485, 0.456, 0.406)
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
augmentation_randncls = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

augmentation_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

