#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck

class CMNet(nn.Module):
    def __init__(self, ):
        super(CMNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.avepool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avepool(x)
      
   
        temp_cov = torch.transpose(x,2,3)  
        x = torch.matmul(x, temp_cov).div(torch.sqrt(torch.tensor(128)).cuda())
        x = torch.tanh(x)
        
        temp_cov = torch.transpose(x,2,3)  
        x = torch.matmul(x, temp_cov)
        return x

class IACL_model(nn.Module):
    def __init__(self,num_classes=8,feat_dim=512):
        super(IACL_model, self).__init__()
        cnns = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*(list(cnns.children())[:-1]))
        self.num_classes = num_classes
        dimension = 512*4
        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.fc = nn.Linear(dimension,self.num_classes)


    def forward(self,x):
        if isinstance(x, list):
            
            feat1 = self.backbone(x[0])
            feat1 = feat1.view(feat1.shape[0],-1)
            feat1_ = self.head(feat1)
            feat1_mlp = F.normalize(feat1_)
            logits = self.fc(feat1)

            feat2 = self.backbone(x[1])
            feat2 = feat2.view(feat2.shape[0], -1)
            feat2_ = self.head(feat2)
            feat2_mlp = F.normalize(feat2_)
            return logits, [feat1_, feat2_], [feat1_mlp, feat2_mlp], feat1
        else:
            feat1 = self.backbone(x)
            feat1 = feat1.view(feat1.shape[0], -1)
            logits = self.fc(feat1)
            return logits, feat1














