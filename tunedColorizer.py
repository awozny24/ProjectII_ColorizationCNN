import cv2
import os
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import glob
# For our model
import matplotlib.pyplot as plt
import numpy as np

# For utilities
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from sys import platform

class colorizer(nn.Module):
    def __init__(self):
        super(colorizer, self).__init__()
        
        # number of filters at each layer (chosen from finetuning)
        outSize = [128, 256, 256, 200, 128, 64, 32, 16, 4] 

        # other attempted filter combinations
          # outSize = [2, 4, 8, 16, 32, 16, 8, 4, 2] 
          # outSize = [4, 16, 32, 64, 128, 200, 256, 256, 128]
          # outSize = [8, 16, 32, 128, 256, 128, 32, 16, 8]
          # outSize = [4, 8, 32, 128, 256, 256, 128, 32, 4]
          # outSize = [2, 4, 8, 8, 8, 8, 8, 4, 2]
          # outSize = [4, 32, 64, 64, 64, 64, 64, 32, 4]
          # outSize = [80, 70, 60, 50, 40, 24, 16, 8, 4]
          # outSize = [2, 4, 8, 16, 32, 16, 8, 4, 2]
    
        #128x128
        self.downsamp1 = nn.Sequential(
             nn.Conv2d(1, outSize[0], kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(outSize[0]),
             )
        #64x64
        self.downsamp2 = nn.Sequential(
             nn.Conv2d(outSize[0], outSize[1], kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(outSize[1]),
             )
        #32x32
        self.downsamp3 = nn.Sequential(
             nn.Conv2d(outSize[1], outSize[2], kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(outSize[2]),
             )
        #16x16
        self.downsamp4 = nn.Sequential(
             nn.Conv2d(outSize[2], outSize[3], kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(outSize[3]),
             )
        #8x8
        self.downsamp5 = nn.Sequential(
             nn.Conv2d(outSize[3], outSize[4], kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(outSize[4]),
             )
        
        #begin upsampling here
        #using convolution transpose
        #8x8
        self.upsamp1 = nn.Sequential(        
             nn.ConvTranspose2d(outSize[4], outSize[5], kernel_size = 2, stride = 2),
             nn.BatchNorm2d(outSize[5]),
             )
        
        #16x16
        self.upsamp2 = nn.Sequential(         
             nn.ConvTranspose2d(outSize[5], outSize[6], kernel_size = 2, stride = 2),
             nn.BatchNorm2d(outSize[6]),
             )
        
        #32x32
        self.upsamp3 = nn.Sequential(
            nn.ConvTranspose2d(outSize[6], outSize[7], kernel_size = 2, stride = 2),    
            nn.BatchNorm2d(outSize[7]),
            )
        
        #64x64
        self.upsamp4 = nn.Sequential(
            nn.ConvTranspose2d(outSize[7], outSize[8], kernel_size = 2, stride = 2),    
            nn.BatchNorm2d(outSize[8]),
            )
        
         #128x128
         #keep at 2 channels
        self.upsamp5 = nn.Sequential(
            nn.ConvTranspose2d(outSize[8], 2, kernel_size = 2, stride = 2),         
            )
           

    def forward(self, x):

        x = x.float()
        out = self.downsamp1(x)
        out = self.downsamp2(out)
        out = self.downsamp3(out)
        out = self.downsamp4(out)
        out = self.downsamp5(out)
        out = self.upsamp1(out)
        out = self.upsamp2(out)
        out = self.upsamp3(out)
        out = self.upsamp4(out)
        out = self.upsamp5(out)

        #collapse dimension 1 along dimension 0 using flatten
        out = torch.flatten(out, 0, 1)
        return out
    
    