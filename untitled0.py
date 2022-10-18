# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:13:04 2022

@author: LuTimothy
"""

import cv2
import os
import glob
import numpy as np
import torch
from sys import platform

import torch
import torchvision.transforms as T
from PIL import Image



path = os.getcwd() 

if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'
    
def load(folder):
    files = glob.glob(folder)
    data =[]
    for f in files:
        image = cv2.imread(f)
        data.append(image)
    return data
    #need to change height and width to 128
    #also need to get randindexes?

def resize(album):
    for img in album:
        transform = T.Resize(size = (128,128))
        img = transform(img)
    


        
def shuffle(album):
    indices = torch.randperm(album.shape[1])
        
    
face_path = path + slash + 'face_images'
originals_path = path + slash + 'ColorfulOriginal'
grayscale_path = path + slash + 'gray'

# for name in glob.glob(face_path + slash + '*', recursive = True):
#     print(name)    


album_faces = load(face_path + slash +'*.jpg')
album_colors = load(originals_path + slash + '**' + slash +'*.jpg')
album_gray = load(grayscale_path + slash + '**' + slash +'*.jpg')
    
print('done!')
#first make albums into arrays instead of list of arrays


#now convert to tensor
album_faces = torch.from_numpy(album_faces).type(torch.FloatTensor)
album_colors = torch.from_numpy(album_colors).type(torch.FloatTensor)
album_gray = torch.from_numpy(album_gray).type(torch.FloatTensor)


#NOTE none of the faces are all aready size 128x128

