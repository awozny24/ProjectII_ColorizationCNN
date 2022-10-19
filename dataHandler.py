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


torch.set_default_tensor_type(torch.FloatTensor)


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
    for index, photo in enumerate(album):
        #have to get the photo from numpy array to pillow object
        transform_pil = T.ToPILImage()
        photo = transform_pil(photo)
    
        transform_size = T.Resize((128,128))
        img = transform_size(photo)
        album[index] = img
          #album[index] = image
    return album
    #album = np.stack(album,axis = 3)
    

        

def convert(album):
    #convert everything in the lists to tensors
    #then stack list of tensors
    #to get 4d tensor
     for index, img in enumerate(album):
         make_tensor = T.ToTensor()
         img = make_tensor(img)
         album[index] = img
         if(img) == None:
             print('there is none object')
    #should convert list of tensors to 4d tensor
     album = torch.stack(album)
     
     return album  
     

    

        
def shuffle(album):
    indices = torch.randperm(album.shape[0])
    #randperm generates valid random indices
    #view returns a new tensor with the same data except the data has
    #been rearranged
    shuffled_album = album[indices].view(album.size())
    return shuffled_album

def augment(album):
    #want to generate 10 times the number 
    #of samples compared to original
    #using crop rotations and flips and scalings of RGB values from [0.6,1.0]
    #should be possible for all three operations to be applied to
    
    
    pass

        
    
face_path = path + slash + 'face_images'
originals_path = path + slash + 'ColorfulOriginal'
grayscale_path = path + slash + 'gray'

# for name in glob.glob(face_path + slash + '*', recursive = True):
#     print(name)    


album_faces = load(face_path + slash +'*.jpg')
album_colors = load(originals_path + slash + '**' + slash +'*.jpg')
album_gray = load(grayscale_path + slash + '**' + slash +'*.jpg')

#note album_faces is already size 128x128 does not need to be resized
album_faces = convert(album_faces)

album_colors = resize(album_colors)
album_colors = convert(album_colors)

album_gray =resize(album_gray)
album_gray = convert(album_gray)



