# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:13:04 2022

@author: LuTimothy
"""

import cv2
import os
import glob
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from sys import platform

import torch
from  torchvision.utils import save_image
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
    

        

def convert_tensor(album):
    #convert everything in the lists to tensors
    #then stack list of tensors
    #to get 4d tensor
     for index, img in enumerate(album):
        
         img = torch.from_numpy(img)
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



def AugmentData(album, multiplier=3, keepOriginal=True):
    #main difference here is that I'm only working with numpy arrays in the list
    #haven't converted to tensor
    
    #generate a list that is 10 times the size of the album in question
    augmented_alb = [[] for _ in range(10 * len(album))]
    #borrowed code
    if keepOriginal:
        augmented_alb[0:len(album)] = np.array(album)[np.random.permutation(len(album))]
        start_index = len(album)
    else:
        start_index = 0    
        #start at starting index
    for index, entry in enumerate(augmented_alb[start_index:]):
        
        chosen_image = random.choice(album)
        
        k = random.randint(1,3)
        selected_transforms = random.sample(range(3),k)
        
        #use function dictionary to decide which transforms to perform
        transforms_table = {0:horiz_flip, 1:scale, 2:crop }
        
        for transform in selected_transforms:
            chosen_image = transforms_table[transform](chosen_image)
            
            #torch.tensor(chosen_image)
            
            #store results
        entry = chosen_image
        augmented_alb[start_index + index] = entry
    
    return augmented_alb
   
    
def horiz_flip(image):
   
        
   # print('in flip')
    transform_flip = T.RandomHorizontalFlip(p=1) 

  
    image = transform_flip(image)
  
    
    return image
    
    

def scale(image):
    make_tensor = T.ToTensor()
    
    scale = np.random.uniform(low=0.6, high=1.0)
    image = torch.mul(image,torch.tensor(scale))
    
    
    return image
    

def crop(image):
   # print('in crop')
    
    
    
    #dividing by 4/3 is the same as shrinking the original image to 75%
    crop_factor_width = random.uniform(1,4/3)
    crop_factor_height = random.uniform(1,4/3)
    
    
    transform_crop = T.RandomCrop(size = (int(128//crop_factor_width), int(128//crop_factor_height)))
    make_tensor = T.ToTensor()  
    transform_pil = T.ToPILImage()
    transform_size = T.Resize((128,128))
    
    
       #converting from tensor to numpy array and back again
       #because T.ToPILImage accepts tensors in the sequence Channels x Height x Width
       #whereas it accepts Ndarrays as Height x Width x Channels
       
    image = np.asarray(image).astype('uint8')
    image = transform_pil(image)
    image = transform_crop(image)
    image = transform_size(image)
    #convert back to tensor once done
    image = np.array(image)
    image = torch.from_numpy(image)

    
    
    return image
    
    
    
 
 
    


################################
#Descriptor:
#Saves the images in augmented data into a new folder
#Params:
#augmented data
################################

def saveAugmented(album, folder_name):
    subfolder_dir = os.path.join(path, folder_name)
    
    if not os.path.exists(subfolder_dir):
        os.mkdir(subfolder_dir)
        
    
   
    count = 0
    #doing a tensor to numpy conversion
    for image in album:
       # image = image.cpu().detach().numpy()
       image = np.asarray(image)
       
       cv2.imwrite(subfolder_dir + slash + str(count) + '.jpg', image)
       count +=1
       #save_image(image,subfolder_dir + str(count), format = '.jpg')
    
    
    

        
    
face_path = path + slash + 'face_images'
originals_path = path + slash + 'ColorfulOriginal'
grayscale_path = path + slash + 'gray'

# for name in glob.glob(face_path + slash + '*', recursive = True):
#     print(name)    


album_faces = load(face_path + slash +'*.jpg')
# album_colors = load(originals_path + slash + '**' + slash +'*.jpg')
# album_gray = load(grayscale_path + slash + '**' + slash +'*.jpg')

#note album_faces is already size 128x128 does not need to be resized
album_faces = convert_tensor(album_faces)

# album_colors = resize(album_colors)
# album_colors = convert(album_colors)

# album_gray =resize(album_gray)
# album_gray = convert(album_gray)

augmented_faces = AugmentData(album_faces)
saveAugmented(augmented_faces, 'augmented_faces')


# imageLAB = ImageToLAB(album_faces[0])
# L,a,b=cv2.split(imageLAB)
# cv2.imshow("LChannel", L) #album_faces[0].permute(1, 2, 0).numpy())
# print("Hit any key to continue:")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('aChannel', a)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('bChannel', b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plot example Original Image
# plt.imshow(album_faces[0].permute(-2, -1, -3))
# plt.title("Original Image")

# # plot example Augmented Image
# plt.imshow(AugmentImage(album_faces[0], min_percent_crop=0.75).permute(-2, -1, -3))
# plt.title("Augmented Image")

# # augment and shuffle data
# album_faces_aug = ShuffleData(AugmentData(album_faces, multiplier=10, keepOriginal=True, min_percent_crop=0.75))

# # show arbitrary example image from augmented and shuffled dataset
# plt.imshow(album_faces_aug[125].permute(-2, -1, -3))
