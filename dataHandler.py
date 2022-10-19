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

from matplotlib.pyplot import title


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


################################
# Description: 
#   Augments data by augmenting "multiplier" times more samples from original data
# Parameters:
#   album: tensor data
#   multiplier: the number of samples in the new album will be album.shape[0]*multiplier
#   keepOriginal: specifies whether or not to keep the original data
#   min_percent_crop: minimum percentage of the image to keep when cropping (if =0.75, then 75% of original data is kept)
################################
def AugmentData(album, multiplier=10, keepOriginal=True, min_percent_crop=0.75):
    # using OpenCV
    # Flip, Random Crop, RGB Scaling
    
    # create empty tensor of 10x more samples w/ same dimensions
    album_new = torch.zeros([album.shape[0]*multiplier] + [i for i in album.shape[1:4]])
    
    # if specified to keep the original data
    if keepOriginal:
        album_new[0:album.shape[0]] = album[np.random.permutation(album.shape[0])]
        count = album.shape[0]
    else:
        count = 0
        
    # until sample is filled in
    while count < album_new.shape[0]:
        
        # create random permutation of all the data indices
        indices = np.random.permutation(album.shape[0])
    
        # for each image in the above permutation
        for ind in indices:
            # get image and augment image
            image = album[ind]
            image = AugmentImage(image, min_percent_crop=min_percent_crop)
            
            # store augmented image in tensor
            album_new[count] = image

            # increase count
            count = count + 1
            
    return album_new
            

################################
# Description: 
#   Augments an RGB image by any combination of horizontal flipping, cropping, and RGB scaling
# Parameters:
#   image: image as tensor array
#   min_percent_crop: minimum percentage of the image to keep when cropping (if =0.75, then 75% of original data is kept)
################################
def AugmentImage(image, min_percent_crop=0.75):
    
    # number of augmentation strategies
    numAugStrat = 3
    
    # get a random permutation of 0 to 2 (index for augmentation technique)
    augTechs = np.random.permutation(numAugStrat)

    # random number of augmentations to use
    numAugTechs = np.random.randint(1, numAugStrat+1)

    # for each sample, perform random augmentation techniques
    augTechs = augTechs[0:numAugTechs]
    for augNum in augTechs:
        # horizontally flip image
        if augNum == 0:
            image = torch.flip(image, dims=(2,))

        # crop and resize image
        if augNum == 1:
            crop_dim = np.random.randint(int(min_percent_crop*image.shape[1]), image.shape[1])
            image = RandomCrop(image, crop_dim, crop_dim)

        # rgb scaling for image
        if augNum == 2:
            scale = np.random.uniform(low=0.6, high=1)
            image = image * scale
                    
    return image
                    
    
    
# First 7 lines from RandomCrop from: https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image 
################################
# Description: 
#   Crops an image and resizes to its original size
# Parameters:
#   image: image data as numpy array
#   crop_height: height of the cropped image
#   crop_width: width of the cropped image
################################
def RandomCrop(image, crop_height, crop_width):
    # get max lower bound of crop
    max_x = image.shape[2] - crop_width
    max_y = image.shape[1] - crop_height

    # get lower bound of crop
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    
    # crop each image channel
    crop = np.zeros([image.shape[0], crop_height, crop_width])
    cropResize = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
    for i in range(0, image.shape[0]):
        
        # crop by selected desired pixels
        crop[i] = image[i][y: y + crop_height, x: x + crop_width]
        
        # resize image to original size
        cropResize[i] = cv2.resize(crop[i], (image.shape[-2], image.shape[-1]))
        
    # convert resized cropped image back to tensor
    cropResize = torch.Tensor(cropResize)
        

    return cropResize


################################
# Description: 
#   Converts a tensor album of images from rgb to lab
# Parameters:
#   album: album of tensor images
################################
def AlbumToLAB(album):
    # convert each RGB image to LAB and store in new album as result
    new_album = torch.zeros([i for i in album.shape])
    for i, image in enumerate(album):
        new_album[i] = ImageToLAB(image)

    return new_album


def ImageToLAB(image):
    # permute tensor and convert to numpy to work properly with cv2.cvtColor
    np_image = image.permute(1, 2, 0).numpy()

    # convert image from rgb to lab
    new_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB)

    # conver new image back to tensor array
    # return new_image    # return numpy array
    return torch.Tensor(torch.Tensor(new_image).permute(1, 0, 2).T) # returns tensor




################################
# Description: 
#   Displays each component of image in LAB form
# Parameters:
#   image: RGB image
################################
def DisplayImageLAB(image):
    imageLAB = ImageToLAB(image)
    imageLAB = imageLAB.permute(1, 2, 0).numpy()
    norm_imageLAB = np.zeros_like(imageLAB)
    norm_imageLAB = cv2.normalize(imageLAB,norm_imageLAB,0,1,cv2.NORM_MINMAX)
    L,a,b=cv2.split(norm_imageLAB)

    cv2.imshow('image', norm_imageLAB, )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("LChannel", L) #album_faces[0].permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('aChannel', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('bChannel', b)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


################################
# Description: 
#   Displays the original image passed in and the image resulting from augmentation
# Parameters:
#   image: RGB image
################################
def DisplayAugmentedImage(image):
    aug_face_image = AugmentImage(image, min_percent_crop=0.75)
    cv2.imshow('Original', image.permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Augmented', aug_face_image.permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


        
    
face_path = path + slash + 'face_images'
originals_path = path + slash + 'ColorfulOriginal'
grayscale_path = path + slash + 'gray'

# for name in glob.glob(face_path + slash + '*', recursive = True):
#     print(name)    


album_faces = load(face_path + slash +'*.jpg')
# album_colors = load(originals_path + slash + '**' + slash +'*.jpg')
# album_gray = load(grayscale_path + slash + '**' + slash +'*.jpg')

#note album_faces is already size 128x128 does not need to be resized
album_faces = convert(album_faces)

# album_colors = resize(album_colors)
# album_colors = convert(album_colors)

# album_gray =resize(album_gray)
# album_gray = convert(album_gray)



# plot Original and Augmented Image
# DisplayAugmentedImage(album_faces[1])

# plot LAB version of image
DisplayImageLAB(album_faces[0])

# # augment and shuffle data
# album_faces_aug = shuffle(AugmentData(album_faces, multiplier=10, keepOriginal=True, min_percent_crop=0.75))

# # show arbitrary example image from augmented and shuffled dataset
# plt.imshow(album_faces_aug[125].permute(-2, -1, -3))
