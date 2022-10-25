# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:53:02 2022

@author: LuTimothy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:24:56 2022

@author: Timothy Lu
"""
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
# For our model
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# For utilities
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from sys import platform
from itertools import combinations

# For color conversions
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb

import torchvision.transforms as T



#CNN resource: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
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

def group(data, album_length):
    #group into chunks of three because of three sets of images in LAB color space
    for i in range (0, album_length, 3):
        yield image_data[i:i+3]
  
def make_numpy(image):
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    return image


class imageDataset(Dataset):
    def __init__(self,  l_color_space, ab_color_space,):
        a = (ab_color_space[:, 0, :, :, :])
        b = (ab_color_space[:, 1, :, :, :])
        l = (l_color_space)
        
        #it seems that I will have to use permute 
        #to get from numpy image representation
        #to torch tensor image 
        
        self.a = a
        self.b = b
        self.l = l
        self.indices = len(l)
        
    def __len__(self):
        return self.indices
      
    
    def __getitem__(self, index):
        return self.a[index], self.b[index], self.l[index]
      
class colorizer(nn.Module):
    def __init__(self):
        super(colorizer, self).__init__()
        
        #128x128
        self.downsamp1 = nn.Sequential(
             nn.Conv2d(3, 6, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #64x64
        self.downsamp2 = nn.Sequential(
             nn.Conv2d(6, 12, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #32x32
        self.downsamp3 = nn.Sequential(
             nn.Conv2d(12, 24, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #16x16
        self.downsamp4 = nn.Sequential(
             nn.Conv2d(24, 48, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #8x8
        self.downsamp5 = nn.Sequential(
             nn.Conv2d(48, 96, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        
        #begin upsampling here
        #using convolution transpose
        #8x8
        self.upsamp1 = nn.Sequential(        
             nn.ConvTranspose2d(96, 48, kernel_size = 2, stride = 2),
             )
        
        #16x16
        self.upsamp2 = nn.Sequential(         
             nn.ConvTranspose2d(48, 24, kernel_size = 2, stride = 2),         
             )
        
        #32x32
        self.upsamp3 = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size = 2, stride = 2),         
            )
        
        #64x64
        self.upsamp4 = nn.Sequential(
            nn.ConvTranspose2d(12, 6, kernel_size = 2, stride = 2),         
            )
        
         #128x128
         #keep at 6 channels
        self.upsamp5 = nn.Sequential(
            nn.ConvTranspose2d(6, 6, kernel_size = 2, stride = 2),         
            )
           
       
        
    def forward(self, x):
        #Normalize input first
        make_Tensor = T.ToTensor()
     #   transform_pil = T.ToPILImage()
       
        out = make_Tensor(x)
        #out = transform_pil(x)
        #want to change this to batch normalization eventually
        out = normalize(out)
        # out = self.mod1(out)
        # out = self.mod2(out)
        # out = self.mod3(out)
        # out = self.mod4(out)
        # out = self.mod5(out)
        # out = self.mod6(out)
        # out = self.mod7(out)
        # out = torch.mean(out,0, True)
        out = self.downsamp1(out)
        out = self.downsamp2(out)
        out = self.downsamp3(out)
        out = self.downsamp4(out)
        out = self.downsamp5(out)
        out = self.upsamp1(out)
        out = self.upsamp2(out)
        out = self.upsamp3(out)
        out = self.upsamp4(out)
        out = self.upsamp5(out)
        
        return out



#set paths 
#resource:
#https://www.kaggle.com/code/anirbanmalick/image-colorization-pytorch-conv-autoencoder

home_dir = os.getcwd() 
#change this parameter depending on which album you want
target_album = 'LAB_TEST_FACES'



image_data = load(home_dir + slash + target_album + slash + '*.jpg')
album_length = len(image_data)

#group images into sets of 3   
grouped_data = list(group(image_data, album_length))
grouped_data = np.asarray(grouped_data)
    
#prepare grouped data for training and test
train_images, test_images = train_test_split(grouped_data, test_size = 0.3)

#further separate them into X's and Y's where L is the input and AB are the targets (LAB colorspace)
#remember the dimensions are Number of grouped images X Index of image
#this needs to be flipped

X_train = train_images[:, 2, :, :, :]
y_train = train_images[:, 0:2, :, :, :]

X_test = test_images[:, 2, :, :, :]
y_test = test_images[:, 0:2, :, :, :]



#prepare datasets for images
train_dataset = imageDataset(X_train, y_train)
test_dataset = imageDataset(X_test, y_test)

#prepare dataloaders for batch training
#create datasets

batch_size = 10
Epochs = 10
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)



#call the regressor on one set of images in the X_train dataset
color = colorizer()
#run forward pass on one grayscale image
sample_a, sample_b, sample_grayscale = train_dataset[0]

#test out opencv's merge
#somehow the LAB images have three channels when they only should have one
#so I'm dropping the duplicate channels when merging
#change order
merged_lab = cv2.merge([sample_grayscale[:,:,0], sample_a[:,:,0], sample_b[:,:,0]])
#merged_lab = cv2.merge([sample_a[:,:,0], sample_b[:,:,0], sample_grayscale[:,:,0]])
#merged_lab = cv2.merge(merged_lab, sample_grayscale)
target = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)


prediction = color(sample_grayscale)
prediction_a = prediction[0:3]
prediction_b = prediction[3:]

#swapaxes and convert to numpy array
prediction_a = prediction_a.detach().numpy().astype(np.uint8)
prediction_b = prediction_b.detach().numpy().astype(np.uint8)

prediction_a = make_numpy(prediction_a)
prediction_b = make_numpy(prediction_b)
merged_prediction = cv2.merge([sample_grayscale[:,:,0], prediction_a[:,:,0], prediction_b[:,:,0]])

#plot the target
plt.imshow(target)
#plot the reconstruction
plt.imsmhow(merged_prediction)


#run color regressor
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(colorizer.parameters(), lr)



#training loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

train_loss = []
for epoch in range(Epochs):  # loop over the dataset multiple times
    colorizer.train()
   
    running_loss = 0.0
    #I want batch to be of length 10 not 3 why?
    for i, batch in enumerate(train_loader):
        
        batch_a = batch[0]
        batch_b = batch[1]
        batch_l = batch[2]
    
        
        #each batch is ten images so loop through all the images per batch
        
        for index, images in enumerate(batch):
            # get the inputs; data is a list of tensors [chrominance_a_tensor, chrominance_b_tensor, grayscale_l_tensor]
          #different images!
            label_a = normalize(batch_a[index].float())
            label_b = normalize(batch_b[index].float())
            input_l = normalize(batch_l[index].float())
            
            #need to get the mean of labels across all dimensions
            mean_a = torch.mean(label_a, dim = [0, 1, 2])
            mean_b = torch.mean(label_b, dim = [0, 1, 2])
            
            labels = torch.tensor((mean_a, mean_b))
            #add new axis to make it consistent with dimension of regressor output tensor
            labels = torch.unsqueeze(labels, 0)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = colorizer (np.asarray(input_l))
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
                  
        
    train_loss.append(loss)
    print('Epoch {} of {}, Train Loss: {:.3f}'.format( epoch+1, Epochs, loss))
          
      

print('Finished Training')

