# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:04:59 2022

@author: Timothy Lu
"""

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
from sklearn.utils import shuffle
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

''''
****SET IMPORTANT HYPERPARAMETERS HERE***

'''
batch_size = 32
Epochs = 100
lr = 0.01
criterion = nn.MSELoss()
torch.set_default_tensor_type(torch.FloatTensor)


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
  
def make_numpy_format(image):
    image = torch.swapaxes(image, 0, 1)
    image = torch.swapaxes(image, 1, 2)
    return image

def make_tensor(image):
    image = np.swapaxes(image, 2,1)
    image = np.swapaxes(image, 1,0)
    return image


class imageDataset(Dataset):
    def __init__(self,  l_color_space, ab_color_space,):
        #dropping redundant channels
        a = (ab_color_space[:, 0, :, :, 0])
        b = (ab_color_space[:, 1, :, :, 0])
        l = (l_color_space[:,:,:,0])
        

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
             nn.Conv2d(1, 2, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(2),
            # nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #64x64
        self.downsamp2 = nn.Sequential(
             nn.Conv2d(2, 4, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(4),
           #  nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #32x32
        self.downsamp3 = nn.Sequential(
             nn.Conv2d(4, 8, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(8),
          #   nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #16x16
        self.downsamp4 = nn.Sequential(
             nn.Conv2d(8, 16, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(16),
          #   nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        #8x8
        self.downsamp5 = nn.Sequential(
             nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
             nn.ReLU(),
             nn.BatchNorm2d(32),
          #   nn.AvgPool2d(kernel_size = (1,1), stride = 1)
             )
        
        #begin upsampling here
        #using convolution transpose
        #8x8
        self.upsamp1 = nn.Sequential(        
             nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2),
             nn.BatchNorm2d(16),
             )
        
        #16x16
        self.upsamp2 = nn.Sequential(         
             nn.ConvTranspose2d(16, 8, kernel_size = 2, stride = 2),
             nn.BatchNorm2d(8),
             )
        
        #32x32
        self.upsamp3 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size = 2, stride = 2),    
            nn.BatchNorm2d(4),
            )
        
        #64x64
        self.upsamp4 = nn.Sequential(
            nn.ConvTranspose2d(4, 2, kernel_size = 2, stride = 2),    
            nn.BatchNorm2d(2),
            )
        
         #128x128
         #keep at 2 channels
        self.upsamp5 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, kernel_size = 2, stride = 2),         
            )
           
       
        
    def forward(self, x):
        #Normalize input first
       # make_Tensor = T.ToTensor()
     #   transform_pil = T.ToPILImage()
        
        #out = make_Tensor(x)
        #out = transform_pil(x)
        #want to change this to batch normalization eventually
      #  out = normalize(out)
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
       #out = nn.linear(2, size)
       # out = out.detach().numpy()
       # out = make_numpy(out)
       # out = np.uint8(out)
      
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
train_images, val_images = train_test_split(train_images, test_size = 0.1)

#further separate them into X's and Y's where L is the input and AB are the targets (LAB colorspace)
#remember the dimensions are Number of grouped images X Index of image
#this needs to be flipped

X_train = train_images[:, 2, :, :, :]
y_train = train_images[:, 0:2, :, :, :]

X_test = test_images[:, 2, :, :, :]
y_test = test_images[:, 0:2, :, :, :]

X_val = test_images[:, 2, :, :, :]
y_val = test_images[:, 0:2, :, :, :]

#prepare datasets for images
train_dataset = imageDataset(X_train, y_train)
test_dataset = imageDataset(X_test, y_test)
val_dataset = imageDataset(X_val, y_val)

#prepare dataloaders for batch training
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,  batch_size = batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,  batch_size = batch_size, shuffle=True)

# select GPU / CPU -hmk
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

#call the regressor on one set of images in the X_train dataset
color = colorizer().to(device)


#run color regressor
optimizer = torch.optim.Adam(color.parameters(), lr)



#training loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#loss_values = []
train_loss = []
validation_loss = []
val_ticker = 0
last_loss = 20000

# rows, cols = (2, Epochs)
# stored_images = [[0 for i in range(cols)] for j in range(rows)]

for epoch in range(Epochs):  # loop over the dataset multiple times
    color.train()
   
    running_loss = 0.0
    #I want batch to be of length 10 not 3 why? self.a[index], self.b[index], self.l[index]
    for i, img in enumerate(train_loader):
        
        a = img[0]
        b = img[1]
        l = img[2]
    
        
        #each batch is ten images so loop through all the images per batch
        # no!!!! this defeats the point of batches if you loop through each image you've essentially made your batch size 1 -hmk
        
        # for index, images in enumerate(batch):
            # get the inputs; data is a list of tensors [chrominance_a_tensor, chrominance_b_tensor, grayscale_l_tensor]
            # different images!
       
     
        #labels = torch.tensor((label_a, label_b))
        #might not be necessary to drop duplicates
        labels = torch.stack((a, b), 1).float().to(device)
        input_l = torch.unsqueeze(l, 1).to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = color((input_l))
        # outputs = outputs.view(2, size)
        
        #flatten labels along dimension 0
        labels = torch.flatten(labels, 0, 1)
        
        
        loss = criterion(outputs, labels)
    
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
                
    train_loss.append(loss)

    if epoch % 10 == 0:
        running_val_loss = 0.0
        with torch.no_grad():
            color.eval()
            for data in val_loader:
                val_l = torch.unsqueeze(data[2], 1).to(device)
                val_outputs = color(val_l)
                val_labels = torch.stack((data[0], data[1]), 1).float().to(device)
                val_loss = criterion(val_outputs, torch.flatten(val_labels, 0, 1))
                running_val_loss += val_loss

        validation_loss.append(running_val_loss)
        print("Number Of Images Tested =", len(val_loader))
        print("\nValidation Loss =", (running_val_loss/len(val_loader)))

        if (running_val_loss/len(val_loader)) - last_loss >= 0.1:
            path = f"./chkpt/color_model_{epoch}.pt"
            torch.save(color.state_dict(), path)
        last_loss = (running_val_loss/len(val_loader))

        # once done with a loop I want to print out the target image 
        # # and colorized image for comparison    
        # sample_target = cv2.merge([l[0].detach().numpy(), a[0].detach().numpy(), b[0].detach().numpy()]) 
        # sample_target = cv2.cvtColor(sample_target, cv2.COLOR_LAB2RGB)
        #plt.imshow(sample_target)
        
        sample_target = cv2.merge([l[0].cpu().detach().numpy(), a[0].cpu().detach().numpy(), b[0].cpu().detach().numpy()]) 
        sample_target = cv2.cvtColor(sample_target, cv2.COLOR_LAB2RGB)
        #plt.imshow(sample_target)
    
        colorized_a = outputs[0].cpu().detach().numpy().astype(np.uint8)
        colorized_b = outputs[1].cpu().detach().numpy().astype(np.uint8)
        sample_colorized = cv2.merge([l[0].detach().numpy(), colorized_a, colorized_b])
        sample_colorized = cv2.cvtColor(sample_colorized, cv2.COLOR_LAB2RGB)
        #plt.imshow(sample_colorized)
        # stored_images[0][epoch] = sample_target
        # stored_images[1][epoch] = sample_colorized
        cv2.imwrite(f"./chkpt/images/target_image_{epoch}.png",sample_target)
        cv2.imwrite(f"./chkpt/images/output_image_{epoch}.png",sample_colorized) # -hmk

    print('Epoch {} of {}, Train Loss: {:.3f}'.format( epoch+1, Epochs, running_loss/len(train_loader)))


print('Finished Training')
train_loss = [epoch.cpu().detach().numpy() for epoch in train_loss] # changed var from val bc this has nothing to do with validation
validation_loss = [val.cpu().detach().numpy() for val in validation_loss]
plt.figure()
plt.plot(np.arange(0,Epochs,1), train_loss, 'r', label='Training Loss')
plt.plot(np.arange(0,Epochs,10), validation_loss,'b', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc="upper right")
plt.savefig(f"./chkpt/training-val-plot.png")
plt.show()

# testing time!!
color.load_state_dict(torch.load(path))

running_test_loss = 0.0
with torch.no_grad():
    color.eval()
    for data in test_loader:
        test_l = torch.unsqueeze(data[2], 1).to(device)
        test_outputs = color(test_l)
        test_labels = torch.stack((data[0], data[1]), 1).float().to(device)
        test_loss = criterion(test_outputs, torch.flatten(test_labels, 0, 1))
        running_test_loss += test_loss
print("Number Of Images Tested =", len(test_loader))
print("\nTest Loss =", (running_test_loss/len(test_loader)))
