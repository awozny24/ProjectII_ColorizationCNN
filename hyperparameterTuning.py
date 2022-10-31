# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:04:59 2022

@author: Timothy Lu
"""
# also hannah kirkland

import cv2
import os
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import glob
# For our model
import matplotlib.pyplot as plt
import numpy as np

from tunedColorizer import colorizer

# For utilities
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from sys import platform

''''
****SET IMPORTANT HYPERPARAMETERS HERE***

'''

torch.set_default_tensor_type(torch.FloatTensor)
path = os.getcwd() 

#CNN resource: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'
 
    

#set paths 
#resource:
#https://www.kaggle.com/code/anirbanmalick/image-colorization-pytorch-conv-autoencoder

def load(folder):
    files = glob.glob(folder)
    data =[]
    for f in files:
        image = cv2.imread(f)
        data.append(image)
    return data


def LoadLabInOrder(folder):

    # find the largest number file to index (do not worry about how these next few lines work)
    files = glob.glob(folder + "*L.jpg")
    for i, f in enumerate(files):
        f = f[f.rfind(slash)+1:]
        files[i] = f[0:f.rfind('.')-1]
    maxFileNum = max([int(f) for f in files])
    
    # for each file index (e.g. ['0a.jpg', '0b.jpg', '0L.jpg'])
    data = []
    for i in range(0, maxFileNum):
        # grab files in order 'a', 'b', 'L'
        files = sorted(glob.glob(folder + str(i) + "?.jpg"), key=str.casefold)

        # append each file
        for f in files:
            image = cv2.imread(f)

            # only take one channel (all the channels here are the same)
            data.append(image)
        
    return data


def group(data, album_length):
    #group into chunks of three because of three sets of images in LAB color space
    for i in range (0, album_length, 3):
        yield data[i:i+3]
  
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
    
    
def convert_LAB(album):
    converted_album = [[] for _ in range( len(album))]
    
    #code borrowed from https://towardsdatascience.com/computer-vision-101-working-with-color-images-in-python-7b57381a8a54
    for index, image in enumerate(album):
        #do LAB conversion here
        image = np.asarray(image).astype('uint8')
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        names = []
        channels = []
        
        for (name, chan) in zip(("L", "a", "b"), cv2.split(converted_image)):
            names.append(name)
            channels.append(chan)
           # cv2.imshow(name, chan)
           
        converted_album[index] = list(zip(names, channels))
              
    return converted_album
        
def saveLAB(album, folder_name):
    subfolder_dir = os.path.join(os.getcwd(), folder_name)
    
    if not os.path.exists(subfolder_dir):
        os.mkdir(subfolder_dir)

    count = 0
    #each entry inside an LAB color space album is a tuple
    for tup in album:
       # image = image.cpu().detach().numpy()
       names, channels = zip(*tup)
       
       for i in range(len(channels)):
           name = names[i]
           channel = channels[i]
           cv2.imwrite(subfolder_dir + slash + str(count) + name + '.jpg', channel)
           
      
       count +=1


Epochs = 100
lrVals = [0.001, 0.01, 0.1]
BatchSizes = [10, 20, 32, 50]

for batch_size in BatchSizes: # comment out this line when done with epochs
    for lr in lrVals: # comment out this line when done with learning rate

        print(f"\n\nBatchSize = {batch_size} and LR = {lr}")

        home_dir = os.getcwd() 
        #change this parameter depending on which album you want
        album = 'LAB_TEST_FACES'
        image_data = LoadLabInOrder(home_dir + slash + album + slash)
        album_length = len(image_data)

        #group images into sets of 3   
        grouped_data = list(group(image_data, album_length))
        grouped_data = np.asarray(grouped_data)
            
        #prepare grouped data for training and test
        train_images, test_images = train_test_split(grouped_data, test_size = 0.1)
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
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
        test_loader = DataLoader(dataset = test_dataset,  batch_size = batch_size, shuffle=True)
        val_loader = DataLoader(dataset = val_dataset,  batch_size = batch_size, shuffle=True)

        # select GPU / CPU -hmk
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Device:', device)

        #call the regressor on one set of images in the X_train dataset
        color = colorizer().to(device)



        #training loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        #loss_values = []
        train_loss = []
        validation_loss = []
        val_ticker = 0
        last_loss = 20000

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(color.parameters(), lr)
        rows, cols = (2, Epochs)
        stored_images = [[0 for i in range(cols)] for j in range(rows)]

        for epoch in range(Epochs):  # loop over the dataset multiple times
            color.train()
          
            running_loss = 0.0
            #I want batch to be of length 10 not 3 why?
            for i, img in enumerate(train_loader):
                
                a = img[0] # i changed these for clarity and less typing i didn't want to type batch everytime -hmk
                b = img[1]
                l = img[2]
            
                
                #each batch is ten images so loop through all the images per batch
                # no!!!! this defeats the point of batches if you loop through each image you've essentially made your batch size 1 -hmk
                
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

            if (epoch % 10 == 0):
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
                print("\nNumber Of Images Tested =", len(val_loader)*batch_size)
                print("Validation MSE Loss =", (running_val_loss/len(val_loader)))

                last_loss = (running_val_loss/len(val_loader))

                print('Epoch {} of {}, Training MSE Loss: {:.3f}'.format( epoch+1, Epochs, running_loss/len(train_loader)))


        print('Finished Training')
        train_loss = [epoch.cpu().detach().numpy() for epoch in train_loss] # changed var from val bc this has nothing to do with validation -hmk
        validation_loss = [val.cpu().detach().numpy() for val in validation_loss]

        fig = plt.figure() # just added the val line and labels to make it pretty and saved it so it can be in the report -hmk
        plt.plot(np.arange(0,Epochs,1), train_loss, 'r', label='Training Loss')
        plt.plot(np.arange(0,Epochs,10), validation_loss,'b', label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for Batch Size = {batch_size} and learning rate = {lr}")
        plt.legend(loc="upper right")
        plt.savefig(f"bs_{batch_size}-lr_{lr}-training-val-plot.png")
        plt.show()


    # plot_path = path + slash + f'colorized_images_from_{album}' + slash + 'training-val-plot.png'

    running_test_loss = 0.0
    with torch.no_grad():
        color.eval()
        for data in test_loader:
            test_l = torch.unsqueeze(data[2], 1).to(device)
            test_outputs = color(test_l)
            test_labels = torch.stack((data[0], data[1]), 1).float().to(device)
            test_loss = criterion(test_outputs, torch.flatten(test_labels, 0, 1))
            running_test_loss += test_loss

    print("\nNumber Of Images Tested =", len(test_loader)*batch_size)
    print("Testing MSE Loss =", (running_test_loss/len(test_loader)))