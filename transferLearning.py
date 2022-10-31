# Authors: Alexander Wozny, Tim Lu, Hannah Kirkland

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
from sys import platform
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from tunedColorizer import colorizer
from pytorchtool import EarlyStopping

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
        a = (ab_color_space[:, 0])
        b = (ab_color_space[:, 1])
        l = (l_color_space)
        
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



def trainModel(color, trainLoader, valLoader, optimizer, criterion, epochs, patience, album):

    # create directories for saving if they do not exist
    home_dir = os.getcwd()
    if not os.path.exists(home_dir + slash + f"chkpt_{album}"):
        os.makedirs(home_dir + slash + f"chkpt_{album}")

    if not os.path.exists(home_dir + slash + f"chkpt_{album}" + slash + "training_images"):
        os.makedirs(home_dir + slash + f"chkpt_{album}" + slash + "training_images")

    if not os.path.exists(home_dir + slash + f"chkpt_{album}" + slash + "sample_images"):
        os.makedirs(home_dir + slash + f"chkpt_{album}" + slash + "sample_images")


    path = '.' + slash + f"chkpt_{album}" + slash + "checkpoint.pt"
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    #training loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    #loss_values = []
    train_loss = []
    validation_loss = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        color.train()
    
        running_loss = 0.0
        for i, img in enumerate(trainLoader):
            
            a = img[0] # i changed these for clarity and less typing i didn't want to type batch everytime -hmk
            b = img[1]
            l = img[2]
            
            # shape labels and input
            labels = torch.stack((a, b), 1).float().to(device)
            input_l = torch.unsqueeze(l, 1).to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = color((input_l))
                        
            #flatten labels along dimension 0
            labels = torch.flatten(labels, 0, 1)
            
            # calculate MSE loss
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
                    
        # store training loss
        train_loss.append(loss)

        running_val_loss = 0.0
        with torch.no_grad():
            color.eval()
            for data in valLoader:
                val_l = torch.unsqueeze(data[2], 1).to(device)
                val_outputs = color(val_l)
                val_labels = torch.stack((data[0], data[1]), 1).float().to(device)
                val_loss = criterion(val_outputs, torch.flatten(val_labels, 0, 1))
                running_val_loss += val_loss

        validation_loss.append(running_val_loss)
        print("\nNumber Of Images Tested =", len(valLoader)*batch_size)
        print("Validation MSE Loss =", (running_val_loss/len(valLoader)))

        if epoch % 10 == 0:
            # once done with a loop I want to print out the target image 
            # # and colorized image for comparison    
            sample_target = cv2.merge([l[0].detach().numpy(), a[0].detach().numpy(), b[0].detach().numpy()]) 
            sample_target = cv2.cvtColor(sample_target, cv2.COLOR_LAB2BGR)
            
            sample_target = cv2.merge([l[0].cpu().detach().numpy(), a[0].cpu().detach().numpy(), b[0].cpu().detach().numpy()]) 
            sample_target = cv2.cvtColor(sample_target, cv2.COLOR_LAB2BGR)
        
            colorized_a = outputs[0].cpu().detach().numpy().astype(np.uint8)
            colorized_b = outputs[1].cpu().detach().numpy().astype(np.uint8)
            sample_colorized = cv2.merge([l[0].detach().numpy(), colorized_a, colorized_b])
            sample_colorized = cv2.cvtColor(sample_colorized, cv2.COLOR_LAB2BGR)
    
            cv2.imwrite('.' + slash + f"chkpt_{album}" + slash + "training_images" + slash + f"target_image_{epoch}.png", sample_target)
            cv2.imwrite('.' + slash + f"chkpt_{album}" + slash + "training_images" + slash + f"output_image_{epoch}.png", sample_colorized)

        # use early stopping to prevent overfitting
        early_stopping(running_val_loss/len(valLoader), color)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('Epoch {} of {}, Training MSE Loss: {:.3f}'.format( epoch+1, epochs, running_loss/len(trainLoader)))



# hyperparameters for training
batch_size = 50
Epochs = 200
lr = 0.1

home_dir = os.getcwd() 

#change this parameter depending on which album you want
target_album = 'LAB_COLORS'
album = 'fruit'

# get names of folders for colorful fruit data
foodFolders = [name for name in os.listdir("." + slash + target_album)]

# get fruit images
food_images = []
for ff in foodFolders:
    food_images.extend(load(home_dir + slash + target_album + slash + ff + slash + '*.jpg'))

# album_length = len(food_images)

for i, val in enumerate(food_images):
    food_images[i] = cv2.resize(val, (128, 128))
food_images_lab = convert_LAB(food_images)
saveLAB(food_images_lab, "LAB_COLORS")

# load L*a*b* images
food_data = LoadLabInOrder(home_dir + slash + target_album + slash)
album_length = len(food_data)

#group images into sets of 3   
food_grouped_data = list(group(food_data, album_length))
food_grouped_data = np.asarray(food_grouped_data)
    
#prepare grouped data for training and test
food_train_images, food_test_images = train_test_split(food_grouped_data, test_size = 0.3)
food_train_images, food_val_images = train_test_split(food_train_images, test_size = 0.1)

#further separate them into X's and Y's where L is the input and AB are the targets (LAB colorspace)
#remember the dimensions are Number of grouped images X Index of image
food_X_train = food_train_images[:, 2, :, :, 0]
food_y_train = food_train_images[:, 0:2, :, :, 0]

food_X_test = food_test_images[:, 2, :, :, 0]
food_y_test = food_test_images[:, 0:2, :, :, 0]

food_X_val = food_val_images[:, 2, :, :, 0]
food_y_val = food_val_images[:, 0:2, :, :, 0]


#prepare datasets for images
food_train_dataset = imageDataset(food_X_train, food_y_train)
food_test_dataset = imageDataset(food_X_test, food_y_test)
food_val_dataset = imageDataset(food_X_val, food_y_val)

# prepare dataloaders for batch training
food_train_loader = DataLoader(dataset = food_train_dataset, batch_size = batch_size, shuffle=True)
food_test_loader = DataLoader(dataset = food_test_dataset,  batch_size = batch_size, shuffle=True)
food_val_loader = DataLoader(dataset = food_val_dataset,  batch_size = batch_size, shuffle=True)

# load fine tuned face model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
path = '.' + slash + f"chkpt_{album}" + slash + "colorizer_finetuned.pt"
cModel = colorizer().to(device)
cModel.load_state_dict(torch.load(path))
cModel.eval()

# freeze layers in model
cModel.downsamp1.requires_grad=False
cModel.downsamp2.requires_grad=False
cModel.downsamp3.requires_grad=False
cModel.downsamp4.requires_grad=False
cModel.downsamp5.requires_grad=False
cModel.upsamp1.requires_grad=False
cModel.upsamp2.requires_grad=False
cModel.upsamp3.requires_grad=False

# allow training of weights for last layer
cModel.upsamp4.requires_grad=True
cModel.upsamp5.requires_grad=True


# parameters for training
criterion = nn.MSELoss()
torch.set_default_tensor_type(torch.FloatTensor)
optimizer = torch.optim.Adam(cModel.parameters(), lr)

# train transfer model on fruit/vegetable images
trainModel(cModel, food_train_loader, food_val_loader, optimizer, criterion, epochs=Epochs, patience=10, album='fruit')

# calculate and print the loss on the test images
running_test_loss = 0.0
result = []
bestModel = colorizer().to(device)
bestModel.load_state_dict(torch.load('.' + slash + f"chkpt_{album}" + slash + "checkpoint.pt"))

with torch.no_grad():
    bestModel.eval()
    
    # for each test image
    for i, data in enumerate(food_test_loader):

        # get the inputs and outputs
        test_l = torch.unsqueeze(data[2], 1).to(device)
        test_outputs = bestModel(test_l)
        test_labels = torch.stack((data[0], data[1]), 1).float().to(device)
        test_loss = criterion(test_outputs, torch.flatten(test_labels, 0, 1))
        running_test_loss += test_loss
        
        # get the images from the last batch
        if i == len(food_test_loader)-1:
            
            # L*a*b* channel data
            a = data[0]
            b = data[1]
            l = data[2]

            test_l = torch.unsqueeze(l, 1).to(device)
            outputs = bestModel(test_l)
            test_a = torch.unsqueeze(a, 1).to(device)
            test_b = torch.unsqueeze(b, 1).to(device)

print("\nNumber Of Images Tested =", len(food_test_loader)*batch_size)
print("Testing MSE Loss =", (running_test_loss/len(food_test_loader)))

# save model sample predictions on the test data (this just uses the last batch of data)
for i in range(1, l.shape[0]):
    colorized_a = outputs[2*i-2].cpu().detach().numpy().astype(np.uint8)
    colorized_b = outputs[2*i-1].cpu().detach().numpy().astype(np.uint8)
    colorized_l = l[i-1].detach().numpy()
    sample_colorized = cv2.merge([colorized_l, colorized_a, colorized_b])
    sample_colorized = cv2.cvtColor(sample_colorized, cv2.COLOR_LAB2BGR)
    cv2.imwrite('.' + slash + f"chkpt_{album}" + slash + "sample_results" + slash + f"output_image_{i}.png", sample_colorized)