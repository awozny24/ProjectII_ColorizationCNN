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

# from colorizerII import colorizer
from cnnColorizer import colorizer, trainModel


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
        f = f[f.rfind('/')+1:]
        files[i] = f[0:f.rfind('.')-1]
    maxFileNum = max([int(f) for f in files])
    
    # for each file index (e.g. ['0L.jpg', '0a.jpg', '0b.jpg'])
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


if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'

    
home_dir = os.getcwd() 

#change this parameter depending on which album you want
target_album = 'ColorfulLab'
if target_album == 'LAB_TEST_FACES':
    album = 'faces'
else:
    album = 'ColorfulLab'

# if the specified directory does not exist, or if it exists but is empty
if not os.path.exists(home_dir + slash + target_album) \
    or (os.path.exists(home_dir + slash + target_album) and not [name for name in os.listdir("." + slash + target_album)]):

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
    saveLAB(food_images_lab, "ColorfulLab")

    # plot rgb image
#     plt.imshow(cv2.cvtColor(food_images[0], cv2.COLOR_BGR2RGB))


batch_size = 32
Epochs = 100
lr = 0.01
criterion = nn.MSELoss()
torch.set_default_tensor_type(torch.FloatTensor)

home_dir = os.getcwd() 
#change this parameter depending on which album you want
target_album = 'ColorfulLab'
if target_album == 'LAB_TEST_FACES':
    album = 'faces'
else:
    album = 'ColorfulLab'


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
#this needs to be flipped

food_X_train = food_train_images[:, 2, :, :, 0]#.astype(dtype=object)
food_y_train = food_train_images[:, 0:2, :, :, 0]#.astype(dtype=object)

food_X_test = food_test_images[:, 2, :, :, 0]#.astype(dtype=object)
food_y_test = food_test_images[:, 0:2, :, :, 0]#.astype(dtype=object)

food_X_val = food_test_images[:, 2, :, :, 0]#.astype(dtype=object)
food_y_val = food_test_images[:, 0:2, :, :, 0]#.astype(dtype=object)


#prepare datasets for images
food_train_dataset = imageDataset(food_X_train, food_y_train)
food_test_dataset = imageDataset(food_X_test, food_y_test)
food_val_dataset = imageDataset(food_X_val, food_y_val)

# prepare dataloaders for batch training
food_train_loader = torch.utils.data.DataLoader(dataset = food_train_dataset, batch_size = batch_size, shuffle=True)
food_test_loader = torch.utils.data.DataLoader(dataset = food_test_dataset,  batch_size = batch_size, shuffle=True)
food_val_loader = torch.utils.data.DataLoader(dataset = food_val_dataset,  batch_size = batch_size, shuffle=True)


cModel = torch.load('./saved_models/model_architecture_11.pt')
        
#run color regressor
lr = 0.01
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
optimizer = torch.optim.Adam(cModel.parameters(), lr)
trainModel(cModel, food_train_loader, food_val_loader, optimizer, 90, 'fruit')


running_test_loss = 0.0
with torch.no_grad():
    cModel.eval()
    for data in food_test_loader:
        test_l = torch.unsqueeze(data[2], 1).to(device)
        test_outputs = cModel(test_l)
        test_labels = torch.stack((data[0], data[1]), 1).float().to(device)
        test_loss = criterion(test_outputs, torch.flatten(test_labels, 0, 1))
        running_test_loss += test_loss

print("\nNumber Of Images Tested =", len(food_test_loader)*batch_size)
print("Testing MSE Loss =", (running_test_loss/len(food_test_loader)))