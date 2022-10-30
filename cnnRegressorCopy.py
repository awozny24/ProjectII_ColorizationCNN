# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:24:56 2022

@author: Timothy Lu
"""
import cv2
import os
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
import torchvision.transforms as T


# CNN resource: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
if platform == 'darwin':
    slash = '/'
else:
    slash = '\\'


def load(folder):
    files = glob.glob(folder)
    data = []
    for f in files:
        image = cv2.imread(f)
        data.append(image)
    return data


def group(data, album_length):
    # group into chunks of three because of three sets of images in LAB color space
    for i in range(0, album_length, 3):
        yield image_data[i:i+3]


def makeChrome(a_or_b, a=None, b=None):
    grayscale = np.ones((128, 128, 1), dtype=np.float32)
    other = np.zeros((128, 128, 1), dtype=np.float32)

    if a_or_b == 0:
        # use A channel
        a_channel = np.full((128, 128, 1), a, dtype=np.float32)
        chrome_image = cv2.merge([grayscale, a_channel, other])

        # remember this is not rgb
       # chrome_image = cv2.cvtColor(chrome_image, cv2.COLOR_LAB2RGB)
        return chrome_image

    elif a_or_b == 1:
        # use B channel
        b_channel = np.full((128, 128, 1), b, dtype=np.float32)
        chrome_image = cv2.merge([grayscale, other, b_channel])

       # chrome_image = cv2.cvtColor(chrome_image, cv2.COLOR_LAB2RGB)
        return chrome_image

    elif a_or_b == 2:
        # use both channels
        a_channel = np.full((128, 128, 1), a, dtype=np.float32)
        b_channel = np.full((128, 128, 1), b, dtype=np.float32)
        chrome_image = cv2.merge([grayscale, a_channel, b_channel])

      #  chrome_image = cv2.cvtColor(chrome_image, cv2.COLOR_LAB2RGB)
        return chrome_image


class imageDataset(Dataset):
    def __init__(self,  l_color_space, ab_color_space,):
        a = (ab_color_space[:, 0, :, :, 0])
        b = (ab_color_space[:, 1, :, :, 0])
        l = (l_color_space[:,:,:,0])

        # it seems that I will have to use permute
        # to get from numpy image representation
        # to torch tensor image

        self.a = a
        self.b = b
        self.l = l
        self.indices = len(l)

    def __len__(self):
        return self.indices

    def __getitem__(self, index):
        return self.a[index], self.b[index], self.l[index]


class chrominance_reg(nn.Module):
    def __init__(self):
        super(chrominance_reg, self).__init__()

        # 128x128
        self.mod1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=(1, 1), stride=1)
        )
        # 64x64
        self.mod2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=1)
        )
        # 32x32
        self.mod3 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=1)
        )
        # 16x16
        self.mod4 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=1)
        )
        # 8x8
        self.mod5 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=1)
        )
        # 4x4
        self.mod6 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=1)
        )
        # 2x2
        self.mod7 = nn.Sequential(

            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(384, 2)
        )

    def forward(self, x):
        # Normalize input first
        # make_Tensor = T.ToTensor()
        # transform_pil = T.ToPILImage()

        # out = make_Tensor(x)
        # out = transform_pil(x)
        # out = normalize(x)
        out = self.mod1(x)
        out = self.mod2(out)
        out = self.mod3(out)
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        # out = torch.mean(out, 0, True)
        return out


# select GPU / CPU -hmk
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# set paths
# resource:
# https://www.kaggle.com/code/anirbanmalick/image-colorization-pytorch-conv-autoencoder
home_dir = os.getcwd()
# egg = cv2.imread(home_dir + slash + 'ColorfulOriginal' +
#                  slash + 'Brinjal' + slash + 'Brinjal1.jpg')
# change this parameter depending on which album you want
target_album = 'LAB_TEST_FACES'

image_data = load(home_dir + slash + target_album + slash + '*.jpg')
album_length = len(image_data)

# group images into sets of 3
grouped_data = list(group(image_data, album_length))
grouped_data = np.asarray(grouped_data)

# prepare grouped data for training and test
train_images, test_images = train_test_split(grouped_data, test_size=0.3)
train_images, val_images = train_test_split(train_images, test_size=0.1)

# further separate them into X's and Y's where L is the input and AB are the targets (LAB colorspace)
# remember the dimensions are Number of grouped images X Index of image
# this needs to be flipped

X_train = train_images[:, 2, :, :, :]
y_train = train_images[:, 0:2, :, :, :]

X_test = test_images[:, 2, :, :, :]
y_test = test_images[:, 0:2, :, :, :]

X_val = val_images[:, 2, :, :, :]
y_val = val_images[:, 0:2, :, :, :]

# prepare datasets for images
train_dataset = imageDataset(X_train, y_train)
test_dataset = imageDataset(X_test, y_test)
val_dataset = imageDataset(X_val, y_val)

# prepare dataloaders for batch training
# create datasets

batch_size = 32
Epochs = 100
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                         batch_size=batch_size, shuffle=True)
                         
regressor = chrominance_reg().to(device)


# what is this?? -hmk
# #call the regressor on one set of images in the X_train dataset
# regressor = chrominance_reg()
# #run forward pass on one grayscale image
# sample_a, sample_b, sample_grayscale = train_dataset[0]
# chrome = regressor(sample_grayscale)


# chrome_a = makeChrome(0, chrome[0][0].detach().numpy(), None)
# chrome_b = makeChrome(1, None, chrome[0][1].detach().numpy())
# chrome_all = makeChrome(2, chrome[0][0].detach().numpy(), chrome[0][1].detach().numpy())

# run color regressor
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(regressor.parameters(), lr)


# display predicted chrominances


# training loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

train_loss = []
validation_loss = []
val_ticker = 0
last_loss = 20000

for epoch in range(Epochs):  # loop over the dataset multiple times
    regressor.train()

    running_loss = 0.0
    # I want batch to be of length 10 not 3 why?
    for i, batch in enumerate(train_loader):

        a = batch[0]  # removed batch from var name bc i dont like typing
        b = batch[1]
        l = batch[2]

        # each batch is ten images so loop through all the images per batch
        # once again, this completely destroys the point of batching dont do this!! -hmk

        # for index, images in enumerate(batch):
        #     # get the inputs; data is a list of tensors [chrominance_a_tensor, chrominance_b_tensor, grayscale_l_tensor]
        #   #different images!
        label_a = normalize(a.float())
        label_b = normalize(b.float())
        input_l = normalize(l.float())

        # need to get the mean of labels across all dimensions
        mean_a = torch.mean(label_a, dim=[1, 2])
        mean_b = torch.mean(label_b, dim=[1, 2])

        labels = torch.stack((mean_a, mean_b), 1).to(device)
        input_l = torch.unsqueeze(l, 1).float().to(device)
        # add new axis to make it consistent with dimension of regressor output tensor
        # labels = torch.unsqueeze(labels, 0)
        # print(labels.size())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = regressor(input_l)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    train_loss.append(loss)

    if epoch % 10 == 0: # added validation -hmk
        running_val_loss = 0.0
        with torch.no_grad():
            regressor.eval()
            for data in val_loader:
                a = data[0]  # removed batch from var name bc i dont like typing
                b = data[1]
                l = data[2]
                
                label_a = normalize(a.float())
                label_b = normalize(b.float())
                input_l = normalize(l.float())

                # need to get the mean of labels across all dimensions
                mean_a = torch.mean(label_a, dim=[1, 2])
                mean_b = torch.mean(label_b, dim=[1, 2])

                labels = torch.stack((mean_a, mean_b), 1).to(device)
                input_l = torch.unsqueeze(l, 1).float().to(device)

                val_outputs = regressor(input_l)

                val_loss = criterion(val_outputs, labels)
                running_val_loss += val_loss

        validation_loss.append(running_val_loss)
        print("\nNumber Of Images Tested =", len(val_loader)*batch_size)
        print("Validation MSE Loss =", (running_val_loss/len(val_loader)))

        if (running_val_loss/len(val_loader)) - last_loss >= 0.1e-8:
            path = f"./chkpt_regressor/regressor_model_{epoch}.pt"
            torch.save(regressor.state_dict(), path)
        last_loss = (running_val_loss/len(val_loader))


    print('Epoch {} of {}, Training MSE Loss: {:.3f}'.format(epoch+1, Epochs, loss))


print('Finished Training')
train_loss = [epoch.cpu().detach().numpy() for epoch in train_loss] # changed var from val bc this has nothing to do with validation -hmk
validation_loss = [val.cpu().detach().numpy() for val in validation_loss]
plt.figure() # just added the val line and labels to make it pretty and saved it so it can be in the report -hmk
plt.plot(np.arange(0,Epochs,1), train_loss, 'r', label='Training Loss')
plt.plot(np.arange(0,Epochs,10), validation_loss,'b', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc="upper right")
plt.savefig(f"./chkpt_regressor/training-val-plot.png")
plt.show()

# testing time!! -hmk
regressor.load_state_dict(torch.load(path))

running_test_loss = 0.0
with torch.no_grad():
    regressor.eval()
    for data in test_loader:
        a = data[0]  # removed batch from var name bc i dont like typing
        b = data[1]
        l = data[2]
        
        label_a = normalize(a.float())
        label_b = normalize(b.float())
        input_l = normalize(l.float())

        # need to get the mean of labels across all dimensions
        mean_a = torch.mean(label_a, dim=[1, 2])
        mean_b = torch.mean(label_b, dim=[1, 2])

        labels = torch.stack((mean_a, mean_b), 1).to(device)
        input_l = torch.unsqueeze(l, 1).float().to(device)

        test_outputs = regressor(input_l)

        test_loss = criterion(test_outputs, labels)
        running_test_loss += test_loss

print("\nNumber Of Images Tested =", len(test_loader)*batch_size)
print("Testing MSE Loss =", (running_test_loss/len(test_loader)))
