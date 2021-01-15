#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import os
from tqdm import tqdm
import time
import numpy as np, random
from PIL import Image
import os
import torch.utils.data as data
from skimage import io, transform, img_as_float
import PIL
import random
import pickle
tr = torchvision.transforms.ToPILImage()

transformation = transforms.Compose([
    transforms.ToTensor(),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr = transforms.ToPILImage()
BATCH_SIZE = 5
LEARNING_RATE = 0.01
lmbda=0.1
ITERATIONS = 1
EPOCHS = 1
Image_size = 256 # This one is to be changed with the image size that we want to train with: (Image_size X Image_size)
model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
in_features = model.fc.in_features
out_features = 2
model.fc = nn.Linear(in_features,out_features)
model_loader = torch.load("./classification_model",map_location=torch.device('cpu'))
model.load_state_dict(model_loader['model_state_dict'])
out_features = 64
model.fc = nn.Linear(in_features,out_features)
#model_loader = torch.load("./counting_model",map_location=torch.device('cpu'))
#model.load_state_dict(model_loader['model_state_dict'])
model.to(device)
'''
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Columbus_CSUAV_AFRL.tbz")
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Potsdam_ISPRS.tbz")
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Selwyn_LINZ.tbz")
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Toronto_ISPRS.tbz")
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Vaihingen_ISPRS.tbz")
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Utah_AGRC.tbz")


os.system("tar -xvjf ./COWC_Counting_Columbus_CSUAV_AFRL.tbz")
os.system("tar -xvjf ./COWC_Counting_Potsdam_ISPRS.tbz")
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_test_list_64_class.txt.bz2")
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_train_list_64_class.txt.bz2")
os.system("bzip2 -d COWC_test_list_64_class.txt.bz2")
os.system("bzip2 -d COWC_train_list_64_class.txt.bz2")
os.system("cp COWC_test_list_64_class.txt ./Columbus_CSUAV_AFRL/")
os.system("cp COWC_train_list_64_class.txt ./Potsdam_ISPRS/")
os.system("cp COWC_train_list_64_class.txt ./Columbus_CSUAV_AFRL/")
os.system("cp COWC_test_list_64_class.txt ./Potsdam_ISPRS/")
'''
Path_columbus ="Columbus_CSUAV_AFRL"
nbr_img_train_columbus = 7595
nbr_img_test_columbus = 2110
nbr_img_train_postdam = 10722
nbr_img_test_postdam = 2078
# Definition of DataSet creator, This needs to be revisited because, I do not have the format of the .txt file
class Dataset(torch.utils.data.Dataset):

    def __init__(self,starting_index, number_images, root_dir, folder, label_file , transform=None):
        # Number of images to be loaded.
        self.number_images = number_images
        self.starting_index = starting_index
        with open(root_dir+label_file, 'r') as file:
            lines=file.read().splitlines()
        self.Names_labels = []
        for element in lines:
            self.Names_labels.append(element.split(' '))
        
        # Here I construct the path to the images
        self.root_dir = root_dir+folder
        # this is to collect the transform, if I supply any.
        self.transform = transform

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.Names_labels[self.starting_index+idx][0].split('/')[-1])
        image = io.imread(img_name)
        resized_img = transform.resize(image, (Image_size, Image_size))
        sample = img_as_float(resized_img)
        target = int(float(self.Names_labels[idx+self.starting_index][1]))

        if self.transform:
            sample = self.transform(sample)
        
        return [sample.float(), target]
#Definition of train and test functions!
def train(model, trainloader,epochs=2, lr=LEARNING_RATE):
    total_loss = 0
    # We need to verify whether we want to modify the Loss function. I believe we do but I am not sure yet.
    loss_function=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_correct=0
        for batch_idx, (images, labels) in enumerate(trainloader):
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad() 
            preds= model(images)
            loss = loss_function(preds,labels)
            loss.backward() 
            optimizer.step() 
            total_loss += loss.item()
    return model
def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
training_datasets = []
train_dataset = Dataset(starting_index=0,number_images=nbr_img_train_columbus, root_dir=Path_columbus
                        , folder = '/train',label_file = '/COWC_train_list_64_class.txt', transform = transformation)
training_datasets.append(train_dataset)
starting_index_train = nbr_img_train_columbus
Path_postdam = "Potsdam_ISPRS"
train_dataset = Dataset(starting_index=starting_index_train ,number_images=nbr_img_train_postdam, root_dir=Path_postdam
                        , folder = '/train',label_file = '/COWC_train_list_64_class.txt', transform = transformation)
training_datasets.append(train_dataset)
'''
starting_index_train = starting_index_train + nbr_img_train_postdam
Path_selwyn = "Selwyn_LINZ"
train_dataset = Dataset(starting_index=starting_index_train ,number_images=nbr_img_train_selwyn, root_dir=Path_selwyn
                        , folder = '/train',label_file = '/COWC_train_list_64_class.txt', transform = transformation)
training_datasets.append(train_dataset)
'''
train_dataset = torch.utils.data.ConcatDataset(training_datasets)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testing_datasets = []
test_dataset = Dataset(starting_index=0,number_images=nbr_img_test_columbus, root_dir=Path_columbus
                        , folder = '/test',label_file = '/COWC_test_list_64_class.txt', transform = transforms.ToTensor())
testing_datasets.append(test_dataset)
starting_index_test = nbr_img_test_columbus
Path_postdam = "Potsdam_ISPRS"
test_dataset = Dataset(starting_index=starting_index_test ,number_images=nbr_img_test_postdam, root_dir=Path_postdam
                        , folder = '/test',label_file = '/COWC_test_list_64_class.txt', transform = transforms.ToTensor())
testing_datasets.append(test_dataset)
'''
starting_index_test = starting_index_test + nbr_img_test_postdam
Path_selwyn = "Selwyn_LINZ"
test_dataset = Dataset(starting_index=starting_index_test ,number_images=nbr_img_test_selwyn, root_dir=Path_selwyn
                        , folder = '/test',label_file = '/COWC_test_list_detection.txt', transform =transforms.ToTensor())
testing_datasets.append(test_dataset)
'''
test_dataset = torch.utils.data.ConcatDataset(testing_datasets)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#torch.save({'model_state_dict':model.state_dict(),},'./counting_model')

#print(test(model, test_dataloader))
for i in range(300):
    model = train(model, train_dataloader,epochs=1, lr=LEARNING_RATE)
    #print(test(model, train_dataloader))
    print(test(model, test_dataloader))
    torch.save({'model_state_dict':model.state_dict(),},'./counting_model')
