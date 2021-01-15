# -*- coding: utf-8 -*-
#Loading of the libraries needed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import time
import numpy as np, random
from PIL import Image
import os
import torch.utils.data as data
from skimage import io, transform, img_as_float
import PIL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
Image_size = 256 # This one is to be changed with the image size that we want to train with: (Image_size X Image_size)

model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
in_features = model.fc.in_features
# In the classification task, we have only 2 classes. 0 for "no Car" and 1 when "at least a car".
out_features = 2
model.fc = nn.Linear(in_features,out_features)
model.to(device)
print("Anything to omit showing the architecture! ;p")

"""### Preparing necessary functions"""

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


nbr_img_train_columbus = 6966
nbr_img_test_columbus = 2050
nbr_img_train_postdam  = 10429
nbr_img_test_postdam = 2009
nbr_img_train_selwyn = 16287
nbr_img_test_selwyn = 4942
print("nbr_img_train_postdam = "+str(nbr_img_train_postdam))
print("nbr_img_test_postdam = "+str(nbr_img_test_postdam))
print("nbr_img_train_selwyn = "+str(nbr_img_train_selwyn))
print("nbr_img_test_selwyn = "+str(nbr_img_test_selwyn))
# For the other images, I am not going to treat them for now. Let's see how much accuracy do we reach.

Path_columbus ="Columbus_CSUAV_AFRL"
transformation = transforms.Compose([
    transforms.ToTensor()
])
training_datasets = []
train_dataset = Dataset(starting_index=0,number_images=nbr_img_train_columbus, root_dir=Path_columbus
                        , folder = '/train',label_file = '/COWC_train_list_detection.txt', transform = transformation)
training_datasets.append(train_dataset)
starting_index_train = nbr_img_train_columbus
Path_postdam = "Potsdam_ISPRS"
train_dataset = Dataset(starting_index=starting_index_train ,number_images=nbr_img_train_postdam, root_dir=Path_postdam
                        , folder = '/train',label_file = '/COWC_train_list_detection.txt', transform = transformation)
training_datasets.append(train_dataset)
starting_index_train = starting_index_train + nbr_img_train_postdam
Path_selwyn = "Selwyn_LINZ"
train_dataset = Dataset(starting_index=starting_index_train ,number_images=nbr_img_train_selwyn, root_dir=Path_selwyn
                        , folder = '/train',label_file = '/COWC_train_list_detection.txt', transform = transformation)
training_datasets.append(train_dataset)
train_dataset = torch.utils.data.ConcatDataset(training_datasets)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

testing_datasets = []
test_dataset = Dataset(starting_index=0,number_images=nbr_img_test_columbus, root_dir=Path_columbus
                        , folder = '/test',label_file = '/COWC_test_list_detection.txt', transform = transformation)
testing_datasets.append(test_dataset)
starting_index_test = nbr_img_test_columbus
Path_postdam = "Potsdam_ISPRS"
test_dataset = Dataset(starting_index=starting_index_test ,number_images=nbr_img_test_postdam, root_dir=Path_postdam
                        , folder = '/test',label_file = '/COWC_test_list_detection.txt', transform = transformation)
testing_datasets.append(test_dataset)
starting_index_test = starting_index_test + nbr_img_test_postdam
Path_selwyn = "Selwyn_LINZ"
test_dataset = Dataset(starting_index=starting_index_test ,number_images=nbr_img_test_selwyn, root_dir=Path_selwyn
                        , folder = '/test',label_file = '/COWC_test_list_detection.txt', transform = transformation)
testing_datasets.append(test_dataset)
test_dataset = torch.utils.data.ConcatDataset(testing_datasets)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""### Training for Classification"""





model = train(model, train_dataloader,epochs=30, lr=LEARNING_RATE)

print(test(model, test_dataloader))

torch.save({'model_state_dict':model.state_dict(),},'./classification_model')



