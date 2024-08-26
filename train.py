#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import time
import copy
import glob

import torch
import torchvision
import torchvision.transforms as transforms
#from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import flor


# In[3]:


device = torch.device(flor.arg("device", "cuda" if torch.cuda.is_available() else "cpu"))
batch_size = flor.arg("batch_size", 32)
data_path = flor.arg("data_path", '/home/ubuntu/IDNet/split_normal/fin2_sidtd/')
model_path = data_path + 'models/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
train_path = data_path + 'train'

resize = flor.arg("resize", True)
hflip = flor.arg("hflip", False)
normalize = flor.arg("normalize", False)
model_name = flor.arg("model_name", "efficientnet_b3")
composes1 = []
composes2 = []
if resize:
    composes1.append(transforms.Resize([224, 224]))
    composes2.append(transforms.Resize([224, 224]))
else:
    composes1.append(transforms.Resize([584, 931]))
    composes2.append(transforms.Resize([584, 931]))
if hflip:
    composes1.append(transforms.RandomHorizontalFlip())
composes1.append(transforms.ToTensor())
composes2.append(transforms.ToTensor())
if normalize:
    composes1.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
    composes2.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
#transform = transforms.Compose(
#                [
#                    transforms.Resize([224, 224]),
#                    #transforms.RandomResizedCrop(224),
#                    transforms.RandomHorizontalFlip(),
#                    transforms.ToTensor(),
#                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#                ])

#transform = transforms.Compose(
#                [
#                    #transforms.Resize([224, 224]),
#                    transforms.Resize([584,931]),
#                    #transforms.RandomResizedCrop(224),
#                    transforms.RandomHorizontalFlip(),
#                    transforms.ToTensor(),
#                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                    #             std=[0.229, 0.224, 0.225])
#                ])
transform1 = transforms.Compose(composes1)
transform2 = transforms.Compose(composes2)
train_dataset = torchvision.datasets.ImageFolder(
    root=train_path,
    transform=transform1
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size= batch_size,
    num_workers=8,
    shuffle=True
)

print(len(train_loader))


# In[4]:


val_path = data_path + 'val'

val_dataset = torchvision.datasets.ImageFolder(
    root=val_path,
    transform=transform2
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size= batch_size,
    num_workers=8,
    shuffle=True
)

print(len(val_loader))


# In[5]:


test1_path = data_path + 'test'

test1_dataset = torchvision.datasets.ImageFolder(
    root=test1_path,
    transform=transform2
)
test1_loader = torch.utils.data.DataLoader(
    test1_dataset,
    batch_size= batch_size,
    num_workers=8,
    shuffle=True
)

print(len(test1_loader))

test2_path = data_path + 'test_sidtd'

test2_dataset = torchvision.datasets.ImageFolder(
    root=test2_path,
    transform=transform2
)
test2_loader = torch.utils.data.DataLoader(
    test2_dataset,
    batch_size= batch_size,
    num_workers=8,
    shuffle=True
)

print(len(test2_loader))



#batch = next(iter(train_loader))
#print(batch[0].shape)
#plt.imshow(batch[0][0].permute(1, 2, 0))
#print(batch[1][0])



if model_name == "efficientnet_b3":
    resnet18 = models.efficientnet_b3(pretrained=True)
    resnet18.fc = nn.Linear(1536, 2)
elif model_name == "resnet50":
    resnet18 = models.resnet50(pretrained=True)
    resnet18.fc = nn.Linear(2048, 2)
else:
    print("Model name not supported")

#resnet18 = models.resnet18(pretrained=True)
#resnet18 = models.resnet50(pretrained=True)




# # # Feature Extracting a Pretrained Model
# 
# Since this pretrained model is trained on ImageNet dataset, the output layers has 1000 nodes. We want to reshape this last classifier layer to fit this dataset which has 2 classes. Furthermore, in feature extracting, we don't need to calculate gradient for any layers except the last layer that we initialize. For this we need to set `.requires_grad` to `False`

# In[10]:


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
#set_parameter_requires_grad(resnet18)


# In[11]:


# Initialize new output layer


# In[12]:



# Check which layer in the model that will compute the gradient
#for name, param in resnet18.named_parameters():
#    if param.requires_grad:
#        print(name, param.data.shape)


# In[13]:
def validate(model, valloaders, criterion, device):
    model.eval()
    running_corrects = 0
    running_loss = 0.0

    with torch.no_grad():
        # Iterate over data.
        for inputs, labels in tqdm(valloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        val_acc = running_corrects.double() / len(valloaders.dataset)
        val_loss = running_loss / len(valloaders.dataset)
        return val_acc, val_loss


from tqdm import tqdm
def train_model(model, trainloaders, valloaders, test1loaders, test2loaders, 
                criterion, optimizer, device, num_epochs=5, is_train=True):
    since = time.time()
    
    acc_history = []
    loss_history = []
    model.to(device)
    best_acc = 0.0
    best_loss = 1e10
    
    with flor.checkpointing(model=model, optimizer = optimizer):
        for epoch in flor.loop("epoch", range(num_epochs)):

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            model.train()
            for inputs, labels in tqdm(trainloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_epoch_loss = running_loss / len(trainloaders.dataset)
            train_epoch_acc = running_corrects.double() / len(trainloaders.dataset)

            print('Training: Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, train_epoch_acc))
            flor.log("train_loss", train_epoch_loss)
            flor.log("train_acc", train_epoch_acc.item())
            
            val_epoch_acc, val_epoch_loss = validate(model, valloaders, criterion, device)
            flor.log("val_loss", val_epoch_loss)
            flor.log("val_acc", val_epoch_acc.item())
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
            print(f"Saved model on epoch {epoch}")
            torch.save(model, os.path.join(model_path, '{0:0=2d}.pth'.format(epoch)))

            val_epoch_acc, val_epoch_loss = validate(model, test1loaders, criterion, device)
            flor.log("test_idnet_loss", val_epoch_loss)
            flor.log("test_idnet_acc", val_epoch_acc.item())

            val_epoch_acc, val_epoch_loss = validate(model, test2loaders, criterion, device)
            flor.log("test_sidtd_loss", val_epoch_loss)
            flor.log("test_sidtd_acc", val_epoch_acc.item())


            acc_history.append(val_epoch_acc.item())
            loss_history.append(val_epoch_loss)
            
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        return acc_history, loss_history


# In[14]:


# Here we only want to update the gradient for the classifier layer that we initialized.
params_to_update = []
for name,param in resnet18.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        #print("\t",name)
            
optimizer = optim.Adam(params_to_update)


# In[15]:



# Setup the loss function
criterion = nn.CrossEntropyLoss()

# Train model
train_acc_hist, train_loss_hist = train_model(resnet18, train_loader, val_loader, test1_loader, 
                                              test2_loader, criterion, optimizer, device)


# In[16]:
import sys
sys.exit(0)


test_path = data_path + 'test'

test_dataset = torchvision.datasets.ImageFolder(
    root=test_path,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=1,
    shuffle=False
)

print(len(test_loader))


# In[17]:


def eval_model(model, dataloaders, device):
    since = time.time()
    print(model_path)
    acc_history = []
    best_acc = 0.0
    saved_models = glob.glob(model_path + '*.pth')
    saved_models.sort()
    print('saved_model', saved_models)

    for mp in saved_models:
        print('Loading model', mp)

        #model.load_state_dict(torch.load(mp))
        model = torch.load(mp)
        model.eval()
        model.to(device)

        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print('Acc: {:.4f}'.format(epoch_acc))
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    
    return acc_history


# In[18]:


val_acc_hist = eval_model(resnet18, test_loader, device)


# In[19]:


plt.plot(train_acc_hist)
plt.plot(val_acc_hist)
plt.show()


# In[20]:


plt.plot(train_loss_hist)
plt.show()


# In[ ]:




