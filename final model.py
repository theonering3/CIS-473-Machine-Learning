#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import timm
import time
import math
import torch
import torchvision
import skimage
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[2]:


#Define DataSet class for the use of DataLoader class

class MaskDataset(Dataset):
    def __init__(self,labeldir,imgdir,transform):
        self.labeldata=pd.read_csv(labeldir)
        self.rootdir= imgdir
        self.transform=transform
    def __getitem__(self,index):
        img_path=os.path.join(self.rootdir,self.labeldata.iloc[index,0])
        image=skimage.io.imread(img_path)
        image=self.transform(image)
        label=torch.tensor(self.labeldata.iloc[index,1])
        return image,label
    def __len__(self):
        return self.labeldata.shape[0]


# In[3]:


#define dir
trainlabel="D:\\python\\CIS473\\CIS-473-Machine-Learning\\mask_data\\labels\\train_label.csv"
traindir="D:\\python\\CIS473\\CIS-473-Machine-Learning\\mask_data\\traindata\\"
testlabel="D:\\python\\CIS473\\CIS-473-Machine-Learning\\mask_data\\labels\\test_label.csv"
testdir="D:\\python\\CIS473\\CIS-473-Machine-Learning\\mask_data\\testdata\\"


# In[4]:


#parameters
device=torch.device("cuda")
LossFunction = nn.CrossEntropyLoss()
loss_base,acc_base=0,0


# In[5]:


wandb.login()


# In[6]:


#config for wandb
sweep_config={
    "method":"random",
    "metric":{
        "name":"loss",
        "goal":"minimize"
    },
    "parameters":{
        "optimizer":{
            "values":["SGD","Adam"]
        },
        "dropout":{
            "values":[0.0,0.3]
        },
        "epochs":{
            "values":[10]
        },
        "learning_rate":{
            "values":[0.005,0.01,0.05]
        },
        "batch_size":{
            "values":[1]
        },
        "data_augmentation":{
            "values":["less","more"]
        },
        "weight_decay":{
            "values":[0.0,0.3]
        }
            
    }
}


# In[7]:


#sweep_id=wandb.sweep(sweep_config,project="Mask")
sweep_id="t9q2ibtq"


# In[5]:


#helper functions for wandb
def build_optimizer(optimizer,model,lr,wd):
    if optimizer=="Adam":
        return optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    elif optimizer=="SGD":
        return optim.SGD(model.parameters(), lr=lr,weight_decay=wd)
    else:
        return optim.RMSprop(model.parameters(), lr=lr,weight_decay=wd)

def build_lr_scheduler(lr_scheduler,optimizer):
    if lr_scheduler=="ROP":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2)
    else:
        return optim.lr_scheduler.MultiStepLR(optimizer,milestones=[9,18],gamma=0.5)
def build_data_augmentation(data_augmentation):
    if data_augmentation=="less":
        ret=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ])
        return ret
    else:
        ret=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ColorJitter(brightness=0.5,contrast=0.5),
                transforms.RandomRotation(degrees=65),
                transforms.RandomHorizontalFlip(0.2),
                transforms.RandomVerticalFlip(0.2),
                transforms.RandomGrayscale(0.2),
                transforms.ToTensor(),
            ])
        return ret


# In[6]:


# Define Model(For testing only, we will use the longer version)

def build_model_cnn(dropout,batch_size):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(            
                nn.Conv2d(3,20,3), #254
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.Conv2d(20,50,3,stride=2), #127
                nn.BatchNorm2d(50),
                nn.ReLU(),
                nn.Conv2d(50,100,4), #124
                nn.BatchNorm2d(100),
                nn.ReLU(),
                nn.Conv2d(100,200,3,stride=2),#62
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(200,300,3), #60
                nn.BatchNorm2d(300),
                nn.ReLU(),
                nn.Conv2d(300,512,3,stride=2), #29
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.feature_size=512*29*29
            self.linear = nn.Sequential(
                nn.Linear(self.feature_size,500),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(500,800),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(800,300),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(300,2),
            )
        def forward(self,x_train):
            out=self.conv(x_train)
            out=out.view(batch_size,self.feature_size)
            out=self.linear(out)
            return out
    model=CNN()
    return model


# In[7]:


#define training and testing function
def train_test_model(nepochs,LossFunction,model,optimizer,scheduler,device,train,test):
    acc_history = []
    loss_history = []
    for epoch in tqdm(range(nepochs),desc="Epoch",colour="#26B69C"):
        #training
        train_start=time.time()
        temp=[]
        for data,label in tqdm(train,colour="#26B69C",mininterval=20):
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = LossFunction(out,label)
            loss.backward()
            optimizer.step()
            temp.append(loss.item())
        mean_loss=sum(temp)/len(temp)
        loss_history.append(mean_loss)
        print(f"Epoch {epoch} Average Loss: {mean_loss}")
        if scheduler!=None:
            scheduler.step(mean_loss)
        train_end=time.time()
        print(f"Epoch {epoch} Train Time: {train_end-train_start}")
        #testing
        with torch.no_grad():
            test_start=time.time()
            correct=0
            total=0
            for data,label in tqdm(test,colour="#26B69C",mininterval=20):
                data = data.to(device)
                label=label.to(device)
                if data.shape==torch.Size([1, 3, 256, 256]):
                    out=model(data)
                    pred=torch.argmax(out)
                    if pred==label:
                        correct+=1
                    total+=1
            acc=round(correct/total,3)
            acc_history.append(acc)
            test_end=time.time()
            print(f"Epoch {epoch} Accuracy: {acc}")
            print(f"Epoch {epoch} Test Time: {test_end-test_start}")
    #graphs
    #train loss
    #plt.plot(loss_history)
    #plt.title("Loss")
    #plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    #plt.show()
    #accuracy
    #plt.plot(acc_history)
    #plt.title("Accuracy")
    #plt.xlabel("Epochs")
    #plt.ylabel("Accuracy")
    #plt.show()
    #wandb.log({"loss": mean_loss, "accuracy": acc})
    #wandb.log({"loss_graph": loss_graph, "acc_graph": acc_graph})
    return loss_history,acc_history


# In[20]:


#use wandb to find the best hyperparameters

def sweep_model(config=None):
    with wandb.init(config=config):
        config=wandb.config
        transform=build_data_augmentation(config.data_augmentation)

        batch_size=config.batch_size
        traindata=MaskDataset(trainlabel,traindir,transform)
        train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
        testdata=MaskDataset(testlabel,testdir,transform)
        test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)

        nepochs=config.epochs
        learning_rate=config.learning_rate
        weight_decay=config.weight_decay
        model=build_model_cnn(config.dropout,batch_size)
        model=model.to(device)
        optimizer=build_optimizer(config.optimizer,model,learning_rate,weight_decay)
        scheduler=build_lr_scheduler(config.lr_scheduler,optimizer)

        train_test_model(nepochs,LossFunction,model,optimizer,scheduler,device,train,test)


# In[8]:


loss_base=[0.41621328369728644, 0.30136923708988683, 0.2571114161587726, 0.22056669360243714, 0.19080633143972284, 0.16835732408838228, 0.13283714914328992, 0.11796278654654126, 0.09547339517112087, 0.08474352601911223] 
acc_base=[0.807, 0.795, 0.814, 0.799, 0.826, 0.831, 0.82, 0.833, 0.831, 0.856]


# In[9]:


#ploting loss and accuracy function
def plot_loss(array):
    for i in range(len(array)):
        loss_array = []
        for j in range(len(array[i][0])):
            t = j + 1
            loss_array.append(t)
        
        plt.plot(loss_array,array[i][0],label = array[i][2],linewidth=1)
        plt.title("Loss",fontsize = 25)
        plt.ylabel("loss",fontsize = 15)
        plt.xlabel("epoch", fontsize = 15)

        plt.tick_params(axis='both', labelsize = 15)
        plt.legend()
        plt.show
def plot_acc(array):
    for i in range(len(array)):
        acc_array = []
        for j in range(len(array[i][1])):
            t = j + 1
            acc_array.append(t)
        
        plt.plot(acc_array,array[i][1],label = array[i][2],linewidth=1)
        plt.title("Accuracy",fontsize = 25)
        plt.ylabel("accuracy",fontsize = 15)
        plt.xlabel("epoch", fontsize = 15)

        plt.tick_params(axis='both', labelsize = 15)
        plt.legend()
        plt.show


# In[21]:


#sweeping
wandb.agent(sweep_id,sweep_model,project="Mask")


# In[9]:


#comparing lr_scheduler vs no lr_scheduler
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=25
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
scheduler=build_lr_scheduler("ROP",optimizer)

loss_scheduler,acc_scheduler=train_test_model(nepochs,LossFunction,model,optimizer,scheduler,device,train,test)
loss_noscheduler,acc_noscheduler=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[14]:


#plot loss
array=[]
temp=[]
temp.append(loss_scheduler)
temp.append(acc_scheduler)
temp.append(["ReduceOnPlateau"])
array.append(temp)
temp=[]
temp.append(loss_noscheduler)
temp.append(acc_noscheduler)
temp.append(["NoScheduler"])
array.append(temp)
plot_loss(array)


# In[15]:


#plot accuracy
plot_acc(array)


# In[10]:


#Less Data Augmentation vs More Data Augmentation
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=10
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_less_aug,acc_less_aug=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)
loss_base,acc_base=loss_less_aug,acc_less_aug

transform=build_data_augmentation("more")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
loss_more_aug,acc_more_aug=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[12]:


array=[]
temp=[]
temp.append(loss_base)
temp.append(acc_base)
temp.append(["LessAugmentation"])
array.append(temp)
temp=[]
temp.append(loss_more_aug)
temp.append(acc_more_aug)
temp.append(["MoreAugmentation"])
array.append(temp)
plot_loss(array)


# In[13]:


plot_acc(array)


# In[16]:


#No Weight Decay vs Weight Decay
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=10
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_noweight,acc_noweight=loss_base,acc_base

weight_decay=0.3
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_weight,acc_weight=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[21]:


array=[]
temp=[]
temp.append(loss_base)
temp.append(acc_base)
temp.append(["NoWeightDecay"])
array.append(temp)
temp=[]
temp.append(loss_weight)
temp.append(acc_weight)
temp.append(["0.3WeightDecay"])
array.append(temp)
plot_loss(array)


# In[22]:


plot_acc(array)


# In[10]:


#Dropout Compare
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=10
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_drop03,acc_drop03=loss_base,acc_base


dropout=0
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
loss_nodrop,acc_nodrop=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[12]:


array=[]
temp=[]
temp.append(loss_nodrop)
temp.append(acc_nodrop)
temp.append(["NoDropOut"])
array.append(temp)
temp=[]
temp.append(loss_base)
temp.append(acc_base)
temp.append(["0.3DropOut"])
array.append(temp)
plot_loss(array)


# In[13]:


plot_acc(array)


# In[10]:


#Learning Rate Compare
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=10
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_lr0005,acc_lr0005=loss_base,acc_base

learning_rate=0.01
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_lr001,acc_lr001=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)

learning_rate=0.05
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_lr005,acc_lr005=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[11]:


array=[]
temp=[]
temp.append(loss_base)
temp.append(acc_base)
temp.append(["0.005LearningRate"])
array.append(temp)
temp=[]
temp.append(loss_lr001)
temp.append(acc_lr001)
temp.append(["0.01LearningRate"])
array.append(temp)
temp=[]
temp.append(loss_lr005)
temp.append(acc_lr005)
temp.append(["0.05LearningRate"])
array.append(temp)
plot_loss(array)


# In[12]:


plot_acc(array)


# In[10]:


#SGD vs Adam Optimizer
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=10
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_SGD,acc_SGD=loss_base,acc_base

optimizer=build_optimizer("Adam",model,learning_rate,weight_decay)
loss_Adam,acc_Adam=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[11]:


array=[]
temp=[]
temp.append(loss_base)
temp.append(acc_base)
temp.append(["SGD"])
array.append(temp)
temp=[]
temp.append(loss_Adam)
temp.append(acc_Adam)
temp.append(["Adam"])
array.append(temp)
plot_loss(array)


# In[12]:


plot_acc(array)


# In[15]:


#This is the custom model we USED.
def build_model_cnn_more(dropout,batch_size):
    class CNN_MORE(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(            
                nn.Conv2d(3,20,3), #254
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.Conv2d(20,30,3), #252
                nn.BatchNorm2d(30),
                nn.ReLU(),
                nn.Conv2d(30,50,3), #250
                nn.BatchNorm2d(50),
                nn.ReLU(),
                nn.Conv2d(50,110,4,stride=2),#124
                nn.BatchNorm2d(110),
                nn.ReLU(),
                nn.Conv2d(110,160,3), #122
                nn.BatchNorm2d(160),
                nn.ReLU(),
                nn.Conv2d(160,200,3), #120
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(200,250,3), #118
                nn.BatchNorm2d(250),
                nn.ReLU(),
                nn.Conv2d(250,300,4,stride=2), #58
                nn.BatchNorm2d(300),
                nn.ReLU(),
                nn.Conv2d(300,350,3), #56
                nn.BatchNorm2d(350),
                nn.ReLU(),
                nn.Conv2d(350,400,3), #54
                nn.BatchNorm2d(400),
                nn.ReLU(),
                nn.Conv2d(400,450,3), #52
                nn.BatchNorm2d(450),
                nn.ReLU(),
                nn.Conv2d(450,512,4,stride=2),#25
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.feature_size=512*25*25
            self.linear = nn.Sequential(
                nn.Linear(self.feature_size,500),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(500,800),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(800,300),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(300,2),
            )
        def forward(self,x_train):
            out=self.conv(x_train)
            out=out.view(batch_size,self.feature_size)
            out=self.linear(out)
            return out
    model=CNN_MORE()
    return model

transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=20
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_lesslayer,acc_lesslayer=loss_base,acc_base

model=build_model_cnn_more(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_morelayer,acc_morelayer=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[16]:


array=[]
temp=[]
temp.append(loss_morelayer)
temp.append(acc_morelayer)
temp.append(["CustomModel"])
array.append(temp)
plot_loss(array)


# In[17]:


plot_acc(array)


# In[11]:


#Initialized Weight vs No Initialized Weight
def build_model_cnn_init(dropout,batch_size):
    class CNN_INIT(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(            
                nn.Conv2d(3,20,3), #254
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.Conv2d(20,50,3,stride=2), #127
                nn.BatchNorm2d(50),
                nn.ReLU(),
                nn.Conv2d(50,100,4), #124
                nn.BatchNorm2d(100),
                nn.ReLU(),
                nn.Conv2d(100,200,3,stride=2),#62
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(200,300,3), #60
                nn.BatchNorm2d(300),
                nn.ReLU(),
                nn.Conv2d(300,512,3,stride=2), #29
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.feature_size=512*29*29
            self.linear = nn.Sequential(
                nn.Linear(self.feature_size,500),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(500,800),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(800,300),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(300,2),
            )
            #Initializing weights
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                 
                elif isinstance(m,nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                 
        def forward(self,x_train):
            out=self.conv(x_train)
            out=out.view(batch_size,self.feature_size)
            out=self.linear(out)
            return out
    model=CNN_INIT()
    return model

transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=10
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)


model=build_model_cnn_init(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_init,acc_init=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[14]:


array=[]
temp=[]
temp.append(loss_base)
temp.append(acc_base)
temp.append(["DefaultInit"])
array.append(temp)
temp=[]
temp.append(loss_init)
temp.append(acc_init)
temp.append(["KaimingUniformInit"])
array.append(temp)
plot_loss(array)


# In[15]:


plot_acc(array)


# In[34]:


#CNN with BottleNeck vs No bottleneck layers
def build_model_cnn_neck(dropout,batch_size):
    class CNN_NECK(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(            
                nn.Conv2d(3,20,3), #254
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.Conv2d(20,10,1),
                nn.ReLU(),
                nn.Conv2d(10,30,3), #252
                nn.BatchNorm2d(30),
                nn.ReLU(),
                nn.Conv2d(30,15,1),
                nn.ReLU(),
                nn.Conv2d(15,50,3), #250
                nn.BatchNorm2d(50),
                nn.ReLU(),
                nn.Conv2d(50,20,1),
                nn.ReLU(),
                nn.Conv2d(20,80,4,stride=2),#124
                nn.BatchNorm2d(80),
                nn.ReLU(),
                nn.Conv2d(80,50,1),
                nn.ReLU(),
                nn.Conv2d(50,110,3), #122
                nn.BatchNorm2d(110),
                nn.ReLU(),
                nn.Conv2d(110,50,1),
                nn.ReLU(),
                nn.Conv2d(50,160,3), #120
                nn.BatchNorm2d(160),
                nn.ReLU(),
                nn.Conv2d(160,70,1),
                nn.ReLU(),
                nn.Conv2d(70,200,3), #118
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(200,100,1),
                nn.ReLU(),
                nn.Conv2d(100,250,4,stride=2), #58
                nn.BatchNorm2d(250),
                nn.ReLU(),
                nn.Conv2d(250,150,1),
                nn.ReLU(),
                nn.Conv2d(150,300,3), #56
                nn.BatchNorm2d(300),
                nn.ReLU(),
                nn.Conv2d(300,150,1),
                nn.ReLU(),
                nn.Conv2d(150,350,3), #54
                nn.BatchNorm2d(350),
                nn.ReLU(),
                nn.Conv2d(350,170,1),
                nn.ReLU(),
                nn.Conv2d(170,450,3), #52
                nn.BatchNorm2d(450),
                nn.ReLU(),
                nn.Conv2d(450,200,1),
                nn.ReLU(),
                nn.Conv2d(200,512,4,stride=2), #25
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.feature_size=512*25*25
            self.linear = nn.Sequential(
                nn.Linear(self.feature_size,500),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(500,800),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(800,300),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(300,2),
            )
        def forward(self,x_train):
            out=self.conv(x_train)
            out=out.view(batch_size,self.feature_size)
            out=self.linear(out)
            return out
    model=CNN_NECK()
    return model

transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
nepochs=10
learning_rate=0.005
weight_decay=0
dropout=0.3
model=build_model_cnn(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_lesslayer,acc_lesslayer=loss_base,acc_base

model=build_model_cnn_neck(dropout,batch_size)
model=model.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_neck,acc_neck=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[10]:


#ResNet101 Transfer Learning (we also used this model)
nepochs=20
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)
learning_rate=0.005
weight_decay=0
dropout=0.3
res101 = timm.create_model('resnet101',pretrained=True)
#change the output channel of fully connected layers to 2
res101.fc=nn.Linear(2048,2)
model = res101.to(device)
optimizer=build_optimizer("SGD",model,learning_rate,weight_decay)
loss_res101,acc_res101=train_test_model(nepochs,LossFunction,model,optimizer,None,device,train,test)


# In[12]:


array=[]
temp=[]
temp.append(loss_res101)
temp.append(acc_res101)
temp.append(["ResNet101"])
array.append(temp)
plot_loss(array)


# In[13]:


plot_acc(array)


# In[9]:


#Naive Bayes model
transform=build_data_augmentation("less")
batch_size=1
traindata=MaskDataset(trainlabel,traindir,transform)
train=DataLoader(dataset=traindata,batch_size=batch_size,shuffle=True)
testdata=MaskDataset(testlabel,testdir,transform)
test=DataLoader(dataset=testdata,batch_size=batch_size,shuffle=True)

x_train=[]
y_train=[]
y_test=[]
correct=0
model=GaussianNB()
sdtr=StandardScaler()
#flatten the image features
for x,y in train:
    x=(x[0].view(1,-1)[0]).numpy()
    y=y[0].numpy()
    x_train.append(x)
    y_train.append(y) 
#x_train=sdtr.fit_transform(x_train)
#train the model
model.fit(x_train,y_train)
#testing
for data,label in test:
    if data.shape==torch.Size([1, 3, 256, 256]):
        data=(data[0].view(1,-1)).numpy()
        label=label[0].numpy()
        out=model.predict(data)
        if out==label:
            correct+=1
acc=round(correct/len(test),3)
print(f"Accuracy: {acc}")


# In[ ]:


#supported vector machine model
model=SVC()
model.fit(x_train,y_train)
correct=0
for data,label in test:
    if data.shape==torch.Size([1, 3, 256, 256]):
        data=(data[0].view(1,-1)).numpy()
        label=label[0].numpy()
        out=model.predict(data)
        if out==label:
            correct+=1
acc=round(correct/len(test),3)
print(f"Accuracy: {acc}")


# In[22]:


#logistic regression model
model=LogisticRegression(max_iter=99999999999)
model.fit(x_train,y_train)
correct=0
for data,label in test:
    if data.shape==torch.Size([1, 3, 256, 256]):
        data=(data[0].view(1,-1)).numpy()
        label=label[0].numpy()
        out=model.predict(data)
        if out[0]==label:
            correct+=1
acc=round(correct/len(test),3)
print(f"Accuracy: {acc}")


# In[20]:


#Save Model

#FILE="D:\\python\\CIS473\\CIS-473-Machine-Learning\\mask_data\\models\\model.pth"
#torch.save(model,FILE)


# In[ ]:




