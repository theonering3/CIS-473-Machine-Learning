#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Various Imports

import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[14]:


# Preprocessing Data 

for i in tqdm(range(0, 1), desc ="Pre-processing Data", colour="#35AF92"):
    data=pd.read_csv("D:/chrome download/credit data/application_train.csv",skiprows=1).dropna()
    data=pd.get_dummies(data)
    sdtr=StandardScaler()
    x_train,x_test,y_train,y_test=train_test_split(data.iloc[1:1000,2:],data.iloc[1:1000,1],test_size=0.2,random_state=0)
        
    x_train=sdtr.fit_transform(x_train)
    x_test=sdtr.transform(x_test)

    y_train=y_train.to_numpy()
    y_test=y_test.to_numpy()

    x_train=torch.from_numpy(x_train.astype(np.float32))
    y_train=torch.from_numpy(y_train.astype(np.float32))
    x_test=torch.from_numpy(x_test.astype(np.float32))
    y_test=torch.from_numpy(y_test.astype(np.float32))

    y_train=y_train.view(y_train.shape[0],1)
    y_test=y_test.view(y_test.shape[0],1)


# In[15]:


# Defining Dataset

class CreditTrainDataset(Dataset):
    def __init__(self,x_train,y_train):   
        self.x_train=x_train
        self.y_train=y_train
        
    def __getitem__(self,index):
        return self.x_train[index],self.y_train[index]
    
    def __len__(self):
        return self.x_train.shape[0]

    
class CreditTestDataset(Dataset):
    def __init__(self,x_test,y_test):
        self.x_test=x_test
        self.y_test=y_test
        
    def __getitem__(self,index):
        return self.x_test[index],self.y_test[index]
    
    def __len__(self):
        return self.x_test.shape[0]
    


# In[16]:


# Loading Data

for i in tqdm(range(0, 1), desc ="Loading Data", colour="#26B69C"):
    train=CreditTrainDataset(x_train,y_train)
    train_data=DataLoader(dataset=train,batch_size=5)
    test=CreditTestDataset(x_test,y_test)
    test_data=DataLoader(dataset=test,batch_size=2)


# In[23]:


# Defining Model

class CreditNet(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.hidden_size=500
        self.input_size=input_size
        self.output_size=output_size
        self.linear1=nn.Linear(self.input_size,self.hidden_size)
        self.linear2=nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,x_train):
        out=self.linear1(x_train)
        out=F.relu(out)
        out=self.linear2(out)
        out=torch.sigmoid(out)
        return out


# In[31]:


# Model Parameters

device=torch.device("cuda")
learning_rate=0.5
nepochs=10
input_size=x_train.shape[1]
output_size=2

model=CreditNet(input_size,output_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
LossFunction = nn.MSELoss()


# In[33]:


#Traning Model

loss_history = []
for epoch in tqdm(range(nepochs),desc="Epoch",colour="#26B69C"):
    for (data,label) in tqdm(train,desc="iter",colour="#26B69C"):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        out = model(data)
        loss = LossFunction(out, label)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    print(f"Epoch {epoch}: loss: {loss.item()}")


# In[ ]:




