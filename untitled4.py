# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:03:16 2020

@author: 張濟
"""


import csv
import os
import shutil

from torchvision import models
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
import torchvision
from torch.optim import lr_scheduler
from PIL import Image

import time
#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Hyperparametes
BATCH_SIZE = 64
LR = 0.1  #0.05 for resnet
EPOCH = 200

def train_model(model, criterion, optimizer, scheduler, num_epochs, train_loader, valid_loader):
    model = model.train()
    for epoch in range(num_epochs):
        training_corrects = 0
        valid_corrects = 0
        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            output = model(bx)
            _, preds = torch.max(output, 1)
            loss = criterion(output, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_corrects += torch.sum(preds == by.data)
        
        with torch.no_grad():
            model = model.eval()
            for (inputs, labels) in valid_loader: 
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                _, preds = torch.max(output, 1)
                valid_corrects += torch.sum(preds == labels.data)
            model = model.train()    
        print("train correct : %d"%training_corrects)   #all 8942
        print("validation correct : %d"%valid_corrects) #all 2237
        scheduler.step()
        #print("accuracy : %d"%epoch_acc)
    return model

def predict_carbrand(model, classes, img, imgid, writer):
    imgid = imgid.split('.')[0]
    model = model.eval()
    with torch.no_grad():
        prob = model(img.to(device)) 
        v,l = torch.max(prob,1)
        pred_r = classes[l.data.item()]
        writer.writerow([imgid, pred_r])
def copy_img(mode, imgclass_index, pathold):
    pathnew = ""
    if mode == "train":
        if imgclass_index<10 and imgclass_index>=0:
            pathnew = "./training_data_p/00" + str(imgclass_index)
        elif imgclass_index<100 and imgclass_index>=10:
            pathnew = "./training_data_p/0" + str(imgclass_index)
        else:
            pathnew = "./training_data_p/%d"%(imgclass_index)
        #if not os.path.exists(pathnew):
        #    shutil.copy(pathold, pathnew)
    else:
        if imgclass_index<10 and imgclass_index>=0:
            pathnew = "./valid_set/00" + str(imgclass_index)
        elif imgclass_index<100 and imgclass_index>=10:
            pathnew = "./valid_set/0" + str(imgclass_index)
        else:
            pathnew = "./valid_set/%d"%(imgclass_index)
        #if not os.path.exists(pathnew):
        #    shutil.copy(pathold, pathnew)   
    return pathnew
def main():
    numofimg = 11185
    with open("training_labels.csv", newline="") as csvfile:
        rows = csv.reader(csvfile)
        i = 0
        classes = []
        for row in rows:
            if i == 0:
                i+=1
                continue
            classname = str(row[1])
            if i-1<10 and i-1>=0:
                path_t = "./training_data_p/00" + str(i-1)
                path_v = "./valid_set/00" + str(i-1)
            elif i-1<100 and i-1>=10:
                path_t = "./training_data_p/0" + str(i-1)
                path_v = "./valid_set/0" + str(i-1)
            else:
                path_t = "./training_data_p/%d"%(i-1)
                path_v = "./valid_set/%d"%(i-1)
            if not classname in classes:
                classes.append(classname)
                if not os.path.exists(path_t):
                    os.mkdir(path_t)
                    os.mkdir(path_v)
                    i+=1
    
    with open("training_labels.csv", newline="") as csvfile:
        i = 0
        rows = csv.reader(csvfile)
        for row in rows:
            if i == 0:
                i+=1
                continue
            imgid = str(row[0])
            pathold = "./training_data/" + imgid + ".jpg"
            imgclass = str(row[1])
            imgclass_index = classes.index(imgclass)
            if i <= numofimg*0.8:
                pathnew = copy_img("train", imgclass_index, pathold)
            else:
                pathnew = copy_img("valid", imgclass_index, pathold)

            if not os.path.exists(pathnew):
                shutil.copy(pathold, pathnew)
            i+=1
    data_transform = {
        # train use data augmentation
        'train':
            torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.RandomRotation(degrees=30),
                torchvision.transforms.RandomCrop(size=(224,224)),
                #(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
                #torchvision.transforms.ColorJitter(),
                #torchvision.transforms.RandomHorizontalFlip(),
                #torchvision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.85,1), ratio=(0.9, 1.1)),
                
                #torchvision.transforms.Resize((224, 224)),
                #torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ]),
        #validation does not need augmentation
        'valid':
            torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
    }
        
    train_data = ImageFolder(
        root = os.path.join("training_data_p"),
        transform = data_transform['train']
    )
    train_loader = Data.DataLoader(
        dataset = train_data,
        batch_size = BATCH_SIZE,
        shuffle = True,
        #num_workers = 2
    )
    valid_data = ImageFolder(
        root = os.path.join("valid_set"),
        transform = data_transform['valid']
    )
    valid_loader = Data.DataLoader(
        dataset = valid_data,
        batch_size = BATCH_SIZE,
        shuffle = True,
        #num_workers = 2
    )
    
    print(type(train_data))
    
    print("num of training data : %d" %(len(train_data)))
    print("num of validation data : %d" %(len(valid_data)))
    print("epoch num:%d"%EPOCH)
    #print(train_data.class_to_idx)
    model = models.densenet161(pretrained=True)
    #model = models.resnet50(pretrained = True)
    #freeze the parameters so that gradiants can't compute in backward()
    #print(model)
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 20:    #(6)for resnset50
            for param in child.parameters():
                param.requires_grad = False
    """
    for param in model.parameters():
        param.requires_grad = False
    """
    #Parameters of newly constructed modules have requires_grad=True by default
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 196)
    
    #for densenet
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 196)
    
    
    """
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.Dropout(0.4),
        nn.ReLU(), 
        nn.Linear(1024, 196)
        )
    """
    #print(model)
    model = model.to(device)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    #optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    loss_func = nn.CrossEntropyLoss()   
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)
    ts = time.time()
    model = train_model(model, loss_func, optimizer, step_lr_scheduler, EPOCH, train_loader, valid_loader)
    te = time.time()
    print("time : %d" %(te-ts))
    #test img transform
    """
    img = img = Image.open("./testing_data/000093.jpg")
    img.show()
    img_t = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                            torchvision.transforms.RandomCrop(size=(224,224)),
                                            #(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
                                            #torchvision.transforms.ToTensor()
                                            ])
    img = img_t(img)
    print(type(img))
    #print(img.size())
    img.show()
    """
    
    #several cases
    imgs = os.listdir("./testing_data")
    with open("output.csv","w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])
        i = 0
        for img in imgs:
            imgid = img
            img = Image.open("./testing_data/"+img)
            try:
                img = data_transform['valid'](img)
            except:
                print("%s set into RGB format"%imgid)
                img = data_transform['valid'](img.convert('RGB'))
            img = torch.unsqueeze(img,0)
            #print(type(img))
            #print(img.shape)
            predict_carbrand(model, classes, img, imgid, writer)
         
            
if __name__ == "__main__":
    main()
