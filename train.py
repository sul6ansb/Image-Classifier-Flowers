#Name: Sultan Bamukhaier

#For changing the directory : cd aipnd-project

#Testing the file on flowers datasets: python train.py --data_dir flowers --epochs 1 --checkpoint 'checkpoint.pth'

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import math
import copy
import argparse
from argparse import ArgumentParser


#Defining command line arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Dataset path')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=list, help='Enter a list of two hidden unit integer numbers such as: [4096,100]')
parser.add_argument('--checkpoint', type=str, help='Checkpoint file used to save the trained model')
args, _ = parser.parse_known_args()


#Function to build my model:
def load_model(arch='vgg19', out_features=102, hidden_units=[4096,1000]):
     
#Loading the pre-trained module for the selected architecture:
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch ,
                         'please select arch : vgg16 or vgg19 or alexnet') 
    

#Initializing the input to my classifier depending on the chosen mode:
    if arch == 'vgg19' or arch == 'vgg16':
        in_features = 25088 
    elif arch == 'alexnet':
        in_features = 9216

#Mapping variables:
    hidden_layers = hidden_units
   
#Freezing the loaded model parameters:
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                             ('fc1', nn.Linear(in_features, hidden_layers[0])),
                              ('relu1', nn.ReLU()),
    
                              ('drop1', nn.Dropout(p=0.2)),
    
                              ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                              ('relu2', nn.ReLU()),
    
                              ('drop1', nn.Dropout(p=0.2)),
        
                              ('fc3', nn.Linear(hidden_layers[1], out_features)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    
    return model
    

#My training function:
def train_model(image_datasets, arch='vgg19', hidden_units=[4096,1000], epochs=10, learning_rate=0.0005, checkpoint='checkpoint.pth'):
    
    
#Assigning variables to the inputs:

    if args.arch:
        arch = args.arch     
        
    if args.hidden_units:
        hidden_units = args.hidden_units
    if type(hidden_units) != list:
        raise ValueError('hidden units must be a list of two integers', hidden_units)

    if args.epochs:
        epochs = args.epochs
            
    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.checkpoint:
        checkpoint = args.checkpoint        
        

        
#Sizez of my data_loader batches:
    train_batch_size=32
    valid_batch_size=32
    test_batch_size=32

#Making a dictionary for my dataloaders:
    dataloaders = {'trainloader': torch.utils.data.DataLoader(image_datasets['train'], batch_size=train_batch_size, shuffle=True),
                   'validloader': torch.utils.data.DataLoader(image_datasets['valid'], batch_size= valid_batch_size)      ,
                   'testloader' : torch.utils.data.DataLoader(image_datasets['test'] , batch_size= test_batch_size )      }
        

#Mapping from dictionary to variables:
    train_data = image_datasets['train']
    valid_data = image_datasets['valid']
    test_data  = image_datasets['test']
 
    trainloader = dataloaders['trainloader']
    validloader = dataloaders['validloader']
    testloader  = dataloaders['testloader']
 

        
    print('Network Architecture:', arch)
    print('List of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    
    
#The output of my classifier (number of classes)
    out_features = len(train_data.classes)
    
#Builing my module:
    model = load_model(arch=arch, out_features=out_features, hidden_units=hidden_units)
    
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)
    
#Using cuda if available. Otherwise: using cpu
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print('please activate the GPU for faster running')

    model.to(device)      

    
#Will be used to copy the weight of the hiest accuracy:
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    
#Setting the starting time, will be used later to calculate the running time:
    start = time.time()

#Assigning the number of loaders entering the training function (Loading & Validating):
    loader_types= [trainloader , validloader]


    print('--- Started the training ---')

#Starting the training loops:
    for e in range(epochs):    


#Will use the loader_types selected (trainloader for training & validloader for validating)
        for loader_type in loader_types:
    
    #Initializing for the training mode:
            if loader_type == trainloader:
        
                model.train()        
                train_running_loss = 0
                steps = 0
        
#Initializing for the validating mode:
            elif loader_type == validloader:
        
                model.eval()
                valid_running_loss = 0
                correct = 0
                total = 0
                steps = 0

#Extracting the inputs & labels from images in trainloader & validloader, quantities depend on number of batches:
            for inputs, labels in loader_type:
            
#Increment the step for counting:
                steps += 1
#Sending inputs & labels to the device (Usually cuda)
                inputs, labels = inputs.to(device), labels.to(device)
#Erasing the gradient datas:    
                optimizer.zero_grad()
#forward step:            
                outputs = model.forward(inputs)
#Using criterion to calculate the loss:           
                loss = criterion(outputs, labels)
#Only in training mode, Do training & update the weights:            
                if loader_type == trainloader:  
                    loss.backward()            
                    optimizer.step()            
                    train_running_loss += loss.item()
                
#When the training mode number of batches are completed:
                    if steps % (math.ceil(len(train_data)/train_batch_size)) == 0:
#Printing the training epoch result:
                        print("Training Loss in Epoch: {}/{} = {:.4f} ".format(e+1, epochs,
                                                        train_running_loss/len(train_data)*train_batch_size))               

#Only in evaluation mode, do the following:
                elif loader_type == validloader:
#Summing the losses and save them in valid_running_loss:       
                    valid_running_loss += loss.item()
            
 #After the forward move, do predicting to each input, find the maximum prediction value, if matches the labels save as 1, otherwise save as 0:           
                    _, predicted = torch.max(outputs.data, 1)
#Adding the number of inputs in the current batch to the variable total:
                    total += labels.size(0)
#Summing the total number of correct prediction in the given batch:
                    correct += (predicted == labels).sum().item()
#When the validating mode number of batches are completed:   
                    if steps % (math.ceil(len(valid_data)/valid_batch_size)) == 0:
#Printing the validating epoch result:               
                        print("Validation Loss in Epoch: {}/{} = {:.4f} ".format(e+1, epochs,
                                                        valid_running_loss/len(valid_data)*valid_batch_size))                          
#Calculate the percentage of the correct prediction in the epoch:
        epoch_acc = (100 * correct / total) 
        print("Accuracy of the network on the {%d} valid images: %d %% "%(len(valid_data) , epoch_acc))
        
#Checking if the last model has the best weights, if yes, save in best_model_wts:
        if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
#Printing the total running time:      
    print(f"DEVICE = {device}; Total Time: {(time.time() - start):.3f} seconds")
#Printing the best accuracy:
    print("The Best Accuracy Was: %d %%" %best_acc) 
    
#Loading the best model weights:
    model.load_state_dict(best_model_wts)

#Training has been completed.
    
#Mapping the classes with indices:
    model.class_to_idx = train_data.class_to_idx
    
    if checkpoint:
        print ('Saving checkpoint to:', checkpoint) 
        
#Creating a dictionary of the checkpoints:
        checkpoints= {
            'arch': arch,
            'class_to_idx': model.class_to_idx,
            'hidden_units': hidden_units,
            'out_features': out_features,
            'state_dict': model.state_dict(),
        }
        
        torch.save(checkpoints, checkpoint)
        

    
    return model





#Images datasets:
if args.data_dir:    
#Processing the images:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    }
    
#Images datasets:
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }
        
        
    train_model(image_datasets) 

    
    
    