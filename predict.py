#Name: Sultan Bamukhaier

#For changing the directory : cd aipnd-project


#Predicing some images from test folder:

#Code: python predict.py --image 'flowers/test/13/image_05761.jpg' --checkpoint 'checkpoint.pth'
#The correct Answer: 'king protea'

#Code: python predict.py --image 'flowers/test/28/image_05270.jpg' --checkpoint 'checkpoint.pth'
#The correct Answer: 'stemless gentian'

#Code: python predict.py --image 'flowers/test/29/image_04145.jpg' --checkpoint 'checkpoint.pth'
#The correct Answer: 'artichoke'

#Code: python predict.py --image 'flowers/test/22/image_05391.jpg' --checkpoint 'checkpoint.pth'
#The correct Answer: 'pincushion flower'


import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train import load_model
import json
import argparse


#Defining command line arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Path to image to predict')
parser.add_argument('--checkpoint', type=str, help='File containing my trained module to use for predicting')
parser.add_argument('--topk', type=int, help='Returning topk predictions')
parser.add_argument('--labels', type=str, help='JSON file for mapping labels with flower names')
args, _ = parser.parse_known_args()


#Defining the prediction function:
def predict(image='flowers/test/13/image_05761.jpg', checkpoint='checkpoint.pth', topk=5, labels='cat_to_name.json'):

#Assigning variables to the inputs:
    if args.image:
        image = args.image     
    if args.checkpoint:
        checkpoint = args.checkpoint
    if args.topk:
        topk = args.topk            
    if args.labels:
        labels = args.labels


#Loading the saved checkpoints from checkpoint_path:
    checkpoint = torch.load(checkpoint)
 
    arch = checkpoint['arch']
              
    hidden_units = checkpoint['hidden_units']
    
    out_features = checkpoint['out_features']

#Building my Network:
    model = load_model(arch=arch, out_features=out_features, hidden_units=hidden_units)       

#Mapping the classes with indices:
    model.class_to_idx = checkpoint['class_to_idx']

#Loading the saved model weights:
    state_dict = checkpoint['state_dict']
         
    model.load_state_dict(state_dict)  
                 
#JSON file for mapping the labels with flower names:
    with open(labels, 'r') as f:
        cat_to_name = json.load(f)
        
#Loading the image:
    pil_image = Image.open(image)

#Defining the image_transforms: Resizing, CenterCropping, Then converting to Tensor:
    image_transform=transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        ])
    
#Applying the image_transforms, output type is Tensor:    
    pil_image = image_transform(pil_image).float()

#Converting the image from Tensor to Numpy:
    np_image = np.array(pil_image)    
    

#subtract the means from each color channel, then divide by the standard deviation:    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std 
    
#PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
#reorder dimensions using ndarray.transpose. The color channel needs to be first and retain the order of the other two dimensions

#Reordering the dimension, it will start with the third dimension (Color Chanel). Then first & second:
    np_image = np.transpose(np_image, (2, 0, 1))
            

        
#Converting the input from numpy to tensor
    image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
    
#Converting the tensor type to torch.FloatTensor
    image_tensor = image_tensor.type(torch.FloatTensor)

    
    model_input = image_tensor.unsqueeze(0)
    
#Sending the input to cuda
        
    model.to('cuda')
        
    model_input = model_input.to('cuda')

#Evaluation mode:
    model.eval()
    
#Forward step in my model:
    probs = torch.exp(model.forward(model_input))

#Saving the top (topk=5) predictions in top_probs & The indicies for the top (topk=5) predictions in top_idicies:
    top_probs,top_idicies = probs.topk(topk)
    
#Convering the results from tensor to Lists:
    top_probs = top_probs.detach().cpu().numpy().tolist()[0]
    top_idicies = top_idicies.detach().cpu().numpy().tolist()[0]
    
#Making a dict such as:{idicies:labels}
    idx_to_class = dict(zip(model.class_to_idx.values(),model.class_to_idx.keys()))
    
#Making a list matching the indices in top_indices with the actual labels (folder names):    
    top_labels = [idx_to_class[idx] for idx in top_idicies]
    
#Mapping labels with flower names:
    top_flowers = [cat_to_name[lable] for lable in top_labels]


    return top_probs,top_labels,top_flowers
    
    
    

#Checking if image and checkpoint are available:
if args.image and args.checkpoint:
    
    print('Selected image',args.image)
    print('Loading the model from',args.checkpoint)

#Start the predicting process:
    top_probs,top_labels,top_flowers = predict(args.image, args.checkpoint)

#Print the results:
print('Top k probabilities :',top_probs)
print('Top flower names :',top_flowers)