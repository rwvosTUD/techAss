import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import transforms, models
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


#%% given functions

'''Small function to easily show an image given its path'''

def load_img(path: str):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[...,::-1]

def list_full_paths(directory: str):
    '''
    Function to get the full path of all files in a directory instead of
    the name of the file only. 

    Parameters:
    directory: the location of the files interested in
    '''
    return [os.path.join(directory, file) for file in os.listdir(directory)]


class RealOrFakeDataset(Dataset):
    '''
    Dataset class for the real and fake faces
    '''

    def __init__(self, img_dir: str, version: str, transform=None,):
        '''

        Parameters:
          img_dir: folder where all images are located
          version: train, validation or test
          transform: transformations to perform to the image (cropping, flipping)

        '''

        self.img_dir = img_dir
        self.version = version 
        self.transform = transform
        self.real_path = os.listdir(img_dir + f'{version}/real/')
        self.fake_path = os.listdir(img_dir + f'{version}/fake/')

        self.images = list_full_paths(img_dir + 
                        f'{version}/real/') + list_full_paths(img_dir + 
                        f'{version}/fake/') 

        self.resize = transforms.Resize([224, 224])
        self.to_tensor = transforms.ToTensor()

    def read_image(self, img_path: str):
        '''
          Function to an image from its path, perform transforms/resizes and
          convert it to a tensor. 

          Paramters:
            img_path: a string where the image is located

          Returns:
            tensor_img: transformed image in tensor format 
        '''
        img = Image.open(img_path).convert('RGB')

        img = self.resize(img)

        tensor_img = self.to_tensor(img)
        #tensor_img = tensor_img.unsqueeze(0) # todo activate again?

        return tensor_img 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = f'{self.images[idx]}'
        image = self.read_image(img_path)
        if "real" in img_path:
            label = 1.
        else:
            label = 0. 

        if self.transform:
            image = self.transform(image)

        return image, label


#%% Training/testing related

def compute_metrics(true, pred):
    threshold = 0.47058823529411764 # training set threshold
    
    correct = (pred == true).sum().item()
    CM =confusion_matrix(true.cpu(), pred.cpu(),labels=[0,1])        
    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0] 
    
    ''' todo remove
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    F1 = (2*recall*precision)/(recall+precision)
    #'''
    
    # collect
    metrics = np.array([[correct, tn, tp, fp, fn]])

    return metrics
    
    

def train_epoch(device, train_loader, net, optimizer, criterion, threshold):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """
  
    avg_loss = 0
    correct = 0
    total = 0
    metrics = np.zeros((1,5))
    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # keep track of metrics
        avg_loss += loss
        predicted =  torch.where(outputs.data > threshold, 1., 0.)
        metrics += compute_metrics(labels, predicted)
        total += labels.size(0)
        #correct += (predicted == labels).sum().item()

    # collect metrics
    metrics_epoch = {}
    metrics_epoch["accuracy"] = metrics[0,0]/total
    metrics_epoch["tn"] =  metrics[0,1]
    metrics_epoch["tp"] =  metrics[0,2]
    metrics_epoch["fp"] =  metrics[0,3]
    metrics_epoch["fn"] =  metrics[0,4]
    metrics_epoch["precision"] = metrics_epoch["tp"]/(metrics_epoch["tp"]+metrics_epoch["fp"])
    metrics_epoch["recall"] = metrics_epoch["tp"]/(metrics_epoch["tp"]+metrics_epoch["fn"]) 
    metrics_epoch["F1"] = (2*metrics_epoch["recall"]*metrics_epoch["precision"])/(metrics_epoch["recall"]+metrics_epoch["precision"])*100
    
    return avg_loss/len(train_loader), metrics_epoch
        

def test_epoch(device, test_loader, net, criterion, threshold):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0
    metrics = np.zeros((1,5))
    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())

            # keep track of metrics
            avg_loss += loss
            predicted =  torch.where(outputs.data > threshold, 1., 0.)
            metrics += compute_metrics(labels, predicted)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
    
        # collect metrics
        metrics_epoch = {}
        #metrics_epoch["metrics"] = metrics
        metrics_epoch["accuracy"] = metrics[0,0]/total
        metrics_epoch["tn"] =  metrics[0,1]
        metrics_epoch["tp"] =  metrics[0,2]
        metrics_epoch["fp"] =  metrics[0,3]
        metrics_epoch["fn"] =  metrics[0,4]
        metrics_epoch["precision"] = metrics_epoch["tp"]/(metrics_epoch["tp"]+metrics_epoch["fp"])
        metrics_epoch["recall"] = metrics_epoch["tp"]/(metrics_epoch["tp"]+metrics_epoch["fn"]) 
        metrics_epoch["F1"] = (2*metrics_epoch["recall"]*metrics_epoch["precision"])/(metrics_epoch["recall"]+metrics_epoch["precision"])*100
    

    return avg_loss/len(test_loader), metrics_epoch



