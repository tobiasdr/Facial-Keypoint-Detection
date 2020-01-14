## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        #input image: 1x224x224 (grayscale squared)
        
        self.conv1 = nn.Conv2d(1, 32, 5) #1x224x224 to 32x220x220
        self.pool1 = nn.MaxPool2d(2, 2) #32x220x220 to 32x110x110
        self.drop1 = nn.Dropout(p=0.1)
            
        self.conv2 = nn.Conv2d(32, 64, 5) #32x110x110 to 64x106x106
        self.pool2 = nn.MaxPool2d(2, 2) #64x106x106 to 64x53x53
        self.drop2 = nn.Dropout(p=0.1)
      
        self.conv3 = nn.Conv2d(64, 128, 4) #64x53x53 to 128x50x50
        self.pool3 = nn.MaxPool2d(2, 2) #128x50x50 to 128x25x25
        self.drop3 = nn.Dropout(p=0.1)
        
        self.lin1 = nn.Linear(128*25*25, 1000)
        
        # dropout with p=0.4
        self.drop4 = nn.Dropout(p=0.1)
        
        # finally, create 10 output channels (for the 10 classes)
        self.lin2 = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = x
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1) #flatten
        
        x = self.lin1(x)
        x = self.lin2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
