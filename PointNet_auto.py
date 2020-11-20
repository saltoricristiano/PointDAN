import os
import torch.nn.functional as F
import torch.nn as nn


class PointNet_AutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,256,1)
        self.conv4 = nn.Conv1d(256,512,1)
        
        self.bn1 = nn.BatchNorm1d(64)   
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        ###############

        self.fc1 = nn.Linear(512, 512) 
        self.fc2 = nn.Linear(512, 512)   
        self.fc3 = nn.Linear(512, num_points*3)

        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(512)
    
    def forward(self, input):
        batchsize, dim, npoints = input.shape
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = self.bn4(self.conv4(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)  

        ######################
        embedding = nn.Flatten(1)(xb)  
        #can also be written as (xb.view(-1, 1024))
        ######################
          
        xb = F.relu(self.bn5(self.fc1(embedding)))
        xb = F.relu(self.bn6(self.fc2(xb)))
        output = self.fc3(xb)
        output = output.view(batchsize, dim, npoints)
        return  output, embedding


