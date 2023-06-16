import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

# Base VoxNet
class Base(nn.Module):
    
    def __init__(self, classes=10):
        super().__init__()
        self.cv3_1 = nn.Conv3d(1, 32, 5)
        self.cv3_2 = nn.Conv3d(32, 32, 3)
        self.relu = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.4)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch1 = nn.BatchNorm3d(32)
        self.batch2 = nn.BatchNorm3d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(42592 , 128)
        self.fc2 = nn.Linear(128, classes)
        
    def forward(self, x):
        x = self.cv3_1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.cv3_2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.drop2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop3(x)
        out_label = self.fc2(x)
    
        
        return out_label
    
# Orion VoxNet    
class Orion(nn.Module):
    
    def __init__(self, classes=10):
        super().__init__()
        self.cv3_1 = nn.Conv3d(1, 32, 5)
        self.cv3_2 = nn.Conv3d(32, 32, 3)
        self.relu = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.4)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch1 = nn.BatchNorm3d(32)
        self.batch2 = nn.BatchNorm3d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(42592 , 128)
        self.fc2 = nn.Linear(128, classes)
        self.fc3 = nn.Linear(128, 105)
        
        
    def set_state_dict(self, state_dict):
        state_dict['fc3.weight'] = nn.init.xavier_uniform_(self.fc3.weight)
        state_dict['fc3.bias'] = self.fc3.bias.data.zero_()
        self.load_state_dict(state_dict)

        
        
    def forward(self, x):
        x = self.cv3_1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.cv3_2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.drop2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop3(x)
        out_label = self.fc2(x)
        out_orientation = self.fc3(x)
    
        
        return out_label, out_orientation


#Pointnet
class TNet(nn.Module):
    
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(self.k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k*self.k)
        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.nn_layers = nn.ModuleList()

        
    def forward(self, x):
        batchsize = x.size()[0]
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x

    
class PointNet(nn.Module):
    
    def __init__(self, feature_transform=True):
        super(PointNet, self).__init__()
        self.tnet = TNet()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.tnetk = TNet(k=64)
            
        #Classification
        self.cls_fc1 = nn.Linear(1024, 512)
        self.cls_fc2 = nn.Linear(512, 256)
        self.cls_fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
        self.cls_bn1 = nn.BatchNorm1d(512)
        self.cls_bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.tnet(x.transpose(2, 1))
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        
        if self.feature_transform:
            trans_feature = self.tnetk(x)
            x = torch.bmm(x.transpose(2, 1), trans_feature)
            x = x.transpose(2, 1)
            
        pointfeat = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        #Classification
        x = self.relu(self.cls_bn1(self.dropout(self.cls_fc1(x))))
        x = self.relu(self.cls_bn2(self.dropout(self.cls_fc2(x))))
        x = self.cls_fc3(x)
        
        return x
