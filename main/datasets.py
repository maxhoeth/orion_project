from torch.utils.data import Dataset
from utils import read_binvox, get_orientation_label, get_labels, read_off
import numpy as np
import os
import re

class VoxelDataset(Dataset):
    
    def __init__(self, root_dir, folder="train", p=0 , rotated_file='D:/NNDL/rotated_modelnet10/Modelnet10/'):
        self.root_dir = root_dir
        self.rotated_file = rotated_file
        folders = get_labels(root_dir)
        self.p = p
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.files = []
        for i in folders:
            path = root_dir + i + '/' + folder
            for file in os.listdir(path):
                if file.endswith('binvox'):
                    sample ={}
                    sample['path'] = path + '/' + file
                    sample['label'] = i
                    self.files.append(sample)

    
    def __rotate__(self, path):
        split_path = re.split(r'[./]+', path)
        if np.random.random() < self.p:
            for i, x in enumerate(os.listdir(self.rotated_file + split_path[-4] + '/' + split_path[-3])):
                if (split_path[-2] in x) and (x.endswith('.binvox')):
                    return self.rotated_file + split_path[-4] + '/' + split_path[-3] + '/' + x, 1
            return path, 0
        else:
            return path, 0
            
        
    def __get_classes__(self):
        return self.classes
    
    def __get_orientation_label__(self, label, rot_angle):
        orientation_label = get_orientation_label(label, rot_angle)
        return orientation_label
    
    def __len__(self):
        return len(self.files)
    
    def __shape__(self):
        path = self.files[1]['path']
        return self.__preproc__(path, 0)[0].shape

    
    def __preproc__(self, file, rotated):
        voxels, rot_angle = read_binvox(file, rotated)
        
        return voxels, rot_angle

    def __getitem__(self, idx):
        path = self.files[idx]['path']
        label = self.files[idx]['label']
        path, rotated = self.__rotate__(path)
        sample, rot_angle = self.__preproc__(path, rotated)
        orientation_label = self.__get_orientation_label__(label, rot_angle)
        return sample, self.classes[label], orientation_label

    
class PointNetDataset(Dataset):
    
    def __init__(self, root_dir='D:/NNDL/modelnet10_pointnet/ModelNet10/', folder="train",  npoints=1024):
        self.root_dir = root_dir
        self.npoints = npoints
        folders = get_labels(root_dir)
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.files = []
        for i in folders:
            path = root_dir + i + '/' + folder
            for file in os.listdir(path):
                if file.endswith('off'):
                    sample ={}
                    sample['path'] = path + '/' + file
                    sample['label'] = i
                    self.files.append(sample)

    
        
    def __get_classes__(self):
        return self.classes

    def __len__(self):
        return len(self.files)
    
    def __shape__(self):
        path = self.files[1]['path']
        return self.__preproc__(path).shape

    
    def __preproc__(self, file):
        points = read_off(file, self.npoints)
        points = points - np.expand_dims(np.mean(points, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
        points = points/ dist
        return points

    def __getitem__(self, idx):
        path = self.files[idx]['path']
        label = self.files[idx]['label']
        sample = self.__preproc__(path)
        return sample, self.classes[label]
