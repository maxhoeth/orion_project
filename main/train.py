from datasets import *
from models import *
from utils import get_default_path, train_base, train_orion, train_pointnet
import torch
import argparse
import os
from torch.utils.data import DataLoader
import time
import gc

torch.cuda.empty_cache()
gc.collect()

parser = argparse.ArgumentParser(
                    prog='Train',
                    description='Specify model you want to train')


parser.add_argument('--model', type=str, default = 'base', help='*base*, *orion* or *pointnet*')
parser.add_argument('--weights', type=str, default = None, help='File for pretrained weights')
parser.add_argument('--filename', type=str, default=None, help= 'path for base dataset')
parser.add_argument('--EPOCHS', type=int, default = 10, help='number of epochs to train')
parser.add_argument('--BATCHSIZE', type=int, default = 64, help='batch size')
parser.add_argument('--lr', type=float, default = 1e-3, help='learning rate')
parser.add_argument('--save-weights', type=int, default = 10, help='every n-step saves the wights')
parser.add_argument('--p-train', type=float, default = 0.02, help='prob of rotated sampels in training set ONLY FOR ORION')
parser.add_argument('--p-test', type=float, default = 0.02, help='prob of rotated sampels in test set ONLY FOR ORION')
parser.add_argument('--n-points', type=int, default = 1024, help='number of random points ONLY FOR POINTNET')
parser.add_argument('--rotated-filename', type=str, default='D:/NNDL/rotated_modelnet10/ModelNet10/' ,help= 'path for rotated dataset')
parser.add_argument('--timeit', type=str, default='yes' ,help= 'if traininf should be timed')
parser.add_argument('--evaluate', type=str, default='no' ,help= 'if it should be evaluated after every epoch')
opt = parser.parse_args()

if not os.path.isdir('weights'):
    os.mkdir('weights')

model_type = opt.model
weights = opt.weights
epochs = opt.EPOCHS
batch_size = opt.BATCHSIZE
lr = opt.lr
p_train = opt.p_train
p_test = opt.p_test
n_points = opt.n_points
save_weigths = opt.save_weights
file = opt.filename
rotated_file = opt.rotated_filename
timeit = opt.timeit
evaluate = opt.evaluate

if not file:
    file = get_default_path(model_type)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)



if model_type == 'base':
    

    if evaluate.lower() == 'yes':

        train_ds = VoxelDataset(file, folder='train', rotated_file=rotated_file)
        test_ds = VoxelDataset(file, folder='test', rotated_file=rotated_file)
    
        print('Train dataset size: ', len(train_ds))
    
        train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_load = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    
    else:

        train_ds = VoxelDataset(file, folder='train', rotated_file=rotated_file)    
        print('Train dataset size: ', len(train_ds))
    
        train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_load = None

    
    model = Base()
    if weights is not None:
        state_dict = torch.load(weights)['model_state_dict']
        model.load_state_dict(state_dict)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    model.to(device)    

    if timeit.lower() == 'yes':
        time_start = time.time()

    train_base(model, optimizer, save_weigths, train_load, test_load, epochs)   
    
    if timeit.lower() == 'yes':
        print('Average Time per Epoch: ', (time.time() - time_start)/epochs)

elif model_type == 'orion':
    
    if evaluate.lower() == 'yes':

        train_ds = VoxelDataset(file, folder='train', p=p_train, rotated_file=rotated_file)
        test_ds = VoxelDataset(file, folder='test', p=p_test, rotated_file=rotated_file)
        
        print('Train dataset size: ', len(train_ds))

        train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_load = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    

    else:

        train_ds = VoxelDataset(file, folder='train', p=p_train, rotated_file=rotated_file)        
        print('Train dataset size: ', len(train_ds))

        train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_load = None

    model = Orion()
    if weights is not None:
        state_dict = torch.load(weights)['model_state_dict']
        try:      
            model.load_state_dict(state_dict)
        except:
            model.set_state_dict(state_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)   
    
    if timeit.lower() == 'yes':
        time_start = time.time()

    train_orion(model, optimizer, save_weigths, train_load, test_load, epochs)        
    
    if timeit.lower() == 'yes':
        print('Average Time per Epoch: ', (time.time() - time_start)/epochs)
        

elif model_type == 'pointnet':
    
    if evaluate.lower() == 'yes':

        train_ds = PointNetDataset(file, folder='train', npoints=n_points)
        test_ds = PointNetDataset(file, folder='test', npoints=n_points)
    
        train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_load = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    else:

        train_ds = PointNetDataset(file, folder='train', npoints=n_points)
    
        train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_load = None
    
    
    model = PointNet()
    if weights is not None:
        state_dict = torch.load(weights)['model_state_dict']
        model.load_state_dict(state_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)     

    if timeit.lower() == 'yes':
        time_start = time.time()

    train_pointnet(model, optimizer, save_weigths, train_load, test_load, epochs)
    
    if timeit.lower() == 'yes':
        print('Average Time per Epoch: ', (time.time() - time_start)/epochs)

else:
    raise ValueError('model parmeter must be *base*, *orion* or *pointnet*')