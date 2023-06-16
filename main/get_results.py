import argparse
from utils import make_results, get_default_path
from datasets import *
from models import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
                    prog='Train',
                    description='Specify model you want to train')


parser.add_argument('--model', type=str, default = 'base', help='*base*, *orion* or *pointnet*')
parser.add_argument('--weightfile', type=str, default=None ,help= 'path for weights')
parser.add_argument('--BATCHSIZE', type=int, default=64 ,help= 'batch size')
parser.add_argument('--filename', type=str, default=None ,help= 'file of dataset')
parser.add_argument('--name', type=str, default=None ,help= 'name for plots')
opt = parser.parse_args()

modeltype = opt.model
file = opt.weightfile
batch_size = opt.BATCHSIZE
data = opt.filename
name = opt.name

if not name:
    name = modeltype

if not data:
    data = get_default_path(parser.parse_args().model)

if modeltype == 'orion':
    model = Orion()
    if file is not None:
        state_dict = torch.load(file)['model_state_dict']
        try:
            model.load_state_dict(state_dict)
        except:
            model.set_state_dict(state_dict)
    test_ds = VoxelDataset(data, folder='test')
    test_load = DataLoader(test_ds, batch_size=batch_size, shuffle=True) 
       
elif modeltype == 'base':
    model = Base()
    if file is not None:
        state_dict = torch.load(file)['model_state_dict']
        model.load_state_dict(state_dict)
        
    test_ds = VoxelDataset(data, folder='test')
    test_load = DataLoader(test_ds, batch_size=batch_size, shuffle=True)    
        
elif modeltype == 'pointnet':
    model = PointNet()
    if file is not None:
        state_dict = torch.load(file)['model_state_dict']
        model.load_state_dict(state_dict)
    
    test_ds = PointNetDataset(data, folder='test')
    test_load = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    
else:
    raise ValueError('Mode must be *orion*, *base* or *pointnet*')
    
make_results(model, test_load, name, data, modeltype)