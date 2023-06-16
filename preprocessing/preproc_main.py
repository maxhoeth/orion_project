import preproc_utils as ppu
import argparse

parser = argparse.ArgumentParser(
                    prog='Preprocess',
                    description='Creates Binvox from Modelnet and creates randomly roateted Binvox based on orientation classes')
                    
parser.add_argument('--filename', type=str, default = 'D:/NNDL/modelnet10/ModelNet10/', help='dataset path')
parser.add_argument('--rotated-filename', type=str, default='D:/NNDL/rotated_modelnet10/ModelNet10/' ,help= 'path for rotated dataset')
opt = parser.parse_args()

main_path = opt.filename
rotated_main_path = opt.rotated_filename

if (rotated_main_path is not None):
    ppu.pre_pre_process(main_path)
    ppu.create_rand_rotated_dataset(main_path, rotated_main_path)
    ppu.create_binvox(main_path)
    ppu.create_binvox(rotated_main_path)
else: 
    ppu.pre_pre_process(main_path)
    ppu.create_binvox(main_path)
    
print('Done!')