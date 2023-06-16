import os
import numpy as np


def get_labels(main_path):
    subdirs = [x for x in os.listdir(main_path)]
    labels = [x for x in subdirs if '.' not in x]
    return labels


def get_filenames(path):
    files = os.listdir(path)
    paths = [path + i for i in files if i.endswith('.off')]
            
    return paths, len(paths)   


def pre_pre_process(main_path):
    labels = get_labels(main_path)
    filenames = []
    path = main_path
    for i in labels:
        paths, _ = get_filenames(path + i + '/train/')
        paths2, _ = get_filenames(path + i + '/test/')
        filenames += paths + paths2
    
    for s in filenames:
        if s.endswith('.off'):
            f = open(s, 'r')
            first = f.readline()
            f.seek(0)
            if first != 'OFF\n':
                with open(s) as f:
                    lines = f.readlines()
                    x = lines[0]
                    new_line = [x[0:3] + '\n' + x[3:]]
                    old_lines = lines[1:]
                    new = new_line + old_lines
                with open(s, 'w') as f:
                    f.writelines(new)
        else:
            continue
            
def random_rot(points):
        
    rotations = np.arange(0, 360, 30)
    rand_int = np.random.randint(12)
    random_angle = rotations[rand_int]
    rot_mat = np.array([[np.cos(random_angle), -np.sin(random_angle), 0],
                        [np.sin(random_angle), np.cos(random_angle), 0],
                        [0, 0, 1]])
    points = np.dot(points, rot_mat)
            
def re_write_off(label, s, rot_angle, file, points, faces, new_path):
    root = new_path
    new_file = file[:-4] + f'_{int(rot_angle)}' + '.off'
    rot_file = root + label + '/' + s + '/' + new_file
    try:
        if os.path.isfile(root + label + '/' + s + '/' + new_file):
            f = open(rot_file, 'r')
            with open(rot_file, 'w') as f:
                f.writelines('OFF\n')
                f.writelines(str(len(points)) + ' ' + str(len(faces)) + ' ' + str(0) + '\n')
                for i in points:
                    f.writelines(re.sub(',', '', str(list(i))[1:-1]) + '\n')
                for i in faces:
                    f.writelines(re.sub(',', '', str(list(i))[1:-1]) + '\n')
        else:    
            f = open(rot_file, 'x')
            with open(rot_file, 'w') as f:
                f.writelines('OFF\n')
                f.writelines(str(len(points)) + ' ' + str(len(faces)) + ' ' + str(0) + '\n')
                for i in points:
                    f.writelines(re.sub(',', '', str(list(i))[1:-1]) + '\n')
                for i in faces:
                    f.writelines(re.sub(',', '', str(list(i))[1:-1]) + '\n')
                
                
    except FileNotFoundError:
        if not os.path.isdir(root + label):
            os.mkdir(root + label)
        os.mkdir(root + label + '/' + s)
        re_write_off(label, s, rot_angle, file, points, faces)
                
        
def create_rand_rotated_dataset(main_path, new_path):
    
    orientation_classes = {'bed' : 12, 'monitor': 12, 'desk': 12, 'chair': 12, 'dresser': 12, 'toilet': 12, 'sofa': 12, 'table': 3, 'night_stand': 12, 'bathtub': 6 }

    for i in orientation_classes.keys():
        for s in ['test', 'train']:
            paths = os.listdir(main_path + i + '/'+ s)
            for l in paths:
                if ('DS' in l) or l.endswith('.binvox'):
                    continue

                points, faces = read_file(main_path + i + '/' + s + '/' + l)
                rot_points, rotation = random_rot(points)
                re_write_off(i, s, rotation, l, rot_points, faces, new_path)
                
def create_binvox(main_path):
    
    labels = get_labels(main_path)
    path = main_path

    filenames = []
    for i in labels:
        paths, _ = get_filenames(path + i + '/train/')
        paths2, _ = get_filenames(path + i + '/test/')
        filenames += paths + paths2
    
    for i in filenames:
        if os.path.isfile(i[:-3] + 'binvox'):
            print('File already exists!')
            continue
        else:
            os.system(f'binvox -cb -pb -e -c -d 28 {i}')