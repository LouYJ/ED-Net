# Created by Yujing on 18/9/5
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import random
import json

SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_sampling/build/pcsample')

SAMPLING_POINT_NUM = 2048
SAMPLING_LEAF_SIZE = 0.005

MODELNET40_PATH = '../datasets/modelnet40'
def export_ply(pc, filename):
	vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	for i in range(pc.shape[0]):
		vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
	ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
	ply_out.write(filename)

# Sample points on the obj shape
def get_sampling_command(obj_filename, ply_filename):
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d ' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f ' % SAMPLING_LEAF_SIZE
    return cmd

# --------------------------------------------------------------
# Following are the helper functions to load MODELNET40 shapes
# --------------------------------------------------------------

# Read in the list of categories in MODELNET40
def get_category_names():
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

# Return all the filepaths for the shapes in MODELNET40 
def get_obj_filenames():
    obj_filelist_file = os.path.join(MODELNET40_PATH, 'filelist.txt')
    obj_filenames = [os.path.join(MODELNET40_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    print('Got %d obj files in modelnet40.' % len(obj_filenames))
    return obj_filenames

# Helper function to create the father folder and all subdir folders if not exist
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal, 
		data_dtype='float32', label_dtype='uint8', noral_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

# Read numpy array data and label from h5_filename
def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Make up rows for Nxk array
# Input Pad is 'edge' or 'constant'
def pad_arr_rows(arr, row, pad='edge'):
    assert(len(arr.shape) == 2)
    assert(arr.shape[0] <= row)
    assert(pad == 'edge' or pad == 'constant')
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'constant', (0, 0))

# Yujing: Transform my data to h5 file
def save_h5_seg(data, label, pid, h5_filename, data_dtype='float32', label_dtype='uint8', pid_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'pid', data=pid,
            compression='gzip', compression_opts=1,
            dtype=pid_dtype)
    h5_fout.close()

def make_h5(model_path, seg_path, label_path, h5_filename, data_dtype='float32', label_dtype='uint8', pid_dtype='uint8'):
    model_lists = os.listdir(model_path)
    seg_lists = os.listdir(seg_path)

    model_lists.sort()
    seg_lists.sort()
    
    model_h5 = {}
    model_h5['data'] = []
    model_h5['label'] = []
    model_h5['pid'] = []

    fn2label = {}
    label_file = open(label_path)
    for line in label_file:
        split_line = line.split('    ')
        fn2label[split_line[1][:-2]] = int(split_line[0])
    label_file.close()

    for i in range(len(model_lists)):
        model_file = open(os.path.join(model_path,model_lists[i]))
        seg_file = open(os.path.join(seg_path,seg_lists[i]))

        models = model_file.readlines()
        points = []
        for p in models:
            p = p.split()
            point = [float(p[1]), float(p[2]), float(p[3])]
            points.append(point)

        seg = []
        for line in seg_file:
            seg.append(int(line[0]))

        model_h5['data'].append(points)
        model_h5['label'].append(fn2label[model_lists[i]])
        model_h5['pid'].append(seg)

        model_file.close()
        seg_file.close()

    save_h5_seg(np.array(model_h5['data']),
                np.array(model_h5['label']).reshape((-1,1)),
                np.array(model_h5['pid']),
                h5_filename)


def divide_data(h5file, ratio=0.7):
    data = h5py.File(h5file)
    length = len(data['data'])

    indices = random.sample(range(length), int(length*0.7))
    indices.sort()

    train_set = {}
    test_set = {}

    train_set['data'] = data['data'][indices, :, :]
    train_set['label'] = data['label'][indices, :]
    train_set['pid'] = data['pid'][indices, :]

    test_set['data'] = []
    test_set['label'] = []
    test_set['pid'] = []

    for i in range(length):
        if i not in indices:
            test_set['data'].append(data['data'][i, :, :])
            test_set['label'].append(data['label'][i, :])
            test_set['pid'].append(data['pid'][i, :])

    save_h5_seg(train_set['data'],
                train_set['label'],
                train_set['pid'],
                'train_obj.h5')

    save_h5_seg(np.array(test_set['data']),
                np.array(test_set['label']).reshape((-1,1)),
                np.array(test_set['pid']),
                'test_obj.h5')

def delete_cat(categories, data_file):
    data = h5py.File(data_file)
    new_data = {}
    indices = []
    for i in range(len(data['data'])):
        if data['label'][i] not in categories:
            indices.append(i)
        else:
            print data['label'][i]

    new_data['data'] = data['data'][indices, :, :]
    new_data['label'] = data['label'][indices, :]
    new_data['pid'] = data['pid'][indices, :]
    
    save_h5_seg(new_data['data'],
                new_data['label'],
                new_data['pid'],
                'new_' + data_file)

def rotate_point_cloud_h5(out_dir, h5_file, rand=1, degree=0):
    data = h5py.File(h5_file)
    pc = data['data']

    if not (os.path.exists(out_dir)):
    	os.mkdir(out_dir)

    rotated_data_x = rotate_point_cloud(pc, 'x', rand, degree)
    rotated_data_y = rotate_point_cloud(pc, 'y', rand, degree)
    rotated_data_z = rotate_point_cloud(pc, 'z', rand, degree)

    save_h5_seg(rotated_data_x,
                data['label'],
                data['pid'],
                os.path.join(out_dir, h5_file[:-3]+'_x'+str(degree)+'.h5'))
    print ('Write ' + h5_file[:-3]+'_x'+str(degree)+'.h5' + ' successfully!')
    save_h5_seg(rotated_data_y,
                data['label'],
                data['pid'],
                os.path.join(out_dir, h5_file[:-3]+'_y'+str(degree)+'.h5'))
    print ('Write ' + h5_file[:-3]+'_y'+str(degree)+'.h5' + ' successfully!')
    save_h5_seg(rotated_data_z,
                data['label'],
                data['pid'],
                os.path.join(out_dir, h5_file[:-3]+'_z'+str(degree)+'.h5'))
    print ('Write ' + h5_file[:-3]+'_z'+str(degree)+'.h5' + ' successfully!')

# 19/9/5 Yujing: Modify the rotation function to realize xyz random rotation.
def rotate_point_cloud(batch_data, axis='z', rand=1, degree=0):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        shape_pc = batch_data[k, ...]
        center = np.mean(shape_pc, axis=0)
        shape_moved = shape_pc - center

        print('Center of shape_moved: 'np.mean(shape_moved, axis=0))

        if (rand == 1):
            rotation_angle_x = np.random.uniform() * 2 * np.pi
            rotation_angle_y = np.random.uniform() * 2 * np.pi
            rotation_angle_z = np.random.uniform() * 2 * np.pi
            matrix_x = np.array([[1, 0, 0],
                                [0, np.cos(rotation_angle_x), -np.sin(rotation_angle_x)],
                                [0, np.sin(rotation_angle_x), np.cos(rotation_angle_x)]])

            matrix_y = np.array([[np.cos(rotation_angle_y), 0, np.sin(rotation_angle_y)],
                                [0, 1, 0]
                                [-np.sin(rotation_angle_y), 0, np.cos(rotation_angle_y)]])

            matrix_z = np.array([[np.cos(rotation_angle_z), -np.sin(rotation_angle_z), 0],
                                [np.sin(rotation_angle_z), np.cos(rotation_angle_z), 0], 
                                [0, 0, 1]])

            shape_moved_rotated = np.dot(np.dot(np.dot(shape_moved, matrix_x.T), matrix_y.T), matrix_z.T)
            print('Center of shape_moved_rotated: ', np.mean(shape_moved_rotated, axis=0))
            shape_rotated = shape_moved_rotated + center

        else:
            rotation_angle = degree/180.0*np.pi

            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)

            if (axis == 'x'):
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, cosval, -sinval],
                                            [0, sinval, cosval]])
            elif (axis == 'y'):
                rotation_matrix = np.array([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]])
            else:
                rotation_matrix = np.array([[cosval, -sinval, 0],
                                            [sinval, cosval, 0],
                                            [0, 0, 1]])        
            shape_rotated = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, ...] = shape_rotated
    return rotated_data


def h5_to_obj(h5_file, id2cat_file):
    file = h5py.File(h5_file)
    models = file['data']
    labels = file['label']

    f = open(id2cat_file, 'r')
    id2name = []
    id2cat = []
    for line in f:
        sp = line.split('\t')
        id2name.append(sp[0])
        id2cat.append(sp[1][:-1])
    f.close()

    len_models = len(models)
    with open('testing_file_yujing.txt', 'a') as test_f:
        if not os.path.exists('objs/'):
            os.makedirs('objs/')
        for model_idx in range(len_models):
            tmp_name = './objs/'+h5_file[9:-3]+'_'+ ('%04d'%model_idx) +'_'+('%02d'%labels[model_idx])+'_'+id2name[labels[model_idx][0]]+'.pts'
            with open(tmp_name, 'w') as pts_f:
                pts = models[model_idx]
                for point in pts:
                    pts_f.write(str(point[0])+' '+str(point[1])+' '+str(point[2])+'\n')
                    
            test_f.write(tmp_name + ' ' + id2cat[labels[model_idx][0]] + '\n')


def divide_pts_label(file):
    pts = []
    label = []
    with open(file, 'r') as pl:
        for line in pl:
            sp = line.split()
            pts.append(sp[:6])
            label.append(sp[6])

    num = len(pts)
    pts_f = open(os.path.join(file[:10], 'points/' + file[11:-4] + '.pts'), 'w')
    label_f = open(os.path.join(file[:10], 'points_label/' + file[11:-4] + '.seg'), 'w')

    for i in range(num):
        for j in range(5):
            pts_f.write(pts[i][j] + ' ')
        pts_f.write(pts[i][5] + '\n')
        label_f.write(label[i][0] + '\n')

    pts_f.close()
    label_f.close()

def divide_all_pl_files(base_dir):
    category = []
    synset = []
    with open('synsetoffset2category.txt') as cat_f:
        for line in cat_f:
            tmp = line.split('\t')
            # print tmp
            category.append(tmp[0])
            synset.append(tmp[1][:-1])

    for s in synset:
        # print (s + '------------------------------')
        path = os.path.join(base_dir, s)
        # print path
        if not os.path.exists(os.path.join(path, 'points')):
            os.makedirs(os.path.join(path, 'points'))
        if not os.path.exists(os.path.join(path, 'points_label')):
            os.makedirs(os.path.join(path, 'points_label'))

        model_list = os.listdir(path)
        for model in model_list:
            # print (os.path.join(path, model))
            if not (model=='points' or model=='points_label'):
                divide_pts_label(os.path.join(path, model))

def load_json_file(path):
    with open(path) as json_file:
        jsondata = json.load(json_file)
        return jsondata

def overallid_to_catid_partid(json_file, seg_path, output_dir):
    match = load_json_file(json_file)
    seg_list = os.listdir(seg_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name in seg_list:
        seg_file = os.path.join(seg_path, name)
        output_path = os.path.join(output_dir, name)

        outf = open(output_path, 'w')
        segf = open(seg_file, 'r')

        for line in segf:
            oaid = int(line.rstrip())
            pid = match[oaid][1]
            outf.write(str(pid) + '\r\n')

        outf.close()
        segf.close()


def create_file_list(base_dir, out_dir):
    category = []
    synset = []
    with open(os.path.join(base_dir, 'synsetoffset2category.txt'), 'r') as cat_f:
        for line in cat_f:
            tmp = line.split('\t')
            category.append(tmp[0])
            synset.append(tmp[1][:-1])

    cat = {}
    with open(os.path.join(base_dir, 'label.txt'), 'r') as cf:
        for line in cf:
            tmp = line.split('    ')
            cat[tmp[1].rstrip()] = int(tmp[0])

    pts_list = os.listdir(os.path.join(base_dir, 'points'))
    pid_list = os.listdir(os.path.join(base_dir, 'points_label'))
    pts_list.sort()
    pid_list.sort()

    of = open(out_dir, 'w')
    list_len = len(pts_list)
    for i in range(list_len):
        name = pts_list[i][:-4]
        of.write(os.path.join('points', pts_list[i]) + ' ')
        of.write(os.path.join('points_label', pid_list[i]) + ' ')
        of.write(synset[cat[name]] + '\n')
    of.close()

def ply_to_obj(plyfile, objfile):
    plydata = PlyData.read(plyfile)
    pc = plydata['vertex'].data
    with open(objfile, 'w') as fout:
        for pt in pc:
            fout.write(str(pt[0]))
            fout.write(' ' + str(pt[1]))
	    fout.write(' ' + str(pt[2]))
            fout.write('\n')

def transform_all_ply(plypath, outpath, filetype, filelist=False):
    if not (os.path.exists(outpath)):
        os.makedirs(outpath)
    cat_list = os.listdir(plypath)
    
    for cat in cat_list:
        f = open('file_list_'+cat+'.txt', 'w')
        print ('*----- Start catgory: %s -----*' % cat)
        cat_path = os.path.join(plypath, cat)
        cat_path_obj = os.path.join(outpath, cat)
        if not (os.path.exists(cat_path_obj)):
            os.makedirs(cat_path_obj)
        ply_list = os.listdir(cat_path)
        for ply in ply_list:
            ply_to_obj(os.path.join(cat_path, ply), os.path.join(cat_path_obj, ply[:-3] + filetype))
            f.write(ply[:-3] + filetype + ' ' + cat + '\n')

        f.close()
        print ('Finish %s!' % cat)



if __name__ == '__main__':
    #transform_all_ply('shapenet_dim32_sdf_pc', 'shapenet_dim32_sdf_pc_pts', 'pts')
    # create_file_list('TestFile', 'testing_file_yujing.txt')

    # overallid_to_catid_partid('overallid_to_catid_partid.json', 'seg_2048_overall', 'seg_2048_partid')

    # divide_all_pl_files('.')

    '''
    nf = open('new.txt', 'w')
    with open('testing_file_yujing.txt', 'r') as f:
        for line in f:
            tmp = line[7:]
            nf.write(tmp)

    nf.close()
    '''

    '''
    file_list = os.listdir('.')
    for file in file_list:
        if '.h5' in file:
            h5_to_obj(file, 'all_object_categories.txt')
    '''

    # make_h5('./model_2048', './seg_2048', './label.txt', 'origin.h5')
    # divide_data('./origin.h5')

    '''
    file_list = os.listdir('.')
    for file in file_list:
        if ('.h5' in file) and (len(file) == 17 or len(file) == 16):
            # delete_cat([4, 15], file)
            print ('----------Start '+file+'----------')
            rotate_point_cloud_h5('rotated_data', file, rand=0, degree=30)
            rotate_point_cloud_h5('rotated_data', file, rand=0, degree=60)
            rotate_point_cloud_h5('rotated_data', file, rand=0, degree=90)
	    print ('----------Finish '+file+'----------')
    '''
    








