# Created by Yujing on 18/9/3
# 1. Evaluate the accuracy of our Encode & Decode network.
# 2. Decode the deep feature for getting oriented point cloud.
import argparse
import tensorflow as tf
import json
import numpy as np
import os
import sys
import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'models'))
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model', default='log/model.ckpt', help='Model checkpoint path')
parser.add_argument('--output_dir', default='./test_results')
FLAGS = parser.parse_args()

# DEFAULT SETTINGS
pretrained_model_path = FLAGS.pretrained_model # os.path.join(BASE_DIR, './pretrained_model/model.ckpt')
output_dir = FLAGS.output_dir
gpu_to_use = 0

# MAIN SCRIPT
point_num = 2048            # the max number of points in the all testing data shapes
batch_size = 1

test_file_list = os.path.join(BASE_DIR, 'data/modelpose/test_files.txt')
TEST_FILES = provider.getDataFiles(test_file_list)
model = importlib.import_module('3_pointnet_using_ae') # import network module

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def load_pts_seg_files(pts_file, seg_file, catid):
    with open(pts_file, 'r') as f:
        pts_str = [item.rstrip() for item in f.readlines()]
        pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
    with open(seg_file, 'r') as f:
        part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
        seg = np.array([cpid2oid[catid+'_'+str(x)] for x in part_ids])
    return pts, seg

def pc_augment_to_point_num(pts, pn):
    assert(pts.shape[0] <= pn)
    cur_len = pts.shape[0]
    res = np.array(pts)
    while cur_len < pn:
        res = np.concatenate((res, pts))
        cur_len += pts.shape[0]
    return res[:pn, :]

def cal_dist(pred, norm):
	return np.mean(np.square(pred-norm))

def predict():
    is_training = False

    with tf.device('/gpu:'+str(gpu_to_use)):
        pointclouds_ph, labels_ph = model.placeholder_inputs(batch_size, point_num)
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = model.get_model(pointclouds_ph, is_training_ph, bn_decay=None)        

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        pc_out = os.path.join(output_dir, 'oriented_pc')
        if not os.path.exists(pc_out):
            os.mkdir(pc_out)

        flog = open(os.path.join(output_dir, 'log_test.txt'), 'w')

        # Restore variables from disk.
        printout(flog, 'Loading model %s' % pretrained_model_path)
        saver.restore(sess, pretrained_model_path)
        printout(flog, 'Model restored.')

        batch_data = np.zeros([batch_size, point_num, 3]).astype(np.float32)

        total_dist= 0.
        sum_dist = 0.

        ftest = open(test_file_list, 'r')
        lines = [line.rstrip() for line in ftest.readlines()]
        ftest.close()

        # Shuffle train files
        test_file_idxs = np.arange(0, len(TEST_FILES))
        np.random.shuffle(test_file_idxs)

        for fn in range(len(TEST_FILES)):
            
            printout(flog, '>>>>> File' + str(fn) + ': ' + TEST_FILES[test_file_idxs[fn]] + ' >>>>>')
            # Here we use old structure of data. We will propose a new kind of data structure.
            current_data, current_label, current_norm = provider.loadDataFile_with_angle(TEST_FILES[test_file_idxs[fn]])
            current_data = current_data[:,0:point_num,:]
            # ===
            gt_data = np.zeros([current_data.shape[0], 2048, 3])
            gt_data[:] = current_data[0]
            # ===

            file_size = current_data.shape[0]
            num_batches = file_size // batch_size
            file_dist = 0
            num = 0
        
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx+1) * batch_size

                feed_dict = {pointclouds_ph: current_data[start_idx:end_idx, :, :],
                             is_training_ph: is_training}
                pred_val= sess.run([pred], feed_dict=feed_dict)
                # batch_dist = cal_dist(pred_val, current_norm[start_idx:end_idx, ...])
                batch_dist = cal_dist(pred_val, gt_data[start_idx:end_idx, ...])

                printout(flog, 'PC {0} distance: {1}'.format(bacth_idx, batch_dist))
                file_dist += bacth_dist
                total_dist += bacth_dist

                # Restore the oriented model.
                printout(flog, 'Store predicted point cloud into nrom_pc{0}.obj ...'.format(num))
                f_pcout = open(os.path.join(pc_out, 'norm_pc{0}.obj'.format(num)), 'w')
               	for point_idx in range(pred_val[0]):
               		f_pcout.write('v ' + pred_val[0][point_idx][0] + ' ' \
               			   	   	   	   + pred_val[0][point_idx][1] + ' ' \
               			   	   	   	   + pred_val[0][point_idx][2] + '\n')
                f_pcout.close()
                num += 1

            file_mean_dist = file_loss/num_batches
            printout(flog, 'File mean distance: {0}', file_mean_dist)
            sum_dist += file_mean_dist
            
        printout(flog, '===== Test mean distance: {0} =====\n'.format(sum_dist / len(TEST_FILES)))

with tf.Graph().as_default():
	predict()





