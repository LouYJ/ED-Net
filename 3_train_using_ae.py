import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import time

# SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', type=str, default='3_pointnet_using_ae', help='Model name: pointnet_using_ae [default: pointnet_using_ae]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
FLAGS = parser.parse_args()

# SCRIPT
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
LOG_DIR = FLAGS.log_dir
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
OUTPUT_DIR = FLAGS.output_dir

MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
# os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure

MAX_NUM_POINT = 2048
# NUM_CLASSES = 3
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
# HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelpose/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelpose/val_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY, batch*BATCH_SIZE, BN_DECAY_DECAY_STEP, BN_DECAY_DECAY_RATE, staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bacth_op = tf.summary.scalar('bacth_number', batch)
            bn_decay = get_bn_decay(batch)
            bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            learning_rate_op = tf.summary.scalar('learning_rate', learning_rate)

            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Get model and loss 
            pred = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            pred = tf.squeeze(pred, axis=None, name=None)
            loss = MODEL.get_loss(pred, labels_pl)
            
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl, 'labels_pl': labels_pl,
               'is_training_pl': is_training_pl, 'pred': pred, 'loss': loss, 'train_op': train_op, 'merged': merged, 'step': batch}

        log_string('--------------------------------')
        log_string('----- Batch Size: {0}'.format(BATCH_SIZE))
        log_string('----- Point Number: {0}'.format(NUM_POINT))
        log_string('----- Traing using GPU: {0}'.format(GPU_INDEX))
        log_string('--------------------------------\n')
        for epoch in range(MAX_EPOCH):
            log_string('************ EPOCH %03d ************' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer, epoch)
            eval_one_epoch(sess, ops, test_writer)
            log_string('************ EPOCH %03d Finished ************' % (epoch))
            
            # Save the variables to disk.
            if (epoch+1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_epoch_" + str(epoch+1) + ".ckpt"))
                log_string("Store model successfully into file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    time_start=time.time()
    epoch_loss = 0
    
    for fn in range(len(TRAIN_FILES)):
        log_string('>>>>> File' + str(fn) + ': ' + TRAIN_FILES[train_file_idxs[fn]] + ' >>>>>')
        # 18/9/5 Yujing: Temporarily neglect the angle.
        current_data, current_label, _ = provider.loadDataFile_with_angle(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        gt_data = np.zeros([current_data.shape[0], 2048, 3])
        gt_data[:] = current_data[0]
        # current_data, current_label, current_angle = provider.shuffle_data_angle(current_data, np.squeeze(current_label), current_angle)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        log_string('Batch number: %d' % num_batches)
        
        loss_sum_file = 0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, ...],
                         ops['labels_pl']: gt_data[start_idx:end_idx, ...],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            loss_sum_file += loss_val
        
        mean_loss = loss_sum_file / float(num_batches)
        log_string('Mean loss: {0}'.format(mean_loss))
        log_string('<<<<< File{0}'.format(fn) + ' of epoch {0}/{1} <<<<<\n'.format(epoch, MAX_EPOCH))
        epoch_loss += mean_loss

    time_end=time.time()
    log_string('Traing one epoch use time: {0}s'.format(time_end-time_start))
    log_string('The mean loss of this epoch: {0}\n'.format(epoch_loss/len(TRAIN_FILES)))

        
def eval_one_epoch(sess, ops, test_writer, epoch_num):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Shuffle train files
    test_file_idxs = np.arange(0, len(TEST_FILES))
    np.random.shuffle(test_file_idxs)
    loss_sum = 0
    
    for fn in range(len(TEST_FILES)):
        log_string('===== === Evaluation for epoch {0} === =====\n'.format(epoch_num))
        log_string('>>>>> File' + str(fn) + ': ' + TRAIN_FILES[train_file_idxs[fn]] + ' >>>>>')
        current_data, current_label, current_norm = provider.loadDataFile_with_norm(TEST_FILES[test_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        file_loss = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_norm[start_idx:end_idx, ...],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            file_loss += loss_val
        loss_sum += file_loss
        log_string('Mean loss of file {0}: {1}'.format(fn, file_loss/num_batches))
            
    log_string('<<<<< Evaluation mean loss: {0} <<<<<\n'.format(loss_sum / len(TEST_FILES)))
         


if __name__ == "__main__":

    

    train()
    LOG_FOUT.close()
