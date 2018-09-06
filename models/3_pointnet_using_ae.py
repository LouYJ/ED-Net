# Created by Yujing on 18/8/27
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import layers
# from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ input is BxNx3, output BxNx3 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    dropout_keep_prob = tf.where(is_training, 0.2, 1.0)
    # Conv Block1
    conv1_1 = layers.conv_btn(input_image, [1, 3], 64, 'conv1_1', is_training=is_training, padding='VALID')
    conv1_2 = layers.conv_btn(conv1_1, [1, 1], 256, 'conv1_2', is_training=is_training, padding='VALID')
    conv_reshape = tf.reshape(conv1_2, [batch_size, num_point, 256, 1])
    print ('ProBlock: ', conv_reshape.shape) # 2048, 256, 1
    pool1   = layers.maxpool(conv_reshape, [8, 1],  'pool1')
    print ('Block1: ', pool1.shape) # 256, 256, 1

    # Encode start
    # Conv Block 2
    conv2_1 = layers.conv_btn(pool1,   [3, 3], 32, 'conv2_1', is_training = is_training, padding='SAME')
    conv2_2 = layers.conv_btn(conv2_1, [3, 3], 32, 'conv2_2', is_training = is_training, padding='SAME')
    pool2   = layers.maxpool(conv2_2, [2, 2],   'pool2')
    print ('Block2: ', pool2.shape) # 128, 128, 32

    # Conv Block 3
    conv3_1 = layers.conv_btn(pool2,   [3, 3], 64, 'conv3_1', is_training = is_training, padding='SAME')
    conv3_2 = layers.conv_btn(conv3_1, [3, 3], 64, 'conv3_2', is_training = is_training, padding='SAME')
    pool3   = layers.maxpool(conv3_2, [2, 2],   'pool3')
    print ('Block3: ', pool3.shape) # 64, 64, 64

    # Conv Block 4
    conv4_1 = layers.conv_btn(pool3,   [3, 3], 128, 'conv4_1', is_training = is_training, padding='SAME')
    conv4_2 = layers.conv_btn(conv4_1, [3, 3], 128, 'conv4_2', is_training = is_training, padding='SAME')
    conv4_3 = layers.conv_btn(conv4_2, [3, 3], 128, 'conv4_3', is_training = is_training, padding='SAME')
    pool4   = layers.maxpool(conv4_3, [2, 2],   'pool4')
    print ('Block4: ', pool4.shape) # 32, 32, 128

    # FC Block
    fc5 = layers.conv_btn(pool4, [17, 17], 256, 'fc5', is_training = is_training, padding='VALID')
    drop5 = layers.dropout(fc5, dropout_keep_prob, 'drop5')
    print ('fc5: ', drop5.shape) # 16, 16, 256
    score = layers.conv_btn(drop5, [1, 1], 50, 'score', is_training = is_training, padding='VALID')
    print ('Score: ', score.shape) # 16, 16, 50
    # Encdoe end

    # Decode start
    # DeConv Block
    upsample1 = layers.deconv_upsample(score, 2,  'upsample1')
    print ('upsample1: ', upsample1.shape) # 64, 64, 50
    score_pool4 = layers.conv_btn(pool4, [1, 1], 50, 'score_pool4', is_training=is_training, padding='VALID')
    fuse_pool4 = tf.add(upsample1, score_pool4, 'fues_pool4')
    compress_fuse = layers.conv_btn(fuse_pool4, [1, 1], 1, 'compress_fuse', is_training=is_training, padding='VALID')
    upsample2 = layers.deconv_upsample(compress_fuse, 64, 'upsample2')
    print ('upsample2: ', upsample2.shape) # 2048, 2048, 50
    
    # Output processing
    net  = layers.conv(upsample2, [1, 2048], 3, 'net', activation_fn = None, padding='VALID')
    print ('Output: ', net.shape)
    decode_pc = tf.reshape(net, [batch_size, num_point, 3])
    print ('Decode PC_Shape: ', decode_pc.shape)
    # Deconde end
    return decode_pc


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred)
    ## 18/8/30 Yujing:  Calculate the loss by using Euclidean distanse.
    loss = tf.reduce_mean(tf.square(tf.subtract(pred, label)))
    tf.summary.scalar('loss', loss)

    # Enforce the transformation as orthogonal matrix
    '''
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)
    '''
    return loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
