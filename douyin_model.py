# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnetVLAD import NetVLAD
import tensorflow as tf
import numpy as np
import os,sys
import traceback
import time
import random as rd 
import glob
from tensorflow.python import debug as tf_debug
from tensorflow import flags
from tensorflow import app
from tensorflow import logging
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_tfrecord','')
flags.DEFINE_integer('batch_size',50,'batch size')
flags.DEFINE_integer('NEG',9,'NEG size')

flags.DEFINE_float('initial_lr',0.001,'')
flags.DEFINE_float('lr_decay_factor',0.7,'')
flags.DEFINE_integer('num_epochs_before_decay',5,'')

flags.DEFINE_float('l2_reg_lambda',0.05,'')
flags.DEFINE_integer('max_frames',15,'max_frames')
flags.DEFINE_integer('cluster_size',128,'cluster_size')
flags.DEFINE_boolean('add_batch_norm',True,'')
flags.DEFINE_boolean('is_training',True,'')
flags.DEFINE_boolean('use_fp16',False,'Train the model using fp16')

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000

MOVING_AVERAGE_DECAY = 0.9999   # The decay to use for the moving average
NUM_EPOCHS_PER_DECAY = 350.0    # Epochs after which learning rate decay


class DouyinAttrRecord(object):
    pass

douyin_attr = DouyinAttrRecord()
douyin_attr.video_fdim = 1024
douyin_attr.audio_fdim = 128
douyin_attr.train_per = 0.9

def resize_axis(tensor,axis,new_size,fill_value=0):
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor)) # just get the shape
    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])
    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)
    resized = tf.concat([tf.slice(tensor, tf.zeros_like(shape), shape),tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))], axis)
    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized

def _variable_on_cpu(name,shape,initializer):
    """ Helper to create a Variable stored on CPU memory"""
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name,shape,initializer = initializer, dtype = dtype)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    """ Helper to create an initialized Variable with weith decay"""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev = stddev, dtype = dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name = 'weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(video_input,audio_input,max_frames,cluster_size,add_batch_norm,is_training,dropout_ratio=1.0):
    
    video_input = tf.convert_to_tensor(video_input)
    audio_input = tf.convert_to_tensor(audio_input)

    reshaped_video_input = tf.reshape(video_input,[-1,douyin_attr.video_fdim])
    reshaped_audio_input = tf.reshape(audio_input,[-1,douyin_attr.audio_fdim])
    
    # Declare netVLAD definition
    video_NetVLAD = NetVLAD(douyin_attr.video_fdim,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetVLAD = NetVLAD(douyin_attr.audio_fdim,max_frames,cluster_size, add_batch_norm, is_training)
    
    with tf.variable_scope('video_netVLAD') as scope: 
        vlad_video = video_NetVLAD.forward(reshaped_video_input)

    with tf.variable_scope('audio_netVLAD') as scope:
        vlad_audio = audio_NetVLAD.forward(reshaped_audio_input)
    
    # video fc
    with tf.variable_scope('vfc_1') as scope:
        dim = vlad_video.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',shape = [dim,2048], stddev = 0.04, wd = 0.004)
        biases = _variable_on_cpu('biases',[2048],tf.constant_initializer(0.1))
        video_fc_1 = tf.nn.relu(tf.matmul(vlad_video,weights) + biases, name = scope.name)
    video_dp_1 = tf.nn.dropout(video_fc_1,dropout_ratio)

    with tf.variable_scope('vfc_2') as scope:
        weights = _variable_with_weight_decay('weights',shape = [2048,1024], stddev = 0.04, wd = 0.004)
        biases = _variable_on_cpu('biases',[1024],tf.constant_initializer(0.1))
        video_fc_2 = tf.nn.relu(tf.matmul(video_dp_1,weights) + biases, name = scope.name)
    video_dp_2 = tf.nn.dropout(video_fc_2,dropout_ratio)

    with tf.variable_scope('vfc_3') as scope:
        weights = _variable_with_weight_decay('weights',shape = [1024,128],stddev = 0.02,wd = 0.002)
        biases = _variable_on_cpu('biases',[128],tf.constant_initializer(0.0))
        video_fc_3 = tf.add(tf.matmul(video_dp_2,weights),biases,name=scope.name)

    # audio fc
    with tf.variable_scope('afc_1') as scope:
        dim = vlad_audio.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',shape = [dim,2048], stddev = 0.04, wd = 0.004)
        biases = _variable_on_cpu('biases',[2048],tf.constant_initializer(0.1))
        audio_fc_1 = tf.nn.relu(tf.matmul(vlad_audio,weights) + biases, name = scope.name)
    audio_dp_1 = tf.nn.dropout(audio_fc_1,dropout_ratio)

    with tf.variable_scope('afc_2') as scope:
        weights = _variable_with_weight_decay('weights',shape = [2048,1024], stddev = 0.04, wd = 0.004)
        biases = _variable_on_cpu('biases',[1024],tf.constant_initializer(0.1))
        audio_fc_2 = tf.nn.relu(tf.matmul(audio_dp_1,weights) + biases, name = scope.name)
    audio_dp_2 = tf.nn.dropout(audio_fc_2,dropout_ratio)

    with tf.variable_scope('afc_3') as scope:
        weights = _variable_with_weight_decay('weights',shape = [1024,128],stddev = 0.02,wd = 0.002)
        biases = _variable_on_cpu('biases',[128],tf.constant_initializer(0.0))
        audio_fc_3 = tf.add(tf.matmul(audio_dp_2,weights),biases,name=scope.name)
    
    return video_fc_3,audio_fc_3

def input_pipeline(tfrecords_dir,file_pattern,shuffle=True):
    tfrecords_list = tf.gfile.Glob(os.path.join(tfrecords_dir,file_pattern))
    train_file_num = int(len(tfrecords_list) * douyin_attr.train_per)
    tfrecords_list = tfrecords_list[0:train_file_num]

    filename_queue = tf.train.string_input_producer(tfrecords_list,shuffle = shuffle)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    
    context_features = {"id":tf.FixedLenFeature([],tf.string)}
    feature_names = ["rgb","audio"]
    sequence_features = {feature_name : tf.FixedLenSequenceFeature([],dtype=tf.string) for feature_name in feature_names}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
    
    video_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['rgb'],tf.float64),tf.float32),[-1,douyin_attr.video_fdim])
    audio_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['audio'],tf.float64),tf.float32),[-1,douyin_attr.audio_fdim])
    num_process_threads = 10
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    video_batch_data,audio_batch_data = tf.train.batch(
            tensors=[video_decoded_features,audio_decoded_features],
            batch_size=FLAGS.batch_size,
            num_threads = num_process_threads,
            capacity = min_queue_examples * FLAGS.batch_size,
            allow_smaller_final_batch = True,
            dynamic_pad=True)
    
    video_batch_data = resize_axis(tensor=video_batch_data,axis=1,new_size=FLAGS.max_frames)
    audio_batch_data = resize_axis(tensor=audio_batch_data,axis=1,new_size=FLAGS.max_frames)
    return video_batch_data,audio_batch_data,train_file_num

def eval_input_pipeline(tfrecords_dir,file_pattern,shuffle=True):
    tfrecords_list = tf.gfile.Glob(os.path.join(tfrecords_dir,file_pattern))
    train_file_num = int(len(tfrecords_list) * douyin_attr.train_per)
    tfrecords_list = tfrecords_list[(train_file_num+1):end]

    filename_queue = tf.train.string_input_producer(tfrecords_list,shuffle = shuffle)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    
    context_features = {"id":tf.FixedLenFeature([],tf.string)}
    feature_names = ["rgb","audio"]
    sequence_features = {feature_name : tf.FixedLenSequenceFeature([],dtype=tf.string) for feature_name in feature_names}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
    
    video_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['rgb'],tf.float64),tf.float32),[-1,douyin_attr.video_fdim])
    audio_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['audio'],tf.float64),tf.float32),[-1,douyin_attr.audio_fdim])
    num_process_threads = 10
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    video_batch_data,audio_batch_data = tf.train.batch(
            tensors=[video_decoded_features,audio_decoded_features],
            batch_size=FLAGS.batch_size,
            num_threads = num_process_threads,
            capacity = min_queue_examples * FLAGS.batch_size,
            allow_smaller_final_batch = True,
            dynamic_pad=True)
    
    video_batch_data = resize_axis(tensor=video_batch_data,axis=1,new_size=FLAGS.max_frames)
    audio_batch_data = resize_axis(tensor=audio_batch_data,axis=1,new_size=FLAGS.max_frames)
    return video_batch_data,audio_batch_data,len(tfrecords_list)

def loss(video_vec,audio_vec,neg=50):
    with tf.name_scope('neg'):
        tmp = tf.tile(audio_vec,[1,1])
        audio_vecs = tf.tile(audio_vec,[1,1])
        for i in range(neg):
            rand = rd.randint(1,FLAGS.batch_size+i) % FLAGS.batch_size
            audio_vecs = tf.concat([audio_vecs,
                tf.slice(tmp,[rand,0],[FLAGS.batch_size - rand,-1]),
                tf.slice(tmp,[0,0],[rand,-1])],0)
    
    with tf.name_scope('cos_sim'):
        query_video_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(video_vec),1,True)),[neg + 1,1])
        audio_norm = tf.sqrt(tf.reduce_sum(tf.square(audio_vecs),1,True))
        prod = tf.reduce_sum(tf.multiply(tf.tile(video_vec,[neg + 1,1]),audio_vecs),1,True)
        norm_prod = tf.multiply(query_video_norm,audio_norm)
        cos_sim_raw = tf.truediv(prod,norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw),[neg+1,-1])) * 20

    with tf.name_scope('hit_loss'):
        prob = tf.nn.softmax(cos_sim)
        hit_prob = tf.slice(prob,[0,0],[-1,1])
        hit_loss = -tf.reduce_sum(tf.log(hit_prob))
        tf.add_to_collection('losses', hit_loss)

    with tf.name_scope('accuracy'):
        correnct_prediction = tf.cast(tf.equal(tf.argmax(prob,1),0),tf.float32)
        hit_acc = tf.reduce_mean(correnct_prediction)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

def eval_accuracy(video_vec,audio_vec,neg=50):
    with tf.name_scope('neg'):
        tmp = tf.tile(audio_vec,[1,1])
        audio_vecs = tf.tile(audio_vec,[1,1])
        for i in range(neg):
            rand = rd.randint(1,FLAGS.batch_size+i) % FLAGS.batch_size
            audio_vecs = tf.concat([audio_vecs,
                tf.slice(tmp,[rand,0],[FLAGS.batch_size - rand,-1]),
                tf.slice(tmp,[0,0],[rand,-1])],0)
    
    with tf.name_scope('cos_sim'):
        query_video_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(video_vec),1,True)),[neg + 1,1])
        audio_norm = tf.sqrt(tf.reduce_sum(tf.square(audio_vecs),1,True))
        prod = tf.reduce_sum(tf.multiply(tf.tile(video_vec,[neg + 1,1]),audio_vecs),1,True)
        norm_prod = tf.multiply(query_video_norm,audio_norm)
        cos_sim_raw = tf.truediv(prod,norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw),[neg+1,-1])) * 20

    with tf.name_scope('accuracy'):
        prob = tf.nn.softmax(cos_sim)
        correnct_prediction = tf.cast(tf.equal(tf.argmax(prob,1),0),tf.float32)
        #hit_acc = tf.reduce_mean(correnct_prediction)
    return correnct_prediction


def _add_loss_summaries(total_loss):
    """ Add summaries for losses in douyin dssm model
    Generate moving averages for all losses and associated summaries for visualizing the performance of the network

    Returns:
        loss_averages_op: op for generating moving averages of losses
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss
    # do the same for the averaged version of the losses
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)',l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def train(total_loss,global_step):
    """ Train douyin dssm model
    Create an optimizer and apply to all trainable variables. Add moving average
    for all trainable variables

    Returns:
        train_op: op for training
    """
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size 
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.initial_lr,
                                    global_step,
                                    decay_steps,
                                    FLAGS.lr_decay_factor,
                                    staircase = True)
    tf.summary.scalar('learning_rate',lr)

    # Generate moving averages of all losses and associated summaries
    loss_averages_op = _add_loss_summaries(total_loss)

    # compute Gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

    # Add histograms for gradients
    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainables variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op






