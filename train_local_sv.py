# -*- coding:utf-8 -*-
from frame_level_models import NetVLAD
import tensorflow as tf
#from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import numpy as np
import os,sys
import traceback
import time
import random as rd 
import glob
from tensorflow.python import debug as tf_debug
import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow import app
from tensorflow import logging
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = flags.FLAGS

if __name__ == '__main__':
    flags.DEFINE_string('data_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_tfrecord','')
    flags.DEFINE_string('train_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_output','')
    flags.DEFINE_float('dropout_keep_prob',0.9,'dropout keep prob')
    flags.DEFINE_integer('batch_size',50,'batch size')
    flags.DEFINE_integer('NEG',9,'NEG size')

    flags.DEFINE_float('initial_lr',0.001,'')
    flags.DEFINE_float('lr_decay_factor',0.7,'')
    flags.DEFINE_integer('num_epochs_before_decay',5,'')

    flags.DEFINE_float('l2_reg_lambda',0.05,'')
    flags.DEFINE_integer('num_epochs',30000,'num_epochs')
    flags.DEFINE_integer('max_frames',15,'max_frames')
    flags.DEFINE_integer('cluster_size',128,'cluster_size')
    flags.DEFINE_bool('add_batch_norm',True,'')
    flags.DEFINE_bool('is_training',True,'')
    flags.DEFINE_string('log_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_log','train log dir')



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

def create_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer, trainable=trainable)


def fullconnect(name, inputs, num_outputs):
    with tf.variable_scope(name):
        
        fc_size = inputs.get_shape()
        input_fc = fc_size[1]
        output_fc = num_outputs

        weights_shape = [input_fc, output_fc]
        #weights_initializer = tf.truncated_normal_initializer(stddev=param.FC_WEIGHT_STDDEV) #this lead to large loss!
        weights_initializer = tf.contrib.layers.xavier_initializer()
        weights = create_variable('weights', shape=weights_shape, initializer=weights_initializer, weight_decay=0.99)

        bias_shape = [num_outputs]
        bias_initializer = tf.zeros_initializer
        bias = create_variable('bias', shape=bias_shape, initializer=bias_initializer, weight_decay=0.99)   
        
        inputs = tf.nn.xw_plus_b(inputs, weights, bias)
                                                
        return inputs

def relu(inputs):
    inputs = tf.nn.relu(inputs)
    return inputs

def build_model_inference(video_input,audio_input,max_frames,cluster_size,add_batch_norm,is_training,dropout_ratio=1.0):
    
    video_input = tf.convert_to_tensor(video_input)
    audio_input = tf.convert_to_tensor(audio_input)

    reshaped_video_input = tf.reshape(video_input,[-1,1024])
    reshaped_audio_input = tf.reshape(audio_input,[-1,128])

    video_NetVLAD = NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetVLAD = NetVLAD(128,max_frames,cluster_size, add_batch_norm, is_training)
    
    with tf.variable_scope('video_netVLAD'): 
        vlad_video = video_NetVLAD.forward(reshaped_video_input)
    with tf.variable_scope('audio_netVLAD'):
        vlad_audio = audio_NetVLAD.forward(reshaped_audio_input)
    
    video_fc_1 = fullconnect('video_fc_1',vlad_video,num_outputs=2048)
    video_relu_1 = relu(video_fc_1)
    video_dp_1 = tf.nn.dropout(video_relu_1, dropout_ratio)

    video_fc_2 = fullconnect('video_fc_2',video_dp_1,num_outputs=1024)
    video_relu_2 = relu(video_fc_2) 
    video_dp_2 = tf.nn.dropout(video_relu_2, dropout_ratio)
    
    video_fc_3 = fullconnect('video_fc_3',video_dp_2,num_outputs=128)

    audio_fc_1 = fullconnect('audio_fc_1',vlad_audio,num_outputs=2048)
    audio_relu_1 = relu(audio_fc_1)
    audio_dp_1 = tf.nn.dropout(audio_relu_1, dropout_ratio)
    
    audio_fc_2 = fullconnect('audio_fc_2',audio_dp_1,num_outputs=1024)
    audio_relu_2 = relu(audio_fc_2) 
    audio_dp_2 = tf.nn.dropout(audio_relu_2, dropout_ratio)
    
    audio_fc_3 = fullconnect('audio_fc_3',audio_dp_2,num_outputs=128)
    
    return video_fc_3,audio_fc_3

def input_pipeline(tfrecords_dir,file_pattern):
    tfrecords_list = glob.glob(os.path.join(tfrecords_dir,file_pattern))
    train_file_num = int(len(tfrecords_list) * 0.9)
    tfrecords_list = tfrecords_list[0:train_file_num]

    filename_queue = tf.train.string_input_producer(tfrecords_list,num_epochs=FLAGS.num_epochs)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    
    context_features = {"id":tf.FixedLenFeature([],tf.string)}
    feature_names = ["rgb","audio"]
    sequence_features = {feature_name : tf.FixedLenSequenceFeature([],dtype=tf.string) for feature_name in feature_names}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
    
    video_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['rgb'],tf.float64),tf.float32),[-1,1024])
    audio_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['audio'],tf.float64),tf.float32),[-1,128])
    
    video_batch_data,audio_batch_data = tf.train.batch(
            tensors=[video_decoded_features,audio_decoded_features],
            batch_size=FLAGS.batch_size,
            num_threads = 10,
            capacity = 4 * FLAGS.batch_size,
            allow_smaller_final_batch = True,
            dynamic_pad=True)
    
    video_batch_data = resize_axis(tensor=video_batch_data,axis=1,new_size=FLAGS.max_frames)
    audio_batch_data = resize_axis(tensor=audio_batch_data,axis=1,new_size=FLAGS.max_frames)
    return video_batch_data,audio_batch_data,train_file_num

def eval_input_pipeline(tfrecords_dir,file_pattern):
    tfrecords_list = glob.glob(os.path.join(tfrecords_dir,file_pattern))
    train_file_num = int(len(tfrecords_list) * 0.9)
    tfrecords_list = tfrecords_list[(train_file_num+1):]

    filename_queue = tf.train.string_input_producer(tfrecords_list,num_epochs=FLAGS.num_epochs)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    
    context_features = {"id":tf.FixedLenFeature([],tf.string)}
    feature_names = ["rgb","audio"]
    sequence_features = {feature_name : tf.FixedLenSequenceFeature([],dtype=tf.string) for feature_name in feature_names}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
    
    video_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['rgb'],tf.float64),tf.float32),[-1,1024])
    audio_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['audio'],tf.float64),tf.float32),[-1,128])
    vid = context_parsed['id']

    vvid,video_batch_data,audio_batch_data = tf.train.batch(
            tensors=[vid,video_decoded_features,audio_decoded_features],
            batch_size=FLAGS.batch_size,
            num_threads = 10,
            capacity = 4 * FLAGS.batch_size,
            allow_smaller_final_batch = True,
            dynamic_pad=True)
    
    video_batch_data = resize_axis(tensor=video_batch_data,axis=1,new_size=FLAGS.max_frames)
    audio_batch_data = resize_axis(tensor=audio_batch_data,axis=1,new_size=FLAGS.max_frames)
    return vvid,video_batch_data,audio_batch_data,len(tfrecords_list)

def closs(video_vec,audio_vec,l2_loss,neg=50,l2_reg_lambda=0.05):
    with tf.name_scope('neg'):
        tmp = tf.tile(audio_vec,[1,1])
        audio_vecs = tf.tile(audio_vec,[1,1])
        for i in range(neg):
            rand = rd.randint(1,FLAGS.batch_size+i) % FLAGS.batch_size
            audio_vecs = tf.concat([audio_vecs,
                tf.slice(tmp,[rand,0],[FLAGS.batch_size - rand,-1]),
                tf.slice(tmp,[0,0],[rand,-1])],0)
    
    with tf.name_scope('Cosine_Similarity'):
        query_video_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(video_vec),1,True)),[neg + 1,1])
        audio_norm = tf.sqrt(tf.reduce_sum(tf.square(audio_vecs),1,True))
        prod = tf.reduce_sum(tf.multiply(tf.tile(video_vec,[neg + 1,1]),audio_vecs),1,True)
        norm_prod = tf.multiply(query_video_norm,audio_norm)
        cos_sim_raw = tf.truediv(prod,norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw),[neg+1,-1])) * 20

    with tf.name_scope('Loss'):
        prob = tf.nn.softmax(cos_sim)
        hit_prob = tf.slice(prob,[0,0],[-1,1])
        my_loss = -tf.reduce_sum(tf.log(hit_prob)) + l2_loss * l2_reg_lambda

    with tf.name_scope('accuracy'):
        correnct_prediction = tf.cast(tf.equal(tf.argmax(prob,1),0),tf.float32)
        acc = tf.reduce_mean(correnct_prediction)
    return my_loss,acc

def main(unused_argv):
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    
    with tf.Graph().as_default() as graph:
        # Set tf logging level
        tf.logging.set_verbosity(tf.logging.INFO)
        # Gett training batch data
        train_data_dir = "/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_tfrecords/"
        train_file_pattern = "*.tfrecords"
        video_batch_input,audio_batch_input,train_num = input_pipeline(train_data_dir,train_file_pattern)
        
        num_batches_per_epoch = int(train_num / FLAGS.batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # one step is one batch processed
        decay_steps = int(FLAGS.num_epochs_before_decay * num_steps_per_epoch)

        video_vec,audio_vec = build_model_inference(video_batch_input,audio_batch_input,FLAGS.max_frames,FLAGS.cluster_size,FLAGS.add_batch_norm,FLAGS.is_training,0.7)
        
        l2_loss = tf.constant(0.0)
        """
        # print the trainable varialbes
        for var in tf.trainable_variables():
            print var,var.name
        """
        loss_op,acc_op = closs(video_vec,audio_vec,l2_loss=l2_loss,neg=FLAGS.NEG,l2_reg_lambda = FLAGS.l2_reg_lambda)
        
        # Optimizer
        tf.summary.scalar('loss',loss_op)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        #global_step += 1

        lr = tf.train.exponential_decay(
                learning_rate = FLAGS.initial_lr,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = FLAGS.lr_decay_factor,
                staircase = True)

        #train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        train_op = slim.learning.create_train_op(loss_op,optimizer)

        summary = tf.summary.merge_all()

        def train_step(sess,train_op,global_step):
            start_time = time.time()
            total_loss,global_step_count = sess.run([train_op,global_step])
            time_elapsed = time.time() - start_time

            logging.info("global step %s: loss: %.2f(%.2f sec/step)",global_step_count,total_loss,time_elapsed)
            return total_loss,global_step_count 
 
        sv = tf.train.Supervisor(logdir = FLAGS.log_dir,summary_op=summary,global_step=global_step,save_model_secs = 600, save_summaries_secs = 60)
        
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # Instantiate a SummaryWriter to output summaries and the Graph.
        with sv.managed_session(config=tf_config) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            for step in xrange(num_steps_per_epoch * FLAGS.num_epochs):
                if sv.should_stop():
                    break
                #logging.info('video_batch_input: %s',sess.run(video_batch_input))
                #logging.info('video_batch_input shape %s',video_batch_input.shape)
                
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s',step/num_batches_per_epoch+1,FLAGS.num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s',learning_rate_value)
                
                # print op names in the graph 
                #ccount = 0
                #for op in tf.get_default_graph().get_operations():
                #    ccount += 1
                
                loss,_ = train_step(sess,train_op,sv.global_step) 
                logging.info('Loss: %s',loss)
                logging.info('accuracy :%s',sess.run(acc_op))
            logging.info('Final loss:%s',loss)
            logging.info('finished.')
            sv.stop()

if __name__ == '__main__':
    app.run()


