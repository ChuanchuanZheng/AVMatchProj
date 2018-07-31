# -*- coding:utf-8 -*-
from frame_level_models import NetVLAD
import tensorflow as tf
import numpy as np
import os,sys
import traceback
import time
import random as rd 
import glob
import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow import app
from tensorflow import logging
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

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


def build_model_inference(video_input,audio_input):
    
    video_input = tf.convert_to_tensor(video_input)
    audio_input = tf.convert_to_tensor(audio_input)
    reshaped_video_input = tf.reshape(video_input,[-1,1024])
    reshaped_audio_input = tf.reshape(audio_input,[-1,128])

    video_NetVLAD = NetVLAD(1024,FLAGS.max_frames,FLAGS.cluster_size, FLAGS.add_batch_norm, FLAGS.is_training)
    audio_NetVLAD = NetVLAD(128,FLAGS.max_frames,FLAGS.cluster_size/2, FLAGS.add_batch_norm, FLAGS.is_training)
    
    with tf.variable_scope('video_netVLAD'): 
        vlad_video = video_NetVLAD.forward(reshaped_video_input)
    with tf.variable_scope('audio_netVLAD'):
        vlad_audio = audio_NetVLAD.forward(reshaped_audio_input)
    
    l2_penalty = 1e-8
    
    video_activation = slim.stack(vlad_video,slim.fully_connected,[2048,1024,128],activation_fn=None,weights_regularizer=slim.l2_regularizer(l2_penalty),scope="video_fc")
    audio_activation = slim.stack(vlad_audio,slim.fully_connected,[2048,1024,128],activation_fn=None,weights_regularizer=slim.l2_regularizer(l2_penalty),scope="audio_fc")

    return video_activation,audio_activation

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
    
    video_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['rgb'],tf.uint8),tf.float32),[-1,1024])
    audio_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['audio'],tf.uint8),tf.float32),[-1,128])
    
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
        prob = tf.nn.softmax((cos_sim))
        hit_prob = tf.slice(prob,[0,0],[-1,1])
        my_loss = -tf.reduce_mean(tf.log(hit_prob)) + l2_loss * l2_reg_lambda
    return my_loss

def main(unused_argv):
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        # getting the training batch data
        train_data_dir = "/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_tfrecords/"
        train_file_pattern = "*.tfrecords"
        video_batch_input,audio_batch_input,train_num = input_pipeline(train_data_dir,train_file_pattern)
        
        num_batches_per_epoch = int(train_num / FLAGS.batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # one step is one batch processed
        decay_steps = int(FLAGS.num_epochs_before_decay * num_steps_per_epoch)

        video_vec,audio_vec = build_model_inference(video_batch_input,audio_batch_input)
        
        l2_loss = tf.constant(0.0)
        """
        # print the trainable varialbes
        for var in tf.trainable_variables():
            print var,var.name
        """
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_netVLAD/cluster_weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_netVLAD/cluster_weights2:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_netVLAD/cluster_weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_netVLAD/cluster_weights2:0'))

        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_fc/video_fc_1/weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_fc/video_fc_1/biases:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_fc/video_fc_2/weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_fc/video_fc_2/biases:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_fc/video_fc_3/weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('video_fc/video_fc_3/biases:0'))

        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_fc/audio_fc_1/weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_fc/audio_fc_1/biases:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_fc/audio_fc_2/weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_fc/audio_fc_2/biases:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_fc/audio_fc_3/weights:0'))
        l2_loss += tf.nn.l2_loss(graph.get_tensor_by_name('audio_fc/audio_fc_3/biases:0'))
        
        video_vec_dp = tf.nn.dropout(video_vec, FLAGS.dropout_keep_prob)
        audio_vec_dp = tf.nn.dropout(audio_vec, FLAGS.dropout_keep_prob)
        
        loss_op = closs(video_vec_dp,audio_vec_dp,l2_loss=l2_loss,neg=FLAGS.NEG,l2_reg_lambda = FLAGS.l2_reg_lambda)
        
        # Optimizer
        tf.summary.scalar('loss',loss_op)
        global_step = tf.Variable(0,name='global_step',trainable=False)

        lr = tf.train.exponential_decay(
                learning_rate = FLAGS.initial_lr,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = FLAGS.lr_decay_factor,
                staircase = True)

        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        train_op = slim.learning.create_train_op(loss_op,optimizer)

        summary = tf.summary.merge_all()
        
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while not coord.should_stop():
                # Instantiate a SummaryWriter to output summaries and the Graph.
                summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
                for step in xrange(num_steps_per_epoch * FLAGS.num_epochs):
                    if step % num_batches_per_epoch == 0:
                        logging.info('Epoch %s/%s',step/num_batches_per_epoch+1,FLAGS.num_epochs)
                        lr_value = sess.run([lr])
                        logging.info('Current Learning Rate: %s',lr_value)
                    _,loss_value = sess.run([train_op,loss_op])
                    if step % 100 == 0:
                        logging.info('Step %s: loss = %.4f',step,loss_value)
                        summary_str = sess.run(summary)
                        summary_writer.add_summary(summary_str,step)
                        summary_writer.flush()
                    
                    vvec = sess.run([video_vec])
                    logging.info('vvec:%s',vvec)
                    #bv = graph.get_operation_by_name('video_fc/video_fc_3/biases').outputs[0]
                    #logging.info('bias v:%s',sess.run(bv))

                    if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.num_epochs * num_steps_per_epoch:
                        checkpoint_file = os.path.join(FLAGS.log_dir,'model.ckpt')
                        saver.save(sess,checkpoint_file,global_step=step)

        except tf.errors.OutOfRangeError:
            logging.info('OutOfRangeError')
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    app.run()
