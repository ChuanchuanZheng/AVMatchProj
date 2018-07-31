# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from frame_level_models import NetVLAD
import douyin_model
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
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = flags.FLAGS

if __name__ == '__main__':

    flags.DEFINE_string('eval_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_eval','')
    flags.DEFINE_string('checkpoint_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_log','')

    flags.DEFINE_integer('eval_interval_secs',60*5,'how often to run the eval')

    flags.DEFINE_integer('num_examples',3896,'Number of examples to run')
    flags.DEFINE_boolean('run_once',False,'Whether to run eval only once')

def eval_once(saver,summary_writer,hit_pred,summary_op):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess,ckpt.model_checkpoint_path)
            # Extract global_step from it
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print ('No checkpoint file found.')
            return
        
        # Start the queue runners
        coord = tf.train.Coordinate()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_QUNNERS):
                threads.extend(qr.create_threads(sess,coord=coord,daemon=True,start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0 
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([hit_pred])
                true_count += np.sum(predictions)
                step += 1

            precision = true_count / total_sample_count
            print ('%s: precision @ 1 = %.3f' % (datetime.now(),precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary,global_step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=10)

def evaluate():
    with tf.Graph().as_default() as g:
        train_file_pattern = "*.tfrecords"
        video_sj,audio_sj,test_sj_num = douyin_model.eval_input_pipeline(FLAGS.data_dir,train_file_pattern,shuffle = False)

        # Build a graph that computes the cosine similarity from the inference model
        video_feature,audio_feature = douyin_model.inference(video_sj,audio_sj,FLAGS.max_frames,FLAGS.cluster_size,FLAGS.add_batch_norm,FLAGS.is_training,FLAGS.dropout_keep_prob)

        # Caculate loss
        hit_pred = douyin_model.eval_accuracy(video_feature,audio_feature,FLAGS.NEG)
        
        # Restore the moving average version of the learned variables for eval.
        variables_averages = tf.train.ExponentialMovingAverage(douyin_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variables_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWrtier(FLAGS.eval_dir, g)

        while True:
            eval_once(saver,summary_writer,hit_pred,summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
    
def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    app.run()


