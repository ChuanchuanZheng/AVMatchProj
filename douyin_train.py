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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

FLAGS = flags.FLAGS

if __name__ == '__main__':

    flags.DEFINE_string('train_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_output','')
    flags.DEFINE_string('log_dir','/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_log','train log dir')

    flags.DEFINE_float('dropout_keep_prob',0.9,'dropout keep prob')
    flags.DEFINE_integer('max_steps',300000,'max_steps')

    flags.DEFINE_integer('log_frequency',10,'how often to log results to the console')
    flags.DEFINE_boolean('log_device_placement',False,'Whether to log device placement')
    flags.DEFINE_boolean('allow_soft_placement',False,'Whether to print the tensor appointment')

def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        
        # Force input pipeline to cpu:0 to avoid operations sometimes ending up 
        # on GPU and resulting in a slow down
        train_file_pattern = "*.tfrecords"
        with tf.device('/cpu:0'):
            video_sj,audio_sj,train_sj_num = douyin_model.input_pipeline(FLAGS.data_dir,train_file_pattern,shuffle = True)

        # Build a graph that computes the cosine similarity from the inference model
        video_feature,audio_feature = douyin_model.inference(video_sj,audio_sj,FLAGS.max_frames,FLAGS.cluster_size,FLAGS.add_batch_norm,FLAGS.is_training,FLAGS.dropout_keep_prob)

        # Caculate loss
        loss = douyin_model.loss(video_feature,audio_feature,FLAGS.NEG)

        train_op = douyin_model.train(loss,global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """ Logs loss and runtime """
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self,run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self,run_context,run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
            
                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
        
        tf_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
        tf_config.gpu_options.allow_growth = True
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir = FLAGS.train_dir,
                hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                        tf.train.NanTensorHook(loss),
                        _LoggerHook()],
                config = tf_config) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    app.run()


