# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import traceback
import time
import os,sys
from tensorflow.python.tools import inspect_checkpoint as chkp
import train_local
import glob
from tensorflow.python import debug as tf_debug
import train_local_sv
import json
from tensorflow import flags
from tensorflow import app
from tensorflow import logging
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

flags.DEFINE_float('dropout_keep_prob',0.9,'dropout keep prob')
flags.DEFINE_integer('batch_size',50,'batch size')
flags.DEFINE_integer('NEG',9,'NEG size')
 
flags.DEFINE_integer('num_epochs',30000,'num_epochs')
flags.DEFINE_integer('max_frames',15,'max_frames')
flags.DEFINE_integer('cluster_size',128,'cluster_size')
flags.DEFINE_bool('add_batch_norm',True,'')
flags.DEFINE_bool('is_training',True,'')

if __name__ == '__main__':
    # Getting the training batch data
    tf_data_dir = "/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_tfrecords/"
    tf_file_pattern = "*.tfrecords" 
    meta_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_output/model.ckpt-29702.meta'
    model_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_output/model.ckpt-29702'
    ckpt_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_output/' 
    
    write_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_eval/av_feature.txt'
    #print all tensors in checkpoint file
    #chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=True)
    
    with tf.Graph().as_default() as graph:
        vvid,video_input,audio_input,eval_num = train_local_sv.eval_input_pipeline(tf_data_dir,tf_file_pattern)
        video_feature,audio_feature = train_local_sv.build_model_inference(video_input,audio_input,max_frames=15,cluster_size=128,add_batch_norm=True,is_training=False,dropout_ratio=1.0)
        _,hit_acc = train_local_sv.closs(video_feature,audio_feature,l2_loss=0.0)
        saver = tf.train.Saver()
        
        tf_config = tf.ConfigProto(operation_timeout_in_ms=20000)
        tf_config.gpu_options.allow_growth = True
        coord = tf.train.Coordinator()
        
        with tf.Session(config=tf_config,graph=graph) as sess:
            """ 
            # print all tensors name 
            tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node] 
            for tensor_name in tensor_name_list: 
                print(tensor_name,'\n')
            """
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run([init_op]) 
            saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess) 
            try:
                while not coord.should_stop():
                    vvid_list,video_vec,audio_vec = sess.run([vvid,video_feature,audio_feature])
                    cdict = {}
                    with open(write_path,'a+') as f:
                        cdict['vid'] = json.dumps(vvid_list.tolist())
                        cdict['vvec'] = json.dumps(video_vec.tolist())
                        cdict['avec'] = json.dumps(audio_vec.tolist())
                        f.write(json.dumps(cdict))
                        f.write('\n')
                        print "%s finish" % (vvid_list)
            except Exception as e:
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
        

