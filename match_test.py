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

class DouyinAVFeatureExtractor(object):
    def __int__(self):
        self.meta_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_log/model.ckpt-277703.meta'
        self.model_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_log/model.ckpt-277703'

        # print all tensors in checkpoint file
        #chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=True)
        
        self.douyin_graph = tf.Graph()
        with self.douyin_graph.as_default():
            self.saver = tf.train.import_meta_graph(self.meta_path)
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
    
            # restore all variables into the restored graph
            self.saver.restore(self.sess,self.model_path)
    
        """
        # print op names in the graph 
        for op in tf.get_default_graph().get_operations():
            print op.name
        """
    
    def extract_av_features(self,video_input,audio_input):
        with self.douyin_graph.as_default():
            video_features,audio_features = self.sess.run(['video_fc/video_fc_3/BiasAdd:0','audio_fc/audio_fc_3/BiasAdd:0'],feed_dict={'Reshape:0':video_input,'Reshape_1':audio_input})
        return video_features,audio_features

def get_inputs(tfrecords_dir,file_pattern):
    tfrecords_list = glob.glob(os.path.join(tfrecords_dir,file_pattern))
    train_file_num = int(len(tfrecords_list) * 0.9)
    test_file_num = len(tfrecords_list) - train_file_num
    print test_file_num
    tfrecords_list = tfrecords_list[train_file_num+1:]

    filename_queue = tf.train.string_input_producer(tfrecords_list,num_epochs=1)
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

    video_data = train_local.resize_axis(tensor=video_decoded_features,axis=0,new_size=15)
    audio_data = train_local.resize_axis(tensor=audio_decoded_features,axis=0,new_size=15)

    return video_data,audio_data

if __name__ == '__main__':
    # getting the training batch data
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    tf_data_dir = "/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_tfrecords/"
    tf_file_pattern = "*.tfrecords"
     
    video_input,audio_input = get_inputs(tf_data_dir,tf_file_pattern)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #init
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op) 
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while not coord.should_stop():
                meta_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_log/model.ckpt-277703.meta'
                model_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_log/model.ckpt-277703'
                saver = tf.train.import_meta_graph(meta_path)
                #for op in tf.get_default_graph().get_operations():
                #    print op.name
                print video_input
                video_res,audio_res = sess.run([video_input,audio_input])
                print video_res.shape,type(video_res)
                ####

    
                # restore all variables into the restored graph
                saver.restore(sess,model_path)
                print "inference"
                
                
                #video_features,audio_features = sess.run(['video_fc/video_fc_3/BiasAdd:0','audio_fc/audio_fc_3/BiasAdd:0'],feed_dict={'Reshape:0':video_res,'Reshape_1:0':audio_res})
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                for op in tf.get_default_graph().get_operations():
                    print op.name
                print video_res
                print audio_res
                #video_features = sess.run('audio_fc/audio_fc_2/BiasAdd',feed_dict={'Reshape:0':video_res,'Reshape_1:0':audio_res})
                video_features = sess.run('Reshape:0',feed_dict={'Reshape:0':video_res,'Reshape_1:0':audio_res})
                print 'finish'
                ####
                print video_features.shape
        except tf.errors.OutOfRangeError:
            print "OutOfRangeError"
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)
    

