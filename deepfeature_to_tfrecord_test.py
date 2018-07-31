# -*- coding=utf-8 -*-
import os,sys
import time
import threadpool
import Queue
import threading
import glob
import traceback
import ffmpy
import subprocess
import tensorflow as tf
import json
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ThreadFeed(object):
    def __init__(self,log = None):
        self.log = log
        self.base_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data'

    def process(self,fv,idx):
        cdict = json.loads(fv)
        imgfeas = np.array(json.loads(cdict['imgfeas']))
        audiofeas = np.array(json.loads(cdict['audiofeas']))
        vid = cdict['videoid']
        print vid
        vid = vid.encode('utf-8')
        tfrecords_filename = vid + '.tfrecords'
        tfrecords_base_path = os.path.join(self.base_path,'tfrecords_test')
        
        # create .tfrecord file preparing for writing
        writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_base_path,tfrecords_filename)) 
         
        imgfeas = imgfeas.reshape(-1,1024)
        audiofeas = audiofeas.reshape(-1,128)
        print imgfeas
        print imgfeas.dtype
        frame_features = [
                tf.train.Feature(bytes_list = tf.train.BytesList(value=[frame_feat.tostring()])) for frame_feat in imgfeas
                ]
        audio_features = [
                tf.train.Feature(bytes_list = tf.train.BytesList(value=[sub_audo_fea.tostring()])) for sub_audo_fea in audiofeas
                ]
        seq_example = tf.train.SequenceExample(
                context = tf.train.Features(feature = {
                    "id": tf.train.Feature(bytes_list = tf.train.BytesList(value=[vid]))
                }),
                feature_lists = tf.train.FeatureLists(feature_list = {
                    "rgb": tf.train.FeatureList(feature = frame_features),
                    "audio":tf.train.FeatureList(feature = audio_features)}
                )
        )
        serialized = seq_example.SerializeToString()
        writer.write(serialized)
        writer.close()
        return os.path.join(tfrecords_base_path,tfrecords_filename)

def consume(tf_path):
    filename_queue = tf.train.string_input_producer(tf_path,num_epochs=1)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)

    context_features = {"id":tf.FixedLenFeature([],tf.string)}
    feature_names = ["rgb","audio"]
    sequence_features = {feature_name : tf.FixedLenSequenceFeature([],dtype=tf.string) for feature_name in feature_names}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=serialized_example,
    context_features=context_features,
    sequence_features=sequence_features)
    
    vid = context_parsed['id']
    video_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['rgb'],tf.float64),tf.float32),[-1,1024])
    audio_decoded_features = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['audio'],tf.float64),tf.float64),[-1,128])
    video_batch,audio_batch = tf.train.batch(tensors=[video_decoded_features,audio_decoded_features],batch_size=1,dynamic_pad=True)
    #video_batch = video_decoded_features
    #audio_batch = audio_decoded_features
    return vid,video_batch,audio_batch

frame_audio_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/videos_ffmpeg'

if __name__ == '__main__': 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    testFeed = ThreadFeed()
    feature_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_feasv2.txt'
    tf_list = []
    count = 1
    print 'initial' + str(count)
    with open(feature_path,'r') as f:
        for num,line in enumerate(f):
            line = line.rstrip('\n')
            cp = testFeed.process(line,'1')
            tf_list.append(cp)
            if count > 1:
                print count
                break
            else:
                count += 1
    
    with tf.Graph().as_default():
        vid,v,a = consume(tf_list)
        #video_batch = tf.train.batch([v],batch_size=1,capacity=2,dynamic_pad=True)
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            try:
                while not coord.should_stop():
                    vvid,vi = sess.run([vid,v])
                    print vvid
                    print vi.shape
                    print vi
            except tf.errors.OutOfRangeError:
                print 'Thing Done.'
            finally:
                coord.request_stop()
            coord.join(threads)
    print "finish!"    
