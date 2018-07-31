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
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

class ThreadFeed(object):
    def __init__(self,log = None):
        self.log = log
        self.base_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data'

    def process(self,fv,idx):
        cdict = json.loads(fv)
        imgfeas = np.array(json.loads(cdict['imgfeas']))
        audiofeas = np.array(json.loads(cdict['audiofeas']))
        vid = cdict['videoid']
        vid = vid.encode('utf-8')
        tfrecords_filename = vid + '.tfrecords'
        tfrecords_base_path = os.path.join(self.base_path,'tfrecords_test')
        
        # create .tfrecord file preparing for writing
        writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_base_path,tfrecords_filename)) 
         
        imgfeas = imgfeas.reshape(-1,1024)
        audiofeas = audiofeas.reshape(-1,128)
        print imgfeas.shape,audiofeas.shape
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
        return tfrecords_base_path
        

def consume(idx,thread,workQueue,queueLock):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            inputs = workQueue.get()
            queueLock.release()
            try:
                res = thread.process(inputs,idx)
            except:
                print traceback.print_exc()
                continue
        else:
            queueLock.release()

frame_audio_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/videos_ffmpeg'

exitFlag = 0

if __name__ == '__main__':
    threads = []
    threads_num = 1
    for i in range(threads_num):
        threads.append(ThreadFeed())

    queue_size = 5
    workQueue = Queue.Queue(queue_size)
    queueLock = threading.Lock()

    for i in range(threads_num):
        threading.Thread(target = consume,args = (str(i),threads[i],workQueue,queueLock)).start()
    
    feature_path = '/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_feasv2.txt'
    with open(feature_path,'r') as f:
        for num,line in enumerate(f):
            line = line.rstrip('\n')
            if num % 1000 == 0:
                print str(num) + 'th finished.'
            if not workQueue.full():
                queueLock.acquire()
                workQueue.put(line)
                queueLock.release()
            else:
                time.sleep(2)
    while not workQueue.empty():
        time.sleep(1)
    exitFlag = 1
    print "finish!"    
