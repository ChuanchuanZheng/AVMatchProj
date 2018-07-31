#! /bin/bash

#nohup python train_local_sv.py --data_dir='/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/douyin_tfrecords/' --train_dir='/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_output/' --dropout_keep_prob=0.9 --batch_size=100 --NEG=50 --l2_reg_lambda=0.05 --num_epochs=1000 --max_frames=15 --cluster_size=128 --add_batch_norm=True --is_training=True --initial_lr=0.001 --lr_decay_factor=0.9 --log_dir='/data1/sina_recmd/simba/trunk/src/content_analysis/douyin/code/data/cdssm_output' &>my.log &

nohup python match_test_sv.py --dropout_keep_prob=1.0 --batch_size=1 --NEG=5 --num_epochs=100 --max_frames=15 --cluster_size=128 --add_batch_norm=True --is_training=False &>730_1.log &
