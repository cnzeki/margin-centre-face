# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import time
import os
import cv2
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorflow.core.protobuf import config_pb2


from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
from eval.tensorflow_extractor import TensorflowExtractor

def get_extractor(args):
    # load model
    model_path = args.model_path
    images = tf.placeholder(name='img_inputs', shape=[None, args.image_size[0], args.image_size[1], 3], dtype=tf.float32)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    print('Buiding net structure')
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)

    # test net  because of batch normal layer
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, 
      reuse=tf.AUTO_REUSE, keep_rate=dropout_rate)
    embedding_tensor = test_net.outputs
    # 3.10 define sess
    #sess = tf.Session()
    gpu_config = tf.ConfigProto(allow_soft_placement=True )
    gpu_config.gpu_options.allow_growth = True

    sess = tf.Session(config=gpu_config)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())
    # restore weights
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    # lfw validate
    feed_dict = {images: None, dropout_rate: 1.0}
    
    #feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
    extractor = TensorflowExtractor(sess, embedding_tensor, args.batch_size, feed_dict, images)
    
    return extractor

   
    