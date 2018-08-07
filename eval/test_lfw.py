# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import logging
import time
import sys
import os
import pickle
import argparse
import cv2
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorflow.core.protobuf import config_pb2

from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
from util.verification import verification, extract_list_feature
from eval.tensorflow_extractor import TensorflowExtractor
from util.config import Config


def crop_image_list(img_list, imsize):
    out_list = []
    h, w, c = img_list[0][0].shape
    x1 = (w - imsize[0])/2
    y1 = (h - imsize[1])/2
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        img1 = img1[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        img1 = ( np.float32(img1) - 127.5 ) /128
        img2 = img2[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        img2 = ( np.float32(img2) - 127.5 ) /128
        out_list.append([img1, img2])
    #print(img1.shape)
    return out_list
    
def get_extractor(model_path, image_size = (112, 112)):
    images = tf.placeholder(name='img_inputs', shape=[None, image_size[1], image_size[0], 3], dtype=tf.float32)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    print('Buiding net structure')
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    #net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
    
    # test net  because of batch normal layer
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, reuse=tf.AUTO_REUSE, keep_rate=dropout_rate)
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
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", default='lfw', help="lfw")
    parser.add_argument("--data",   help="lfw.np or pair.txt")
    parser.add_argument("--prefix", default='.', help="data prefix")
    parser.add_argument("--model_path", help= 'specify which model to test ')
    parser.add_argument('--image_size', default=[112, 96], help='image size height, width')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    parser.add_argument("--dist_type", default='cosine', help="distance measure ['cosine', 'L2', 'SSD']")

    args = parser.parse_args()
    output_dir = '.'
    # parse args   
    image_size = (args.image_size[1], args.image_size[0]) 
    model_name = args.model_path
    test_set = args.test_set
    dist_type = args.dist_type

    print('Dataset  \t: %s (%s,%s)' % (args.test_set, args.data, args.prefix))
    print('Testing  \t: %s' % model_name)
    print('Distance \t: %s' % dist_type)
    # model
    print('Image size\t: {}'.format(image_size))
    # load images
    if args.data.find('.np') > 0:
        pos_img, neg_img = pickle.load(open(args.data, 'rb'))
        #pos_img, neg_img = pickle.load(open(lfw_data, 'rb'), encoding='iso-8859-1')
        
    # crop image
    pos_img = crop_image_list(pos_img, image_size)
    neg_img = crop_image_list(neg_img, image_size)
    #print(type(pos_img[0][0]))
    #exit()  
    extractor = get_extractor(model_name, image_size)
    # compute feature
    print('Extracting features ...')
    pos_list = extract_list_feature(extractor, pos_img, args.batch_size)
    print('  Done positive pairs')
    neg_list = extract_list_feature(extractor, neg_img, args.batch_size)
    print('  Done negative pairs')

    # evaluate
    print('Evaluating ...')
    precision, std, threshold, pos, neg, _ = verification(pos_list, neg_list, dist_type = dist_type)    
    #_, title = os.path.split(model_name)
    #draw_chart(title, output_dir, {'pos': pos, 'neg': neg}, precision, threshold)
    print('------------------------------------------------------------')
    print('Precision on %s : %1.5f+-%1.5f \nBest threshold   : %f' % (args.test_set, precision, std, threshold))
   
   

