# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import logging
import time
import os
import pickle
import argparse
import cv2
import sklearn
import sklearn.preprocessing
from tqdm import *


from util.verification import verification, compute_distance
from eval.resnet50_extractor import get_extractor
from util.config import Config

def parse_split_file(split_path, prefix, count = -1):
    pos_list = []
    neg_list = []
    with open(split_path, 'r') as f:
        # skip header
        f.readline()
        for line in f.readlines():
            vec = line.strip().split(',')
            #print(vec)
            flag = int(vec[4].strip())
            img_dir1 = '%s/%s' % (prefix, vec[2].strip())
            img_dir2 = '%s/%s' % (prefix, vec[3].strip())
            pair = [img_dir1, img_dir2]
            if not os.path.exists(img_dir1) or not os.path.exists(img_dir2):
                print(pair)
                continue
            if flag:
                pos_list.append(pair)
            else:
                neg_list.append(pair)
    if count > 0:
        return pos_list[0:100], neg_list[0:100]
    return pos_list, neg_list

def is_in_list(q, L):
    for l in L:
        if q.find(l) >= 0:
            return True
    return False
    
    
def filt_error_pairs(error_file, plist):
    elist = []
    with open(error_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            elist.append(line)
    #
    out_list = []
    for p in plist:
        if is_in_list(p[0], elist) or is_in_list(p[1], elist):
            continue
        out_list.append(p)
    return out_list
    
    
def get_dir_list(pos_list, neg_list):
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    all_list = []
    all_list += pos_list[:,0].tolist()
    all_list += pos_list[:,1].tolist()
    all_list += neg_list[:,0].tolist()
    all_list += neg_list[:,1].tolist()
    new_list = []
    for id in all_list:
        if id not in new_list:
            new_list.append(id)  
    return new_list

    
def extract_list_feature(extractor, img_list, batch_size, size = 0):
    feat_list = []
    npairs = len(img_list)
    if size == 0:
        size = len(img_list)
    size = min(size, npairs)
    nbatch = (size + batch_size - 1) // batch_size

    for batch in range(nbatch):
        # make a batch
        x_list = []
        for i in range(batch_size):
            pairid = (batch * batch_size + i)
            if pairid >= npairs:
                pairid = npairs - 1
            x_list.append(img_list[pairid])
        #
        x_batch = np.stack(x_list, axis=0)
        feat = extractor.extract(x_batch)
        
        for i in range(batch_size):
            a = feat[i,:]
            if len(feat_list) < size:
                feat_list.append(a)
    
    return feat_list


def load_set_images(img_dir):
    if not os.path.exists(img_dir):
        return []
    sub_imgs = os.listdir(img_dir)
    sub_imgs.sort()
    img_list = []
    for img_path in sub_imgs:
        title, ext = os.path.splitext(img_path)
        if ext not in ['.jpg', '.png']:
            continue
        im = cv2.imread(os.path.join(img_dir,img_path))
        if im is None:
            print(img_dir,img_path)
            continue
        img_list.append(im)
    return img_list   
    
    
def crop_image_list(img_list, imsize):
    out_list = []
    h, w, c = img_list[0].shape
    x1 = (w - imsize[0])/2
    y1 = (h - imsize[1])/2
    for pair in img_list:
        img1 = pair
        img1 = img1[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        img1 = ( np.float32(img1) - 127.5 ) /128
        out_list.append(img1)
    #print(img1.shape)
    return out_list
    
def extract_set_features(dir_list, extractor, imsize = (96, 112)):
    feat_dict = {}
    for seq in tqdm(dir_list):
        if seq in feat_dict:
            continue
        imgs = load_set_images(seq) 
        imgs = crop_image_list(imgs, imsize)
        feat_list = extract_list_feature(extractor, imgs, extractor.batch_size)
        feat_dict[seq] = feat_list
    return feat_dict
    

def pool_mean(F):
    F = np.array(F)
    feature = np.mean(F, axis=0, keepdims=True)
    feature = sklearn.preprocessing.normalize(feature).flatten()
    return feature
    
def pool_norm_mean(F):
    F = np.array(F)
    F = sklearn.preprocessing.normalize(F)
    feature = np.mean(F, axis=0, keepdims=True)
    feature = sklearn.preprocessing.normalize(feature).flatten()
    return feature
    
def pool_set_features(feat_dict, pool_func):
    pool_dict = {}
    for k, v in feat_dict.items():
        #print(v)
        pool_dict[k] = pool_func(v)
    return pool_dict
   
   
def extract_pool_feature(pair_list, pool_dict):
    pool_list = []
    #print(pool_dict)
    for pair in pair_list:
        #print(pair[0])
        pool_list.append([pool_dict[pair[0]], pool_dict[pair[1]]])
    return pool_list
   
def run_extract_feature(args, dir_list):
    # load model
    extractor = get_extractor(args)
    
    # extract features
    print('\nExtracting features ...')
    feat_dict = extract_set_features(dir_list, extractor, imsize = (args.image_size[1], args.image_size[0]))
    return feat_dict


def load_dir_img(dname):
    flist = os.listdir(dname)
    for f in flist:
        img = cv2.imread(os.path.join(dname, f))
        if img is None:
            continue
        return img

def save_error_list(pos_dist, threshold, pos_list, dst_dir, flag):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    count = 0    
    for i in range(len(pos_dist)):
        dist = pos_dist[i][0]
        diff = dist - threshold
        if diff * flag < 0:
            continue
        # pair images
        p0 = load_dir_img(pos_list[i][0])
        _segs = pos_list[i][0].split('/')
        p0_name = _segs[-2]
        p0_id = _segs[-1]
        p1 = load_dir_img(pos_list[i][1])
        _segs = pos_list[i][1].split('/')
        p1_name = _segs[-2]
        p1_id = _segs[-1]
        # image shape
        h, w, c = p0.shape
        image = np.zeros((h + 40, w * 2, 3), dtype=np.uint8)
        
        image[0:h,0:w,:] = p0
        image[0:h,w:(2*w),:] = p1
        cv2.putText(image, '%.4f'%(dist), (10,h+10), 1, 1, (0,255,0) )
        image_name = '%.4f-%03d_%s_%s-%s_%s.png'%(dist, count,p0_name, p0_id,p1_name, p1_id )
        cv2.imwrite(os.path.join(dst_dir, image_name), image)
        count += 1
        
        
def save_error_pair(pos_dist, neg_dist, threshold, pos_list, neg_list, dst_dir):
    save_error_list(pos_dist, threshold, pos_list, dst_dir+'/pos', flag = 1)
    save_error_list(neg_dist, threshold, neg_list, dst_dir+'/neg', flag = -1)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.ini', help='config file')
    parser.add_argument("--model_path",   help="ckpt path")
    parser.add_argument('--image_size', default=[112, 96], help='image height width')
    parser.add_argument("--dist_type", default='cosine', help="distance measure ['cosine', 'L2', 'SSD']")
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    args = parser.parse_args()
    # ytf data
    config = Config(args.config)
    args.data = config.get('ytf').splits
    args.prefix = config.get('ytf').prefix
    args.ytf_error = config.get('ytf').ytf_error
    args.filt_error = True
    # model
    #args.model_path = './output/vgg-ms1m/iter_258000_lfw_0.99133_ytf_0.95900.ckpt'
    #args.model_path = './output/large-ds-triplet/iter_424000_lfw_0.99100_ytf_0.95167.ckpt'
    #args.model_path = './models/InsightFace_iter_best_710000.ckpt'
    #args.model_path = './output/large-ds-triplet/iter_426000_lfw_0.99333_ytf_0.95333.ckpt'
    _, title = os.path.split(args.model_path)
    args.feat_cache = './output/' + title + '.pkl'
    output_dir = './output/'
    # load pairs
    pos_list, neg_list = parse_split_file(args.data, args.prefix)
    print("pos:%d neg:%d"%(len(pos_list), len(neg_list)))
    dir_list = get_dir_list(pos_list, neg_list)
    print("total sets:%d" % (len(dir_list)))
    if args.filt_error:
        print("Filt pos sets")
        pos_list = filt_error_pairs(args.ytf_error, pos_list)
        print("pos:%d neg:%d"%(len(pos_list), len(neg_list)))
        #dir_list = get_dir_list(pos_list, neg_list)
        #print("total sets:%d" % (len(dir_list)))
    
    if os.path.exists(args.feat_cache):
        feat_dict = pickle.load(open(args.feat_cache, 'rb'))
    else:
        feat_dict = run_extract_feature(args, dir_list)
        pickle.dump(feat_dict, open(args.feat_cache, 'wb'), 2)
    # pool seq feature
    pool_feat = pool_set_features(feat_dict, pool_norm_mean)

    # pair feature
    pos_feat = extract_pool_feature(pos_list, pool_feat)
    neg_feat = extract_pool_feature(neg_list, pool_feat)
    
    # evaluate
    print('Evaluating ...')
    precision, std, threshold, pos, neg, _ = verification(pos_feat, neg_feat, dist_type = args.dist_type)    
    #draw_chart(title, output_dir, {'pos': pos, 'neg': neg}, precision, threshold)
    print('------------------------------------------------------------')
    print('Precision on ytf : %1.5f+-%1.5f \nBest threshold   : %f' % ( precision, std, threshold))
    # error pairs
    pos_dist, neg_dist = compute_distance(pos_feat, neg_feat, dist_type = args.dist_type) 
    save_error_pair(pos_dist, neg_dist, threshold, pos_list, neg_list, output_dir)
   
    