# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


def _pairwise_distances(Y1, Y2, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        Y1: tensor of shape (p, q)
        Y2: tensor of shape (p, q)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (p, p)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    Y2T = tf.transpose(Y2)
    SY1 = tf.reduce_sum(tf.square(Y1), axis = 1, keep_dims = True)
    SY2 = tf.reduce_sum(tf.square(Y2T), axis = 0, keep_dims = True)
    dot_product = tf.matmul(Y1, Y2T)

    distances = SY1 - 2 * dot_product
    distances = SY2 + distances
    
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 1e-16)

    if not squared:
        distances = tf.sqrt(distances)

    return distances
    
  
def _HardNet_loss(anchor, positive, anchor_swap = True,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """
    eps = 1e-8
    dist_matrix = _pairwise_distances(anchor, positive) + eps

    # steps to filter out same patches that occur in distance matrix as negatives
    diag = tf.diag(dist_matrix, name = 'dist_diag')
    pos1 = tf.diag_part(dist_matrix)
    diag_off = dist_matrix - diag
    if batch_reduce == 'min':
        min_neg = tf.reduce_min(diag_off, 1)
        if anchor_swap:
            min_neg2 = tf.reduce_min(diag_off, 0)
            min_neg = tf.minimum(min_neg,min_neg2)
        min_neg = min_neg
        pos = pos1
    
    if loss_type == "triplet_margin":
        #loss = tf.clip_by_value(margin + pos - min_neg, 0, 3)
        loss = tf.maximum(margin + pos - min_neg, 0)
    
    loss = tf.reduce_mean(loss)
    return loss

    
def HardNet_loss(embeddings, margin = 1.0):
    images_s = tf.split(embeddings, num_or_size_splits=2, axis=0)
    Y1 = images_s[0]
    Y2 = images_s[1]
    return _HardNet_loss(Y1, Y2, margin = margin)