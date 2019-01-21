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
    dot_product = tf.matmul(Y1, tf.transpose(Y2))

    distances = 2 - dot_product
    
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 1e-16)

    if not squared:
        distances = tf.sqrt(distances)

    return distances

def relative_softmax(reverse_d):
    # 
    S = tf.exp(reverse_d)
    Sii = tf.diag_part(S)
    # column
    sum_col = tf.reduce_sum(S, 0)
    S_c = tf.divide(Sii, sum_col)
    # row
    sum_row = tf.reduce_sum(s, 1).flatten()
    S_r = tf.divide(Sii, sum_row)
    # E1 loss
    loss_c = tf.reduce_sum(tf.log(S_c))
    loss_r = tf.reduce_sum(tf.log(S_r))
    loss = -0.5 * (loss_c + loss_r)
    return loss

    
def relative_distance_loss(embeddings, squared=False):
    batch_size = embedding.shape.as_list()[0]
    q = embedding.shape.as_list()[1]
    p = batch_size/2
    # slice Ys
    Y1 = tf.slice(embeddings, 0, p)
    Y2 = tf.slice(embeddings, p, 2*p)
    # D of Y1, Y2
    D = _pairwise_distances(Y1, Y2)
    reverse_d = 2 - D
    # E1
    E1 = relative_softmax(reverse_d)
    # E2 
    R1 = tf.matmul(tf.transpose(Y1), Y1) / q
    R1 = tf.square(R1)
    R2 = tf.matmul(tf.transpose(Y2), Y2) / q
    R2 = tf.square(R2)
    R1_ii = tf.diag_part(R1)
    R2_ii = tf.diag_part(R2)
    E2_1 = tf.reduce_sum(R1) - tf.reduce_sum(R1_ii)
    E2_2 = tf.reduce_sum(R2) - tf.reduce_sum(R2_ii)
    E2 = 0.5 * (E2_1 + E2_2)
    return E1 + E2
        
def DIF_loss(embeddings):
    batch_size = embedding.shape.as_list()[0]
    q = embedding.shape.as_list()[1]
    p = batch_size/2
    # slice Ys
    Y1 = tf.slice(embeddings, 0, p)
    Y2 = tf.slice(embeddings, p, 2*p)
    # G
    G = tf.matmul(Y1, tf.transpose(Y2))
    # loss
    E3 = relative_softmax(G)
    return E3
    