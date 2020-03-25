##Tested

import tensorflow as tf
import numpy as np

def dist(a, b):
    M1 = tf.tile(tf.expand_dims(a, axis=1), [1, b.shape[0], 1])  # (na, nb, 2)
    M2 = tf.tile(tf.expand_dims(b, axis=0), [a.shape[0], 1, 1])  # (na, nb, 2)
    M = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(M1, M2)), axis=2))  # (na, nb)
    return M

def sinkhorn_map(a, b, M,  reg, L):
    '''
    params:
    a : pmf for 1st distribution; tensor of shape (n, )
    b : pmf for 2nd distribution;  tensor of shape (m, )
    M : distance matrix;  tensor of shape (n, m)
    reg : regularizer
    L : number of iterations
    
    return: transportaion probability map;  tensor of shape (n, m)
    '''
    gamma = tf.cast(1/reg, dtype=tf.float32)
    K = tf.exp(-tf.math.scalar_mul(gamma, M))
    v = tf.cast(np.ones((b.shape[0],)), dtype = tf.float32) # v_0 = 1_m
    for _ in range(L):
        u = tf.math.divide(a, tf.linalg.matvec(K, v)) # u_l = a/(Kv_l)
        v = tf.math.divide(b, tf.linalg.matvec(K, u, transpose_a = True)) # v_{l+1} = b/(K'u_l)
    P = tf.linalg.matmul(tf.linalg.tensor_diag(u), K)
    P = tf.linalg.matmul(P, tf.linalg.tensor_diag(v)) # P = diag(u_L) K diag(v_{L+1})
    return P

def sinkhorn_dist(x, y, reg, L):
    M = dist(x, y)
    a = tf.cast(np.ones((x.shape[0],))/x.shape[0], dtype = tf.float32)
    b = tf.cast(np.ones((y.shape[0],))/y.shape[0], dtype = tf.float32)
    P = sinkhorn_map(a, b, M, reg, L)
    return tf.reduce_sum(tf.math.multiply(M, P))

