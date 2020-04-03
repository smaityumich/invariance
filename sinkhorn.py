##Tested

import tensorflow as tf
import numpy as np

def dist(a, b):
    #a = tf.math.l2_normalize(a)
    #b = tf.math.l2_normalize(b)
    M1 = tf.tile(tf.expand_dims(a, axis=1), [1, b.shape[0], 1])  # (na, nb, 2)
    M2 = tf.tile(tf.expand_dims(b, axis=0), [a.shape[0], 1, 1])  # (na, nb, 2)
    M = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(M1, M2)), axis=2))  # (na, nb)
    #M = tf.reduce_sum(tf.square(tf.subtract(M1, M2)), axis=2)  # (na, nb)
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
    M = tf.clip_by_value(tf.math.scalar_mul(gamma, M), 1e-3, 5)
    K = tf.exp(-M)
    #K = tf.clip_by_value(K, 1e-4, 10)
    v = tf.cast(np.ones((b.shape[0],)), dtype = tf.float32)/b.shape[0] # v_0 = 1_m
    for _ in range(L):
        u = tf.math.divide(a, tf.linalg.matvec(K, v) + 1e-10) # u_l = a/(Kv_l)
        v = tf.math.divide(b, tf.linalg.matvec(K, u, transpose_a = True) + 1e-10) # v_{l+1} = b/(K'u_l)
    P = tf.linalg.matmul(tf.linalg.tensor_diag(u), K)
    P = tf.linalg.matmul(P, tf.linalg.tensor_diag(v)) # P = diag(u_L) K diag(v_{L+1})
    return P

def sinkhorn_dist(x, y, reg, L):
    M = tf.square(dist(x, y))
    a = tf.cast(np.ones((x.shape[0],))/x.shape[0], dtype = tf.float32)
    b = tf.cast(np.ones((y.shape[0],))/y.shape[0], dtype = tf.float32)
    P = sinkhorn_map(a, b, M, reg, L)
    return tf.reduce_sum(tf.math.multiply(M, P))

