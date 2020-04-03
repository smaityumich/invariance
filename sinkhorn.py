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

def sinkhorn(a, b, M,  reg, L):
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
    M = tf.math.scalar_mul(gamma, M)
    K = tf.exp(-M)
    #K = tf.clip_by_value(K, 1e-4, 10)
    v = tf.cast(np.ones((b.shape[0],)), dtype = tf.float32) # v_0 = 1_m
    for _ in range(L):
        u = tf.math.divide(a, tf.linalg.matvec(K, v) + 1e-16) # u_l = a/(Kv_l)
        v = tf.math.divide(b, tf.linalg.matvec(K, u, transpose_a = True) + 1e-16) # v_{l+1} = b/(K'u_l)
    P = tf.linalg.matmul(tf.linalg.tensor_diag(u), K)
    P = tf.linalg.matmul(P, tf.linalg.tensor_diag(v)) # P = diag(u_L) K diag(v_{L+1})
    return P


def sinkhorn_log_stabilized(a, b, M, reg, L, tau = 1e3):
    '''
    log-stabilizd sinkhorn: A simplified version
    params:
        same as sinkhorn
        tau: tolerence level for max dual potentials
        
    return:
        same as sinkhorn
        
    reference: 
        Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized 
        Transport Problems. arXiv preprint arXiv:1610.06519.
    '''
    def get_K(M, alpha, beta, reg):
        alpha_expanded = tf.tile(tf.expand_dims(alpha, axis=1), [1, beta.shape[0], 1])
        beta_expanded = tf.tile(tf.expand_dims(beta, axis=0), [alpha.shape[0], 1, 1])
        return tf.exp(-(M - alpha_expanded - beta_expanded)/reg)
    
    alpha = tf.zeros([a.shape[0],], dtype = tf.float32)
    beta = tf.zeros([b.shape[0],], dtype = tf.float32)
    
    v = tf.ones([b.shape[0],], dtype = tf.float32)/b.shape[0]
    u = tf.ones([a.shape[0],], dtype = tf.float32)/a.shape[0]
    
    for _ in range(L):
        
        K = get_K(M, alpha, beta, reg)
        v = tf.math.divide(b, tf.linalg.matvec(K, u, transpose_a = True) + 1e-16)
        u = tf.math.divide(a, tf.linalg.matvec(K, v) + 1e-16)
        
        if tf.math.reduce_max(tf.math.abs(u)) > tau or tf.math.reduce_max(tf.math.abs(v)) > tau:
            alpha, beta = alpha + reg*tf.math.log(u), beta + reg*tf.math.log(v)
            v, u = tf.ones([b.shape[0],], dtype = tf.float32)/b.shape[0], \
                tf.ones([a.shape[0],], dtype = tf.float32)/a.shape[0]
                
    K = get_K(M, alpha, beta, reg)
    P = tf.linalg.matmul(tf.linalg.tensor_diag(u), K)
    P = tf.linalg.matmul(P, tf.linalg.tensor_diag(v)) # P = diag(u_L) K diag(v_{L+1})
    return P
        
        

def sinkhorn_dist(x, y, reg, L):
    M = tf.square(dist(x, y))
    a = tf.cast(np.ones((x.shape[0],))/x.shape[0], dtype = tf.float32)
    b = tf.cast(np.ones((y.shape[0],))/y.shape[0], dtype = tf.float32)
    P = sinkhorn(a, b, M, reg, L)
    return tf.reduce_sum(tf.math.multiply(M, P))

