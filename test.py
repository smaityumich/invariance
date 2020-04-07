import tensorflow as tf
import sinkhorn as sh
import numpy as np
from setup import *
import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import json



## Train Data

def informative_feature(y):
    return np.random.normal(1, 1, (4,)) if y else np.random.normal(0, 1, (4,))

def spurious_label(y, p):
    z = np.random.random()
    return y if z<p else 1-y #np.random.binomial(1, 0.5) #1-y


y0 = np.random.binomial(1, 0.5, (1200,)) ##Bayes error 0.14
y1 = np.random.binomial(1, 0.7, (1500,))
#f = lambda y: np.random.normal(1, 1, (4,)) if y else np.random.normal(0, 1, (4,))
x0_inv = [informative_feature(spurious_label(y, 0.25)) for y in y0]
x1_inv = [informative_feature(spurious_label(y, 0.25)) for y in y1]
x0_non_inv = [informative_feature(spurious_label(y, 0.2)) for y in y0]
x1_non_inv = [informative_feature(spurious_label(y, 0.1)) for y in y1]
x0 = np.concatenate((x0_inv, x0_non_inv), axis = 1)
x1 = np.concatenate((x1_inv, x1_non_inv), axis = 1)
#x0 = np.concatenate((x0, np.random.normal(6, 1, (1200, 60))), axis = 1)
#x1 = np.concatenate((x1, np.random.normal(2, 1, (1500, 60))), axis = 1)
y0 = tf.one_hot(y0, 2)
y1 = tf.one_hot(y1, 2)
x0 = tf.cast(x0, dtype = tf.float32)
x1 = tf.cast(x1, dtype = tf.float32)

data_train = [[x0, y0], [x1, y1]]



## Test data

y0 = np.random.binomial(1, 0.5, (1000,))
y1 = np.random.binomial(1, 0.7, (1000,))
x0_inv = [informative_feature(spurious_label(y, 0.25)) for y in y0]
x1_inv = [informative_feature(spurious_label(y, 0.25)) for y in y1]
x0_non_inv = [informative_feature(spurious_label(y, 0.8)) for y in y0]
x1_non_inv = [informative_feature(spurious_label(y, 0.9)) for y in y1]
x0 = np.concatenate((x0_inv, x0_non_inv), axis = 1)
x1 = np.concatenate((x1_inv, x1_non_inv), axis = 1)
#x0 = np.concatenate((x0, np.random.normal(9, 1, (1000, 60))), axis = 1)
#x1 = np.concatenate((x1, np.random.normal(7, 1, (1000, 60))), axis = 1)
y0 = tf.one_hot(y0, 2)
y1 = tf.one_hot(y1, 2)
x0 = tf.cast(x0, dtype=tf.float32)
x1 = tf.cast(x1, dtype = tf.float32)

data_test = [[x0, y0], [x1, y1]]


#reg_wasserstein, reg_var, lr, gamma_wasserstein = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

fitted_graph = InvarLabelShift(data_train, data_test, num_steps=10, 
                        reg_wasserstein=1e-1, reg_var = 1e-1, learning_rate = 1e-3, 
                        wasserstein_epoch = 10, gamma_wasserstein = 1, sinkhorn_iter = 5)
