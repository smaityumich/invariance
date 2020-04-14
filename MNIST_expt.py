import tensorflow as tf
import sinkhorn as sh
import numpy as np
from setup import *
import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import json
from mnist_data_load import *

np.random.seed(1)
x0_train, y0_train, x0_test, y0_test = make_environment(path='MNIST', red_0_corrs=[0.1, 0.9])
x1_train, y1_train, x1_test, y1_test = make_environment(path='MNIST', red_0_corrs=[0.3, 0.7])


x0_train, x0_test = tf.cast(x0_train, dtype=tf.float32), tf.cast(x0_test, dtype=tf.float32)
x1_train, x1_test = tf.cast(x1_train, dtype=tf.float32), tf.cast(x1_test, dtype=tf.float32)

y0_train, y0_test = tf.one_hot(y0_train, 2), tf.one_hot(y0_test, 2)
y1_train, y1_test = tf.one_hot(y1_train, 2), tf.one_hot(y1_test, 2)


data_train = [[x0_train, y0_train], [x1_train, y1_train]]
data_test = [[x0_test, y0_test], [x1_test, y1_test]]


#np.random.seed(1)
#data_train, _ = make_environments(path = 'MNIST', red_0_corrs = [0.7, 0.9])
#x0, y0 = data_train[0]
#x1, y1 = data_train[1]
#y0 = tf.one_hot(y0, 2)
#y1 = tf.one_hot(y1, 2)
#x0 = tf.cast(x0, dtype = tf.float32)
#x1 = tf.cast(x1, dtype = tf.float32)
#data_train = [[x0, y0], [x1, y1]]

#data_test, _ = make_environments(path = 'MNIST', red_0_corrs = [0.3, 0.1])
#x0, y0 = data_test[0]
#x1, y1 = data_test[1]
#y0 = tf.one_hot(y0, 2)
#y1 = tf.one_hot(y1, 2)
#x0 = tf.cast(x0, dtype = tf.float32)
#x1 = tf.cast(x1, dtype = tf.float32)
#data_test = [[x0, y0], [x1, y1]]

reg_wasserstein, reg_var, lr, gamma_wasserstein, wasserstein_epoch = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(float(sys.argv[5]))
num_steps = 120
sinkhorn_iter = 5
fitted_graph, current_time, expt_id = InvarLabelShift(data_train, data_test, num_steps=num_steps, 
                        reg_wasserstein=reg_wasserstein, reg_var = reg_var, learning_rate = lr, 
                        wasserstein_epoch = wasserstein_epoch, gamma_wasserstein = gamma_wasserstein, sinkhorn_iter = sinkhorn_iter)



accuracy = {'reg_wasserstein': reg_wasserstein, 'reg_var': reg_var, 'learning_rate': lr, 'datetime': current_time, 
            'num-steps': num_steps, 'wasserstein-epoch': wasserstein_epoch, 'sinkhorn-iteration': sinkhorn_iter}
accuracy['id'] = expt_id
accuracy['train'] = dict()
for index, data in enumerate(data_train):
    x, y = data[0], data[1]
    predict = fitted_graph(x, env = index, predict = True)
    accuracy['train'][index] = float(tf.reduce_mean(tf.cast(tf.equal(y[:,1], predict), dtype = tf.float32)))


accuracy['test'] = dict()
for index, data in enumerate(data_test):
    x, y = data[0], data[1]
    predict = fitted_graph(x, env = index, predict = True)
    accuracy['test'][index] = float(tf.reduce_mean(tf.cast(tf.equal(y[:,1], predict), dtype = tf.float32)))

with open('out_mnist3.json', 'a') as f:
    f.writelines(str(accuracy)+'\n')


