import tensorflow as tf
import sinkhorn as sh
import numpy as np
from setup import *
import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import json
from data_load import *



np.random.seed(1)
data_train, _ = make_environments(path = 'MNIST', red_0_corrs = [0.7, 0.9])
data_test, _ = make_environments(path = 'MNIST', red_0_corrs = [0.3, 0.1])

reg_wasserstein, reg_var, lr, gamma_wasserstein, wasserstein_epoch = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(float(sys.argv[5]))
num_steps = 12000
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

with open('out_mnist1.json', 'a') as f:
    f.writelines(str(accuracy)+'\n')


