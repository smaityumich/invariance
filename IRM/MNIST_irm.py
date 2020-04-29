import tensorflow as tf
import sinkhorn as sh
import numpy as np
import setup_irm
import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import json
from data_load import *

seed_id = np.random.randint(10000)

np.random.seed(seed_id)
data_train, data_test = make_environments(path='MNIST', red_0_corrs = [0.7, 0.9])

x0_train, y0_train = data_train[0]
x1_train, y1_train = data_train[1]
x_test, y_test = data_test

x0_train, x1_train = tf.cast(x0_train, dtype=tf.float32), tf.cast(x1_train, dtype=tf.float32)
y0_train, y1_train = tf.one_hot(y0_train, 2), tf.one_hot(y1_train, 2)
data_train = [[x0_train, y0_train], [x1_train, y1_train]]

x_test, y_test = tf.cast(x_test, dtype=tf.float32), tf.one_hot(y_test, 2)
data_test = [x_test, y_test]




# Parameter infos
'''Parameter infos:
reg_wasserstein: (float) regularization parameter for wasserstein regularizer
reg_var: (float) regularization parameter for l_2 regularizer in loss
lr: (float) learning rate
gamma_wasserstein: (float) gamma parameter in sinkhorn algorithm
wasserstein_epoch: (int) epoch interval at which wasserstin regularizer is activated
sinkhorn_iter: (int) iteration in sinkhorn algorithm'''

if len(sys.argv) > 7:
    reg_wasserstein, reg_var, lr, gamma_wasserstein, wasserstein_epoch, sinkhorn_iter, filename, num_steps = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(float(sys.argv[5])), int(float(sys.argv[6])), sys.argv[7], int(sys.argv[8])
    normalize = sys.argv[9]=='True' or sys.argv[9]=='true'
elif len(sys.argv) > 6:
    reg_wasserstein, reg_var, lr, gamma_wasserstein, wasserstein_epoch, sinkhorn_iter, filename, num_steps = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(float(sys.argv[5])), int(float(sys.argv[6])), sys.argv[7], int(sys.argv[8])
    normalize = False
else:
    reg_wasserstein, reg_var, lr, gamma_wasserstein, wasserstein_epoch, sinkhorn_iter, filename = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(float(sys.argv[5])), int(float(sys.argv[6])), sys.argv[7]
    num_steps = 10000
    normalize = False

#sinkhorn_iter = 5
fitted_graph, current_time, expt_id = setup_irm.IRM(data_train, data_test, num_steps=num_steps, 
                        reg_wasserstein=reg_wasserstein, reg_var = reg_var, learning_rate = lr, 
                        wasserstein_epoch = wasserstein_epoch, gamma_wasserstein = gamma_wasserstein, sinkhorn_iter = sinkhorn_iter, wasserstein_start_step= 400, normalize=normalize)



accuracy = {'reg_wasserstein': reg_wasserstein, 'reg_var': reg_var, 'learning_rate': lr, 'datetime': current_time, 
            'num-steps': num_steps, 'wasserstein-epoch': wasserstein_epoch, 'sinkhorn-iteration': sinkhorn_iter}
accuracy['id'] = expt_id
accuracy['seed'] = seed_id
accuracy['train'] = dict()
for index, data in enumerate(data_train):
    x, y = data[0], data[1]
    predict = fitted_graph(x, predict = True)
    accuracy['train'][index] = float(tf.reduce_mean(tf.cast(tf.equal(y[:,1], predict), dtype = tf.float32)))


accuracy['test'] = dict()
x, y = data_test[0], data_test[1]
predict = fitted_graph(x, predict = True)
accuracy['test'] = float(tf.reduce_mean(tf.cast(tf.equal(y[:,1], predict), dtype = tf.float32)))

with open(filename, 'a') as f:
    f.writelines(str(accuracy)+'\n')


