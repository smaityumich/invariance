# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:29:10 2020

@author: maity
"""

import tensorflow as tf
import sinkhorn as sh
import numpy as np
import utils2 as utils
import datetime
from tensorflow import keras


# Simulated Data

## Environment 1
y0 = np.random.binomial(1, 0.5, (1200,))
y1 = np.random.binomial(1, 0.8, (1500,))
f = lambda y: np.random.normal(1, 1, (4,)) if y else np.random.normal(0, 1, (4,))
x0 = [f(y) for y in y0]
x1 = [f(y) for y in y1]
x0 = np.concatenate((x0, np.random.normal(2, 1, (1200, 60))), axis = 1)
x1 = np.concatenate((x1, np.random.normal(2, 1, (1500, 60))), axis = 1)
y0 = tf.one_hot(y0, 2)
y1 = tf.one_hot(y1, 2)
x0 = tf.cast(x0, dtype = tf.float32)
x1 = tf.cast(x1, dtype = tf.float32)
data_train = [[x0, y0], [x1, y1]]


## Environment 2
y0 = np.random.binomial(1, 0.5, (1000,))
y1 = np.random.binomial(1, 0.8, (1000,))
f = lambda y: np.random.normal(1, 1, (4,)) if y else np.random.normal(0, 1, (4,))
x0 = [f(y) for y in y0]
x1 = [f(y) for y in y1]
x0 = np.concatenate((x0, np.random.normal(2, 1, (1000, 60))), axis = 1)
x1 = np.concatenate((x1, np.random.normal(2, 1, (1000, 60))), axis = 1)
y0 = tf.one_hot(y0, 2)
y1 = tf.one_hot(y1, 2)
x0 = tf.cast(x0, dtype=tf.float32)
x1 = tf.cast(x1, dtype = tf.float32)
data_test = [[x0, y0], [x1, y1]]



def InvarLabelShift(data_train, data_test, batch_size = 100, num_steps = 2500, learning_rate = 0.01, reg_wasserstein = 0.5, gamma_wasserstein = 2, reg_var = 0.01, sinkhorn_iter = 10):
    graph = utils.InvarianceNNGraph()
    batch_data = []
    for env in data_train:
        batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
        batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        batch_data.append(batch.take(num_steps))
            
    for env in data_test:
        batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
        batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        batch_data.append(batch.take(num_steps))

    optimizer = tf.optimizers.Adam(learning_rate)


    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.Mean('train_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.Mean('test_accuracy', dtype=tf.float32)


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parameter = f'_num_steps_{num_steps}_lr_{learning_rate}_reg_wasserstein_{reg_wasserstein}_reg_var_{reg_var}_gamma_wasserstein_{gamma_wasserstein}_sinkhorn_iter_{sinkhorn_iter}'
    train_log_dir = 'logs/gradient_tape/' + current_time + parameter + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + parameter + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    
    def _accuracy(y, ypred):
        acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
        return tf.reduce_mean(acc)

    
    def train_step(data_train, full_data, step):
        with tf.GradientTape() as g:
            loss = tf.cast(0, dtype = tf.float32)
            for index, (x, y) in enumerate(data_train):
                probs = graph(x, env = index)
                loss = loss + utils.EntropyLoss(y, probs)
            loss_train = loss
            if step % 50 ==0:
                for label in [0,1]:
                    conditional_data = [env[0][env[1][:, 1] == label] for env in full_data]
                    loss = loss + reg_wasserstein*sh.sinkhorn_dist(conditional_data[0], conditional_data[1], gamma_wasserstein, sinkhorn_iter)
            variables = graph.trainable_variables
            for v in variables:
                loss = loss + reg_var*tf.norm(v)
            trainable_variables = graph.trainable_variables
            gradients = g.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            
            train_loss(loss_train)
            
            accuracy_train = tf.cast(0, dtype = tf.float32)
            for index, (x, y) in enumerate(data_train):
                predict = graph(x, env = index, predict = True)
                accuracy_train = accuracy_train + _accuracy(y[:,1], predict)
            accuracy_train = accuracy_train/2
            train_accuracy(accuracy_train)
            
    def test_step(data_test):
        loss = tf.cast(0, dtype = tf.float32)
        for index, (x, y) in enumerate(data_test):
            probs = graph(x, env = index)
            loss = loss + utils.EntropyLoss(y, probs)
        test_loss(loss)
            
        accuracy_test = tf.cast(0, dtype = tf.float32)
        for index, (x, y) in enumerate(data_test):
            predict = graph(x, env = index, predict = True)
            accuracy_test = accuracy_test + _accuracy(y[:,1], predict)
        accuracy_test = accuracy_test/2
        test_accuracy(accuracy_test)

    for step, data in enumerate(zip(*batch_data), 1):
        batch_data_train = data[:2]
        batch_data_test = data[2:]
        train_step(batch_data_train, data_train, step)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=step)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
                
        test_step(batch_data_test)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=step)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=step)
                
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        if step % 100 == 0:
            print(f'Done step {step}\n')
            
    return graph


graph = InvarLabelShift(data_train, data_test, num_steps=2500)


## tensorboard --logdir logs/gradient_tape for tensorboard

