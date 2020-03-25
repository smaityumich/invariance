# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:34:10 2020

@author: maity
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
import sinkhorn as sh
import numpy as np
from utils import *
import datetime


class InvaranceLabelShift():
    
    def __init__(self, data_train, data_test, batch_size = 150, num_steps = 2000, learning_rate = 0.01, reg_wasserstein = 0.5, gamma_wasserstein = 2, reg_var = 0.2, sinkhorn_iter = 5):
        self.graph = InvaranceNNGraph()
        self.data_train = data_train
        self.data_test = data_test
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.reg_wasserstein = reg_wasserstein
        self.reg_var = reg_var
        self.sinkhorn_iter = sinkhorn_iter
        self.gamma_wasserstein = gamma_wasserstein
        
        self.batch = []
        for env in self.data_train:
            batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
            batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
            self.batch.append(batch.take(num_steps))
            
        for env in self.data_test:
            batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
            batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
            self.batch.append(batch.take(num_steps))
            
        
            
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.Mean('train_accuracy', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.Mean('test_accuracy', dtype=tf.float32)
        
        
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.parameter = f'num_steps_{self.num_steps}_lr_{self.learning_rate}_reg_wasserstein_{self.reg_wasserstein}_reg_var_{self.reg_var}_gamma_wasserstein_{self.gamma_wasserstein}_sinkhorn_iter_{self.sinkhorn_iter}'
        self.train_log_dir = 'logs/gradient_tape/' + self.parameter + '/train'
        self.test_log_dir = 'logs/gradient_tape/' + self.parameter + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
        
        
    def _accuracy(self, y, ypred):
        acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
        return tf.reduce_mean(acc)
        
    def train_step(self, data_train):
        with tf.GradientTape() as g:
            loss = tf.cast(0, dtype = tf.float32)
            for index, (x, y) in enumerate(data_train):
                probs = self.graph(x, env = index)
                loss = loss + EntropyLoss(y, probs)
            loss_train = loss
            for label in [0,1]:
                conditional_data = [env[0][env[1][:, 1] == label] for env in data_train]
                loss = loss + self.reg_wasserstein*sh.sinkhorn_dist(conditional_data[0], conditional_data[1], self.gamma_wasserstein, self.sinkhorn_iter)
            variables = self.graph.trainable_variables
            for v in variables:
                loss = loss + self.reg_var*tf.norm(v)
            trainable_variables = self.graph.trainable_variables
            gradients = g.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            
            self.train_loss(loss_train)
            
            accuracy_train = tf.cast(0, dtype = tf.float32)
            for index, (x, y) in enumerate(data_train):
                predict = self.graph(x, env = index, predict = True)
                accuracy_train = accuracy_train + self._accuracy(y, predict)
            self.train_accuracy(accuracy_train)
            
            
    def test_step(self, data_test):
        loss = tf.cast(0, dtype = tf.float32)
        for index, (x, y) in enumerate(data_test):
            probs = self.graph(x, env = index)
            loss = loss + EntropyLoss(y, probs)
        self.test_loss(loss)
            
        accuracy_test = tf.cast(0, dtype = tf.float32)
        for index, (x, y) in enumerate(data_test):
            predict = self.graph(x, env = index, predict = True)
            accuracy_test = accuracy_test + self._accuracy(y, predict)
        self.test_accuracy(accuracy_test)
            
            
        
    def run_optimizer(self):
        for step, data in enumerate(zip(*self.batch_data), 1):
            data_train = data[:2]
            data_test = data[2:]
            self.train_step(data_train)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=step)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)
                
            self.test_step(data_test)
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=step)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)
                
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
                
        