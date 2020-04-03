# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:40:55 2020

@author: maity
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np




class InvarianceNNGraph(keras.Model, keras.layers.Layer):
    
    def __init__(self):
        super(InvarianceNNGraph, self).__init__()
        initializer = 'random_normal'
        self.weight = {'weight1': self.add_weight(shape=(8, 32), initializer=initializer, trainable=True), 
                        'weight2': self.add_weight(shape=(32, 16), initializer=initializer, trainable=True), 
                        'weight3': self.add_weight(shape=(16, 16), initializer=initializer, trainable=True), 
                        'weight4': self.add_weight(shape=(16, 1), initializer=initializer, trainable=True)}
        self.bias = {'bias1': self.add_weight(shape=(32, ), initializer=initializer, trainable=True), 
                        'bias2': self.add_weight(shape=(16, ), initializer=initializer, trainable=True), 
                        'bias3': self.add_weight(shape=(16, ), initializer=initializer, trainable=True),
                        'bias4': self.add_weight(shape=(1, ), initializer=initializer, trainable=True),
                        'bias_final0': self.add_weight(shape = (1,), initializer = initializer, trainable = True), 
                        'bias_final1': self.add_weight(shape = (1,), initializer = initializer, trainable = True)}
        
    def invariant_map(self, x):
        out = tf.nn.relu(tf.add(tf.matmul(x, self.weight['weight1']), self.bias['bias1']))
        out = tf.nn.relu(tf.add(tf.matmul(out, self.weight['weight2']), self.bias['bias2']))
        #out = tf.nn.sigmoid(tf.add(tf.matmul(out, self.weight['weight3']), self.bias['bias3']))
        out = tf.nn.relu(tf.add(tf.matmul(out, self.weight['weight3']), self.bias['bias3']))
        out = tf.nn.relu(tf.add(tf.matmul(out, self.weight['weight4']), self.bias['bias4']))
        #out = tf.math.log(out+1)
        #out = tf.math.log(out+1)
        out = (out-tf.math.reduce_mean(out))/tf.math.reduce_std(out)
        return out
        
    def call(self, x, env = 0, predict = False):
        out = self.invariant_map(x)
        #out = tf.math.log(out+1)
        if env == 0:
            out = tf.add(out, self.bias['bias_final0'])
            out = tf.concat([-out, out], axis = 1)
            out = tf.nn.softmax(out)
            return tf.cast(tf.argmax(out, axis = 1), dtype = tf.float32) if predict else out
            
        elif env == 1:
            out = tf.add(self.invariant_map(x), self.bias['bias_final1'])
            out = tf.concat([-out, out], axis = 1)
            out = tf.nn.softmax(out)
            return tf.cast(tf.argmax(out, axis = 1), dtype = tf.float32) if predict else out




def EntropyLoss(y, prob):
    #prob = tf.concat([1-prob, prob], axis = 1)
    #y = tf.one_hot(y, 2)
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob+1e-16)))


    


