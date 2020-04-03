import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow import keras




class InvarianceNNGraph(keras.Model, keras.layers.Layer):
    
    def __init__(self):
        super(InvarianceNNGraph, self).__init__()
        self.weight = {'weight1': self.add_weight(shape=(8, 32), initializer='random_normal', trainable=True), 
                        'weight2': self.add_weight(shape=(32, 16), initializer='random_normal', trainable=True), 
                        'weight3': self.add_weight(shape=(16, 3), initializer='random_normal', trainable=True),
                        'weight_final': self.add_weight(shape = (3,1), initializer = 'random_normal', trainable = True)}
        self.bias = {'bias1': self.add_weight(shape=(32, ), initializer='random_normal', trainable=True), 
                        'bias2': self.add_weight(shape=(16, ), initializer='random_normal', trainable=True), 
                        'bias3': self.add_weight(shape=(3, ), initializer='random_normal', trainable=True),
                        'bias_final0': self.add_weight(shape = (1,), initializer = 'random_normal', trainable = True), 
                        'bias_final1': self.add_weight(shape = (1,), initializer = 'random_normal', trainable = True)}
        
    def invariant_map(self, x):
        out = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weight['weight1']), self.bias['bias1']))
        out = tf.nn.sigmoid(tf.add(tf.matmul(out, self.weight['weight2']), self.bias['bias2']))
        out = tf.nn.sigmoid(tf.add(tf.matmul(out, self.weight['weight3']), self.bias['bias3']))
        return out
        
    def call(self, x, env = 0, predict = False):
        if env == 0:
            out = tf.add(tf.matmul(self.invariant_map(x), self.weight['weight_final']) , self.bias['bias_final0'])
            out = tf.concat([-out, out], axis = 1)
            out = tf.nn.softmax(out)
            return tf.cast(tf.argmax(out, axis = 1), dtype = tf.float32) if predict else out
            
        elif env == 1:
            out = tf.add(tf.matmul(self.invariant_map(x), self.weight['weight_final']), self.bias['bias_final1'])
            out = tf.concat([-out, out], axis = 1)
            out = tf.nn.softmax(out)
            return tf.cast(tf.argmax(out, axis = 1), dtype = tf.float32) if predict else out




def EntropyLoss(y, prob):
    #prob = tf.concat([1-prob, prob], axis = 1)
    #y = tf.one_hot(y, 2)
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))


    


