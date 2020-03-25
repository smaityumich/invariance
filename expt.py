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
inv = InvarianceNNGraph()


class InvaranceLabelShift():
    
    def __init__(self, data, batch_size = 150, num_steps = 2000, learning_rate = 0.01):
        self.graph = InvaranceNNGraph()
        self.data = data
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.batch_data = []
        for env in self.data:
            batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
            batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
            self.batch_data.append(batch.take(num_steps))
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        
    def run_optimizer(self):
        for step, data in enumerate(zip(*self.batch_data), 1):
                with tf.GradientTape() as g:
                    loss = tf.cast(0, dtype = tf.float32)
                    for index, (x, y) in enumerate(data):
                        probs = inv(x, env = index)
                        loss = loss + EntropyLoss(y, probs)
                    for label in [0,1]:
                        conditional_data = [env[0][env[1][:, 1] == label] for env in data]
                        loss = loss + sh.sinkhorn_dist(conditional_data[0], conditional_data[1], 2, 5)
                    variables = self.graph.trainable_variables
                    for v in variables:
                        loss = loss + tf.norm(v)
                trainable_variables = self.graph.trainable_variables
                gradients = g.gradient(loss, trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        