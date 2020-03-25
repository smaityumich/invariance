import tensorflow as tf
from tensorflow.keras import Model, layers
#import ot
import numpy as np

class InvarianceTestGraph(Model, layers.Layer):
    
    def __init__(self):
        super(InvarianceTestGraph, self).__init__()
        
        self.weight = {'weight1': self.add_weight(shape=(64, 32), initializer='random_normal', trainable=True), 
                        'weight2': self.add_weight(shape=(32, 16), initializer='random_normal', trainable=True), 
                        'weight3': self.add_weight(shape=(16, 2), initializer='random_normal', trainable=True)}
        
        self.bias = {'bias1': self.add_weight(shape=(32, ), initializer='random_normal', trainable=True), 
                        'bias2': self.add_weight(shape=(16, ), initializer='random_normal', trainable=True), 
                        'bias3': self.add_weight(shape=(2, ), initializer='random_normal', trainable=True)}
        
    def invariantMap(self, x):
        out = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weight['weight1']), self.bias['bias1']))
        out = tf.nn.sigmoid(tf.add(tf.matmul(out, self.weight['weight2']), self.bias['bias2']))
        out = tf.nn.sigmoid(tf.add(tf.matmul(out, self.weight['weight3']), self.bias['bias3']))
        return out


    def call(self, x):
        return self.invariantMap(x)
    
inv = InvarianceTestGraph()
    
y = np.random.binomial(1, 0.5, (1200,))
f = lambda y: np.random.normal(1, 1, (64,)) if y else np.random.normal(0, 1, (64,))
x = [f(i) for i in y]
y = tf.one_hot(y)
batch = tf.data.Dataset.from_tensor_slices((x, y))
batch = batch.repeat().shuffle(5000).batch(120).prefetch(1)


learningRate = 0.01
optimizer = tf.optimizers.Adam(learningRate)


for step, (batch_x, batch_y) in enumerate(batch.take(200), 1):
    with tf.GradientTape() as g:
        loss = tf.nn.softmax_cross_entropy_with_logits(batch_y, inv(batch_x))
        
    trainable_variables = inv.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    

        
    
    


