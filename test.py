<<<<<<< HEAD
import numpy as np
=======
import tensorflow as tf
from expt import *
import numpy as np


y0 = np.random.binomial(1, 0.5, (1200,))
y1 = np.random.binomial(1, 0.8, (1500,))
f = lambda y: np.random.normal(1, 1, (64,)) if y else np.random.normal(0, 1, (64,))
x0 = [f(y) for y in y0]
x1 = [f(y) for y in y1]
y0 = tf.one_hot(y0, 2)
y1 = tf.one_hot(y1, 2)
x0 = tf.cast(x0, dtype=tf.float32)
x1 = tf.cast(x1, dtype = tf.float32)

data_train = [[x0, y0], [x1, y1]]

y0 = np.random.binomial(1, 0.5, (1000,))
y1 = np.random.binomial(1, 0.8, (1000,))
f = lambda y: np.random.normal(1, 1, (64,)) if y else np.random.normal(0, 1, (64,))
x0 = [f(y) for y in y0]
x1 = [f(y) for y in y1]
y0 = tf.one_hot(y0, 2)
y1 = tf.one_hot(y1, 2)
x0 = tf.cast(x0, dtype=tf.float32)
x1 = tf.cast(x1, dtype = tf.float32)

data_test = [[x0, y0], [x1, y1]]

inv = InvaranceLabelShift(data_train, data_test)


        
    
    


>>>>>>> b4b309bb25a4d3afeb3729fe44f64317871b37b1
