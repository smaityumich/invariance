import tensorflow as tf
from tensorflow import keras

class NeuralNet(keras.Model):

    """
    Neural net graph
    """

    def __init__(self, architecture = [20, 10, 10, 2], activation = tf.nn.relu):

        if type(architecture) != list:
            raise TypeError('Architecture is not a list.')
        else:
            self.LAYERS = [] 
            if len(architecture) == 1:
                self.LAYERS.append(keras.layers.Dense(architecture[0], name = 'output-layer'))
            else:
                for i, nodes in enumerate(architecture[:-1]):
                    self.LAYERS.append(\
                        keras.layers.Dense(nodes, activation = activation, name = f'layer-{i}'))
                
                self.LAYERS.append(keras.layers.Dense(architecture[-1], name = 'output-layer'))

        self.model = keras.Sequential(self.LAYERS)

    def call(self, x):
        return self.model(x)


def NNGraph(architecture = [20, 10, 10, 2], activation = tf.nn.relu, seed = 0):


    if type(architecture) != list:
            raise TypeError('Architecture is not a list.')
    else:
        tf.random.set_seed(seed)
        LAYERS = [] 
        if len(architecture) == 1:
            LAYERS.append(keras.layers.Dense(architecture[0], name = 'output-layer'))
        else:
            for i, nodes in enumerate(architecture[:-1]):
                LAYERS.append(\
                        keras.layers.Dense(nodes, activation = activation, name = f'layer-{i}'))
                
            LAYERS.append(keras.layers.Dense(architecture[-1], name = 'output-layer'))

        model = keras.Sequential(LAYERS)

        return model


def ConvexGraph(architecture = [20, 10, 10, 2], activation = tf.nn.relu, seed = 0, name = 'CNN'):


    if type(architecture) != list:
            raise TypeError('Architecture is not a list.')
    else:
        tf.random.set_seed(seed)
        kernel_constraint = keras.constraints.NonNeg()
        kernel_initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.1, seed = 1)
        LAYERS = [] 
        if len(architecture) == 1:
            LAYERS.append(keras.layers.Dense(architecture[0],\
                kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer, name = 'output-layer'))
        else:
            for i, nodes in enumerate(architecture[:-1]):
                LAYERS.append(\
                        keras.layers.Dense(nodes, activation = activation,\
                            kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer, name = f'layer-{i}'))
                
            LAYERS.append(keras.layers.Dense(architecture[-1],\
                kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer, name = 'output-layer'))

        model = keras.Sequential(LAYERS, name = name)

        return model





@tf.function
def probabilities(logits):
    return tf.nn.softmax(logits)

@tf.function
def entropy(logits, y):
    return - tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.nn.softmax(logits) * y, axis = 1)))

@tf.function
def predict(logits):
    return tf.cast(tf.argmax(logits, axis = 1), dtype = tf.float32)

@tf.function
def accuracy(logits, y):
    predict = tf.cast(tf.argmax(logits, axis = 1), dtype = tf.float32)
    return tf.reduce_mean(tf.cast(predict == y[:, 1], dtype = tf.float32))


@tf.function
def covar(logits, y):
    prob = probabilities(logits)[:, 1]
    labels = y[:, 1]
    return tf.reduce_mean(prob * labels) - tf.reduce_mean(prob) * tf.reduce_mean(labels)

def covariance(x, y):
    return tf.reduce_mean(x * y) - tf.reduce_mean(x) * tf.reduce_mean(y)
