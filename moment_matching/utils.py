import tensorflow as tf

def moment(x, order = 2):
    mean = tf.math.reduce_mean(x)
    return tf.reduce_mean(tf.math.abs(x-mean)**order)


def _accuracy(y, ypred):
        acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
        return tf.reduce_mean(acc)


def EntropyLoss(y, prob):
    #prob = tf.concat([1-prob, prob], axis = 1)
    #y = tf.one_hot(y, 2)
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob+1e-16)))