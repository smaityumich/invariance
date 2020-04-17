import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# from sklearn.preprocessing import OneHotEncoder

def read_idx(filename, flatten=True):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        if len(shape)==3 and flatten:
            shape = (shape[0], shape[1]*shape[2])
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    
def read_lecun_mnist(path=None, flatten=True, size = 12000):
    if path is None:
        path = '/Users/moonfolk/Desktop/Yuekai/valley_hypothesis/mnist_data/'
    # Download data from here http://yann.lecun.com/exdb/mnist/
    x = read_idx(path + '/train-images-idx3-ubyte', flatten)
    y = read_idx(path + '/train-labels-idx1-ubyte', flatten)
    
    # scale_const = X_train.max()
    scale_const = 255
    x = x.astype(float)/scale_const
    
    # one_hot = OneHotEncoder(sparse=False)
    # one_hot.fit(y_train.reshape(-1,1))
    # y_train = one_hot.transform(y_train.reshape(-1,1))
    # y_test = one_hot.transform(y_test.reshape(-1,1))
    
    idx = np.random.choice(x.shape[0], size=size, replace=False)
    x, y = x[idx], y[idx]
    y = binarize(y)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n0, n1 = len(idx0), len(idx1)
    n0_train, n1_train = int(np.ceil(n0*10000/12000)), int(np.ceil(n1*10000/12000))
    idx0_train, idx0_test = idx0[:n0_train], idx0[n0_train:]
    idx1_train, idx1_test = idx1[:n1_train], idx1[n1_train:]
    idx_train, idx_test = list(idx0_train) + list(idx1_train), list(idx0_test) + list(idx1_test)
    idx_train, idx_test = np.random.permutation(idx_train), np.random.permutation(idx_test)



    return x[idx_train], y[idx_train], x[idx_test], y[idx_test] #x[idx], y[idx] #, X_test, y_test
    # return X_train, y_train, X_test, y_test

def binarize(y, label_noise=0.25):
    
    y = np.copy(y) > 4
    y = np.logical_xor(y, np.random.binomial(1, label_noise, size=len(y)))
    
    return y.astype(int)
    
def color_digits(X, y, color_noise, downsample=True):
    
    if downsample:
        X = np.copy(X)[:,::2,::2]
    
    color = np.logical_xor(y, np.random.binomial(1, color_noise, size=len(y)))
    colored_X = np.repeat(X[:,None,:,:],2,axis=1)
    colored_X[color,0,:,:] = 0
    colored_X[~color,1,:,:] = 0
    
    colored_X = colored_X.reshape(X.shape[0],-1)
    
    return colored_X


def make_environment(data=None, downsample=True, path=None, red_0_corrs = None):
    if data is None:
        x_train, y_train, x_test, y_test = read_lecun_mnist(path=path, flatten=False)
    else:
        x_train, y_train, x_test, y_test = data
    
    if red_0_corrs == None: 
        red_0_corrs = [0.1, 0.9]
    
    
    
    return color_digits(x_train, y_train, red_0_corrs[0]), y_train, color_digits(x_test, y_test, red_0_corrs[1]), y_test