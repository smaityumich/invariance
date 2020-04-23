import tensorflow as tf
import sinkhorn as sh
import numpy as np
import irm_simple as nn_graph
import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import json





def IRM(data_train, data_test, batch_size = 1500, num_steps = 2500, 
                    learning_rate = 1e-5, 
                    reg_wasserstein = 0, wasserstein_epoch = 1, 
                    gamma_wasserstein = 1e-2, wasserstein_start_step = 1000,
                    reg_var = 10, sinkhorn_iter = 5, clip_grad = 40):


    '''
    Invariant Risk Minimization: finds invariant representation for classification model
    
    Input: 
        data_train: list of two environments where each environment is a list [x, y]
        data_test: list [x, y]
        batch_size: (int) batch size for the environments
        num_steps: (int) number of training epochs
        learning_rate: (float) learining rate 
        reg_wasserstein: (float) regularization parameter for wasserstein regularizer
        wasserstein_epoch: (int) epoch interval at which wasserstin regularizer is activated
        gamma_wasserstein: (float) gamma parameter in sinkhorn algorithm
        wasserstein_start_step: (int) epoch at which wasserstein regularized is activated for the first time
        reg_var: (float) regularization parameter for l_2 regularizer in loss
        sinkhorn_iter: (int) iteration in sinkhorn algorithm
        clip_grad: (float) Clipping value for clip_by_norm for gradients

    Output: 
        graph: fitted NN graph
        current_time: start time of experiment
        expt_id: a random integer as experiment id
    '''
    graph = nn_graph.InvarianceNNGraph()
    batch_data = []
    for env in data_train:
        batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
        batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        batch_data.append(batch.take(num_steps))
            
    
    batch = tf.data.Dataset.from_tensor_slices((data_test[0], data_test[1]))
    batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    batch_data.append(batch.take(num_steps))

    optimizer = tf.optimizers.Adam(learning_rate)


    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.Mean('test_accuracy', dtype=tf.float32)
    var_norm = tf.keras.metrics.Mean('total_norm', dtype=tf.float32)
    grad_norm = tf.keras.metrics.Mean('Gradient_norm', dtype=tf.float32)
    train_accuracy_env = dict()
    for env in [0,1]:
        train_accuracy_env[env] = tf.keras.metrics.Mean('train_accuracy_env:'+str(env), dtype=tf.float32)
        
    train_wasserstein_y = dict()
    for y in [0,1]:
        train_wasserstein_y[y] = tf.keras.metrics.Mean('train_wasserstein_y:'+str(y), dtype=tf.float32)

    
        
    

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    expt_id = np.random.randint(1000000)
    parameter = f'_expt-id_{expt_id}_num_steps_{num_steps}_lr_{learning_rate}_reg_wasserstein_{reg_wasserstein}_wasserstein_epoch_{wasserstein_epoch}_reg_var_{reg_var}_gamma_wasserstein_{gamma_wasserstein}_sinkhorn_iter_{sinkhorn_iter}'
    train_log_dir = 'logs/' + current_time + parameter + '/train'
    test_log_dir = 'logs/' + current_time + parameter + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    
    def _accuracy(y, ypred):
        acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
        return tf.reduce_mean(acc)

    
    
    def train_step(data_train_epoch, data_train_wasserstein, full_data, step):

        with tf.GradientTape() as g:
            loss = tf.cast(0, dtype = tf.float32)
            for index, (x, y) in enumerate(data_train_epoch):
                probs = graph(x)
                loss = loss + nn_graph.EntropyLoss(y, probs)
            loss_train = loss
            WD = [0,0]
            if step % wasserstein_epoch == 0:
                for label in [0,1]:
                    conditional_data = [env[0][env[1][:, 1] == label] for env in data_train_wasserstein]
                    wasserstein_dist = sh.sinkhorn_dist(graph.invariant_map(conditional_data[0]), 
                                                                   graph.invariant_map(conditional_data[1]), 
                                                                   gamma_wasserstein, sinkhorn_iter)
                    train_wasserstein_y[label](wasserstein_dist)
                    WD[label] = wasserstein_dist
                    if step > wasserstein_start_step:
                        loss = loss + reg_wasserstein*wasserstein_dist
                    
                
                    
            variables = graph.trainable_variables
            norm = tf.cast(0, dtype = tf.float32)
            for v in variables:
                norm = norm + tf.norm(v)
            
                
            loss = loss + reg_var*(norm)
            
        trainable_variables = graph.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        clipped_grad = [tf.clip_by_norm(g, clip_grad) for g in gradients]
        optimizer.apply_gradients(zip(clipped_grad, trainable_variables))
        norm_grad = tf.cast(0, dtype = tf.float32)    
        for v in gradients:
            norm_grad = norm_grad + tf.norm(v, ord = 2)
            
        train_loss(loss_train)
        var_norm(norm**2)
        grad_norm((learning_rate*norm_grad))
            
        for index, (x, y) in enumerate(data_train_epoch):
            predict = graph(x, predict = True)
            accuracy_train_env = _accuracy(y[:,1], predict)
            train_accuracy_env[index](accuracy_train_env)

        logits = dict()
        for index, (x, y) in enumerate(full_data):
            logits[index] = graph(x)[:, 1]
        return WD[0], WD[1], logits
            
    def test_step(data_test_epoch, full_data, step):
        loss = tf.cast(0, dtype = tf.float32)
        x, y = data_test_epoch[0], data_test_epoch[1]
        probs = graph(x)
        loss = nn_graph.EntropyLoss(y, probs)
        test_loss(loss)
            
        
        predict = graph(x, predict = True)
        accuracy_test = _accuracy(y[:,1], predict)
        test_accuracy(accuracy_test)

        x, y = full_data
        return graph(x)[:, 1]

    for step, data in enumerate(zip(*batch_data), 1):
        batch_data_train = data[:2]
        batch_data_test = data[2]
        #_, _ = train_step(batch_data_train, data_train, step) # If using full data to calculate wasserstein distance
        _, _, logits = train_step(batch_data_train, batch_data_train, data_train, step) # If using batch data to calculate wasserstein distance

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=step)
            for env in [0,1]:
                tf.summary.scalar('accuracy-train-env:'+str(env), train_accuracy_env[env].result(), step = step)
                tf.summary.histogram('train-data/train-logit-hist-env-'+str(env), data = logits[env], step = step)
                
            for y in [0,1]:
                tf.summary.scalar('wasserstein-train-y:'+str(y), train_wasserstein_y[y].result(), step = step)  
            
            tf.summary.scalar('norm', var_norm.result(), step = step)
            tf.summary.scalar('gradient-norm', grad_norm.result(), step = step) 
        
        
        test_probs = test_step(batch_data_test, data_test, step)
      
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=step)
            tf.summary.scalar('test-accuracy', test_accuracy.result(), step=step)
            tf.summary.histogram('test-data/logit-histogram', data = test_probs, step = step)
              
                
              
                
        train_loss.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        var_norm.reset_states()
        grad_norm.reset_states()
        
        for env in [0,1]:
            train_accuracy_env[env].reset_states()
        if step % wasserstein_epoch == 0:    
            for y in [0,1]:
                train_wasserstein_y[y].reset_states()
            
        if step % 100 == 0:
            print(f'Done step {step}\n')
            
    return graph, current_time, expt_id

