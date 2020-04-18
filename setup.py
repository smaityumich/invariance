import tensorflow as tf
import sinkhorn as sh
import numpy as np
import nn_graph_logistic as nn_graph
import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import json





def InvarLabelShift(data_train, data_test, batch_size = 250, num_steps = 2500, 
                    learning_rate = 1e-5, 
                    reg_wasserstein = 0, wasserstein_epoch = 1, 
                    gamma_wasserstein = 1e-2, wasserstein_start_step = 2000,
                    reg_var = 10, sinkhorn_iter = 5, clip_grad = 40):


    '''
    Invariant Label Shift: finds invariant representation for (generalized) label shift model
    
    Input: 
        data_train: list of two environments where each environment is a list [x, y]
        data_test: same as data_train; but with test data
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
            
    for env in data_test:
        batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
        batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        batch_data.append(batch.take(num_steps))

    optimizer = tf.optimizers.Adam(learning_rate)


    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.Mean('train_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.Mean('test_accuracy', dtype=tf.float32)
    test_accuracy_env0 = tf.keras.metrics.Mean('train_accuracy_env:'+str(env), dtype=tf.float32)
    test_accuracy_env1 = tf.keras.metrics.Mean('train_accuracy_env:'+str(env), dtype=tf.float32)
    var_norm = tf.keras.metrics.Mean('total_norm', dtype=tf.float32)
    grad_norm = tf.keras.metrics.Mean('Gradient_norm', dtype=tf.float32)
    train_accuracy_env = dict()
    test_accuracy_env = dict()
    for env in [0,1]:
        train_accuracy_env[env] = tf.keras.metrics.Mean('train_accuracy_env:'+str(env), dtype=tf.float32)
        test_accuracy_env[env] = tf.keras.metrics.Mean('test_accuracy_env:'+str(env), dtype=tf.float32)
        
    train_wasserstein_y = dict()
    test_wasserstein_y = dict()
    for y in [0,1]:
        train_wasserstein_y[y] = tf.keras.metrics.Mean('train_wasserstein_y:'+str(y), dtype=tf.float32)
        test_wasserstein_y[y] = tf.keras.metrics.Mean('test_wasserstein_y:'+str(y), dtype=tf.float32)
        
    

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

    
    
    def train_step(data_train_epoch, full_data, step):
        with tf.GradientTape() as g:
            loss = tf.cast(0, dtype = tf.float32)
            for index, (x, y) in enumerate(data_train_epoch):
                probs = graph(x, env = index)
                loss = loss + nn_graph.EntropyLoss(y, probs)
            loss_train = loss
            WD = [0,0]
            if step % wasserstein_epoch == 0 and step > wasserstein_start_step:
                for label in [0,1]:
                    conditional_data = [env[0][env[1][:, 1] == label] for env in full_data]
                    wasserstein_dist = sh.sinkhorn_dist(graph.invariant_map(conditional_data[0]), 
                                                                   graph.invariant_map(conditional_data[1]), 
                                                                   gamma_wasserstein, sinkhorn_iter)
                    train_wasserstein_y[label](wasserstein_dist)
                    #print(f'WD train y:{label} at step {step} is {wasserstein_dist}\n')
                    WD[label] = wasserstein_dist
                    
                    loss = loss + reg_wasserstein*wasserstein_dist
                    
                
                    
            #else:
            #    for label in [0,1]:
            #        train_wasserstein_y[y](tf.Variable(0, dtype = tf.float32))
                    
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
            
        accuracy_train = tf.cast(0, dtype = tf.float32)
        for index, (x, y) in enumerate(data_train_epoch):
            predict = graph(x, env = index, predict = True)
            #predict = tf.cast(tf.argmax(prob, axis = 1), dtype = tf.float32)
            #with train_summary_writer.as_default():
            #    figure = plt.hist(prob[:,1])
            #    cm_image = plot_to_image(figure)
            #    tf.summary.image("probability_hist_env:"+str(index), cm_image, step=epoch)
                
            accuracy_train_env = _accuracy(y[:,1], predict)
            accuracy_train = accuracy_train + accuracy_train_env
            train_accuracy_env[index](accuracy_train_env)
        accuracy_train = accuracy_train/2
        train_accuracy(accuracy_train)
        return WD[0], WD[1]
            
    def test_step(data_test_epoch, full_data, step):
        loss = tf.cast(0, dtype = tf.float32)
        for index, (x, y) in enumerate(data_test_epoch):
            probs = graph(x, env = index)
            loss = loss + nn_graph.EntropyLoss(y, probs)
        test_loss(loss)
            
        accuracy_test = tf.cast(0, dtype = tf.float32)
        for index, (x, y) in enumerate(data_test_epoch):
            predict = graph(x, env = index, predict = True)
            accuracy_test_env = _accuracy(y[:,1], predict)
            accuracy_test = accuracy_test + accuracy_test_env
            test_accuracy_env[index](accuracy_test_env)
        accuracy_test = accuracy_test/2
        test_accuracy(accuracy_test)
        WD = [0,0]
        if step % wasserstein_epoch == 0 and step > wasserstein_start_step:
                for label in [0,1]:
                    conditional_data = [env[0][env[1][:, 1] == label] for env in full_data]
                    wasserstein_dist = sh.sinkhorn_dist(graph.invariant_map(conditional_data[0]), 
                                                                   graph.invariant_map(conditional_data[1]), 
                                                                   gamma_wasserstein, sinkhorn_iter)
                    test_wasserstein_y[label](wasserstein_dist)
                    WD[label] = wasserstein_dist
                    
                    #print(f'WD test y:{label} at step {step} is {wasserstein_dist}\n')
        #else:
            #for label in [0,1]:
                #test_wasserstein_y[y](tf.Variable(0, dtype = tf.float32))
        return WD[0], WD[1]

    for step, data in enumerate(zip(*batch_data), 1):
        batch_data_train = data[:2]
        batch_data_test = data[2:]
        #wd0_tr, wd1_tr = train_step(batch_data_train, data_train, step)
        _, _ = train_step(batch_data_train, batch_data_train, step)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=step)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
            for env in [0,1]:
                tf.summary.scalar('accuracy_env:'+str(env), train_accuracy_env[env].result(), step = step)
                
            for y in [0,1]:
                tf.summary.scalar('wasserstein_y:'+str(y), train_wasserstein_y[y].result(), step = step)   
        
        #w0, w1 = test_step(batch_data_test, data_test, step)
        _, _ = test_step(batch_data_test, batch_data_test, step)
      
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=step)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=step)
            tf.summary.scalar('norm', var_norm.result(), step = step)
            tf.summary.scalar('gradient_norm', grad_norm.result(), step = step)
            
            for env in [0,1]:
                tf.summary.scalar('accuracy_env:'+str(env), test_accuracy_env[env].result(), step = step)   
                
            for y in [0,1]:
                tf.summary.scalar('wasserstein_y:'+str(y), test_wasserstein_y[y].result(), step = step)   
                
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        var_norm.reset_states()
        grad_norm.reset_states()
        
        for env in [0,1]:
            train_accuracy_env[env].reset_states()
            test_accuracy_env[env].reset_states()
        if step % wasserstein_epoch == 0:    
            for y in [0,1]:
                train_wasserstein_y[y].reset_states()
                test_wasserstein_y[y].reset_states()
            #print(f'Training WD for step {step} are {wd0_tr} {wd1_tr}\n')
            #print(f'Test WD for step {step} are {wd0_test} {wd1_test}\n')
        if step % 50 == 0:
            print(f'Done step {step}\n')
            
    return graph, current_time, expt_id

