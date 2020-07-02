import tensorflow as tf
import utils


class Invariance():

    def __init__(self, epoch = 1000, learning_rate = 1e-4, reg_covar = 1, reg_start = 100, clip_grad = 40, seed = 1,\
        lr_wasserstein = 1e-4, epsilon = 1):

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.reg_covar = reg_covar
        self.reg_start = reg_start
        self.clip_grad = clip_grad
        self.seed = seed
        self.lr_wasserstein = lr_wasserstein
        self.epsilon = epsilon
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.wasserstein_optimizer = tf.optimizers.Adam(self.lr_wasserstein)

        # Set seed
        tf.random.set_seed(self.seed)




    def create_graph(self, architecture = [20, 20, 2], activation = tf.nn.relu):
        self.model = utils.NNGraph(architecture = architecture, activation = activation, seed = self.seed)


    def potential_graph(self, architecture = [20, 1], activation = tf.nn.relu):
        self.potentials = []
        for label in [0, 1]:
            potential_list = []
            for env in [0, 1]:
                potential_list.append(utils.ConvexGraph(architecture, activation, seed= self.seed, name = f'potental-{label}-{env}'))
            self.potentials.append(potential_list)

        self.shuffled_potentials = []
        for label in [0, 1]:
            potential_list = []
            for env in [0, 1]:
                potential_list.append(utils.ConvexGraph(architecture, activation, seed= self.seed, name = f'potental-{label}-{env}'))
            self.shuffled_potentials.append(potential_list)


    
    def load_full_data(self, data_train, data_test):
        self.full_data_train = data_train
        self.full_data_test = data_test




    def data_steam(self, data_train, data_test, batch_size = 250):
        
        # Training stream
        self.batch_data = []
        for env in data_train:
            batch = tf.data.Dataset.from_tensor_slices((env[0], env[1]))
            batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
            self.batch_data.append(batch.take(self.epoch))
            
    
        batch = tf.data.Dataset.from_tensor_slices((data_test[0], data_test[1]))
        batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        self.batch_data.append(batch.take(self.epoch))

    def shuffle_probabilities(self, prob0, prob1):
        n0, n1 = prob0.shape[0]//2, prob1.shape[1]//2
        shuffled_prob0 = tf.concat([prob0[:n0], prob1[:n1]], axis = 0)
        shuffled_prob1 = tf.concat([prob0[n0:], prob1[n1:]], axis = 0)
        return shuffled_prob0, shuffled_prob1




    def create_tensorboard(self):

        name = f'epoch-{self.epoch}-lr-{self.learning_rate}-reg-{self.reg_covar}'
        self.train_summary_writer = tf.summary.create_file_writer('logs/' + name + '/train')
        self.test_summary_writer = tf.summary.create_file_writer('logs/' + name + '/test')



    def wasserstein_distance(self, x1, x2, potential1 = None, potential2 = None, \
         reguralizer = 'entropy', update = True):
        """
        Given dual potential functions calculates the regularized Wasserstein distance
        parameters:
        x1: tensor of dimension 2
        x2: tensor of dimension 2
        potential1: (tensor graph) dual potential for first data
        potential2: (tensor graph) dual potential for second data
        epsilon: (float) regularizing parameter
        regularizer: (string) method of regularization. Options: (1) entropy, and (2) L2
        return:
        distance: (tensor of shape ()) regularized wasserstein distance
        reference:
        [1] Seguy et. al.: 'Large-scale optimal transport and mapping estimation'
        """
        
        if potential1 == None:
            potential1 = self.potentials[0][0]
        if potential2 == None:
            potential2 = self.potentials[0][1]
        
        if not update:
            u = potential1(x1)
            v = potential2(x2)
            distance = tf.cast(0, dtype = tf.float32)
            distance += tf.reduce_mean(u) + tf.reduce_mean(v) 
            if tf.math.is_nan(distance):
                distance = tf.cast(0, dtype = tf.float32)
            return distance
        else:
            with tf.GradientTape(persistent = True) as tape:
                u = potential1(x1)
                v = potential2(x2)
                distance = tf.reduce_mean(u) + tf.reduce_mean(v) 
                x1_expanded = tf.tile(tf.expand_dims(x1, axis = 1), [1, x2.shape[0], 1]) # shape (nx1, nx2, 2)
                x2_expanded = tf.tile(tf.expand_dims(x2, axis = 0), [x1.shape[0], 1, 1]) # shape (nx1, nx2, 2)
                pairwise_distance = 0.5 * tf.reduce_sum((x1_expanded - x2_expanded)**2, axis = 2) # shape (nx1, nx2)
                u_expanded = tf.tile(tf.expand_dims(u, axis = 1), [1, v.shape[0], 1]) # shape (nu, nv, 1)
                v_expanded = tf.tile(tf.expand_dims(v, axis = 0), [u.shape[0], 1, 1]) # shape (nu, nv, 1)
                pairwise_distance = tf.reshape(pairwise_distance, (pairwise_distance.shape[0], pairwise_distance.shape[1], 1))
                L = u_expanded + v_expanded - pairwise_distance
                if reguralizer == 'entropy':
                    penalty = -self.epsilon * tf.exp((1/self.epsilon) * L)
                elif reguralizer == 'L2':
                    penalty = -(1/(4*self.epsilon))*(tf.nn.relu(L)**2)
                else:
                    raise TypeError('Wrong entry in regularizer. Options: entropy and L2')
                regularized_distance = distance + tf.reduce_mean(penalty)

            variables_u = potential1.trainable_variables
            gradient = tape.gradient(regularized_distance, variables_u)
            clipped_grad = [-tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
            self.wasserstein_optimizer.apply_gradients(zip(clipped_grad, variables_u))

            variables_v = potential2.trainable_variables
            gradient = tape.gradient(regularized_distance, variables_v)
            clipped_grad = [-tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
            self.wasserstein_optimizer.apply_gradients(zip(clipped_grad, variables_v))

            del tape

            if tf.math.is_nan(distance):
                distance = tf.cast(0, dtype = tf.float32)
                
            
            return distance

    def train_step(self, data_train, step = 0):


        with tf.GradientTape() as g:

            # Entropy loss and covariance loss
            entropy_loss = tf.cast(0, dtype = tf.float32)
            #covars = []
            probs = []
            for i, env in enumerate(data_train):
                x, y = env
                logits = self.model(x)
                prob = utils.probabilities(logits)[:, 1]
                probs.append(prob)
                entropy_loss += utils.entropy(logits, y)
                #covars.append(utils.covar(logits, y))
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(f'Train accuracy for environment: {i}', utils.accuracy(logits, y), step=step)
                
                # Save histogram
                if step == 1 or step % 250 == 0:
                    with self.train_summary_writer.as_default():
                        tf.summary.histogram(f'Train histogram environment: {i} label 0', prob[y[:, 1] == 0], step=step)
                        tf.summary.histogram(f'Train histogram environment: {i} label 1', prob[y[:, 1] == 1], step=step)
            
            # Covariance regularizer 
            (_, y0), (_, y1) = data_train
            #env = tf.concat([tf.repeat(0, y0.shape[0]), tf.repeat(1, y1.shape[0])], axis = 0)
            #y = tf.concat([y0, y1], axis = 0)
            #prob = tf.concat(probs, axis = 0)
            w_regularizer = tf.cast(0, dtype = tf.float32)
            for label in [0, 1]:
                prob0 = tf.reshape(probs[0][y0[:, 1] == label], (-1, 1))
                prob1 = tf.reshape(probs[1][y1[:, 1] == label], (-1, 1))
                shuffled_prob0, shuffled_prob1 = self.shuffle_probabilities(prob0, prob1)
                
                w_regularizer += self.wasserstein_distance(prob0, prob1, self.potentials[label][0], self.potentials[label][1])/\
                    self.wasserstein_distance(shuffled_prob0, shuffled_prob1, self.shuffled_potentials[label][0], self.shuffled_potentials[label][1])


            loss = entropy_loss
            if step >= self.reg_start:
                loss += self.reg_covar * w_regularizer

            with self.train_summary_writer.as_default():
                tf.summary.scalar('Entropy', entropy_loss/2, step=step)
                tf.summary.scalar('Wasserstein regularizer', w_regularizer, step=step)

        trainable_variables = self.model.trainable_variables
        gradient = g.gradient(loss, trainable_variables)
        clipped_grad = [tf.clip_by_norm(g, self.clip_grad) for g in gradient]
        self.optimizer.apply_gradients(zip(clipped_grad, trainable_variables))

    def test_step(self, test_data, step = 1):
        with self.test_summary_writer.as_default():
            x, y = test_data
            logits = self.model(x)
            prob = utils.probabilities(logits)[:, 1]
            tf.summary.scalar('Entropy', utils.entropy(logits, y), step=step)
            tf.summary.scalar('Test accuracy', utils.accuracy(logits, y), step=step)
            if step == 1 or step % 250 == 0:
                tf.summary.histogram(f'Test histogram label 0', prob[y[:, 1] == 0], step=step)
                tf.summary.histogram(f'Test histogram label 1', prob[y[:, 1] == 1], step=step)


    def fit(self):
        for step, data in enumerate(zip(*self.batch_data), 1):
            data_train = data[:-1]
            data_test = data[-1]
            self.train_step(data_train, step=step)
            self.test_step(data_test, step=step)
            if step % 100 == 0:
                x, y = self.full_data_test
                logits = self.model(x)
                accuracy = utils.accuracy(logits, y)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('Test accuracy on full data', accuracy, step=step)
                print(f'Test accuracy at step {step}: {accuracy.numpy()}')

        





