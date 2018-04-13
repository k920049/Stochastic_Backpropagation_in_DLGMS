import tensorflow as tf
import numpy as np

class Generative:
    # initializer
    def __init__(self, mu_stack, sigma_stack, num_layers, num_units, num_outputs, scope = "generative"):
        self.scope = scope
        self.mu_stack = mu_stack
        self.sigma_stack = sigma_stack
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_outputs = num_outputs
        self.param = None
        self.residual = None
        self.samples = None
        self.act = tf.nn.elu

        self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.scope):
            residual_list = []
            sample_list = []
            G = tf.get_variable(name="G0",
                                shape=[self.num_units, self.num_units])
            epsilon = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=self.mu_stack[0],
                                                                                covariance_matrix=tf.matmul(self.sigma_stack[0], tf.transpose(self.sigma_stack[0]))).sample()
            sample_list.append(epsilon)
            residual_list.append(self._get_residual(epsilon, self.mu_stack[0], self.sigma_stack[0]))
            h = G * epsilon
            for i in range(1, self.num_layers):
                current_scope = "layer" + str(i)
                h = tf.layers.dense(inputs=h, units=self.num_units, activation=self.act, name=current_scope, reuse=tf.AUTO_REUSE)
                epsilon = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=self.mu_stack[i], covariance_matrix=tf.matmul(self.sigma_stack[i], tf.transpose(self.sigma_stack[i]))).sample()
                sample_list.append(epsilon)
                residual_list.append(self._get_residual(epsilon, self.mu_stack[i], self.sigma_stack[i]))
                G = tf.get_variable(name="G" + str(i), shape=[self.num_units, self.num_units])
                h = h + G * epsilon
            self.samples = sample_list
            self.residual = residual_list

            current_scope = "layer" + str(self.num_layers)
            h = tf.layers.dense(inputs=h, units=self.num_outputs, activation=self.act, name=current_scope)
            self.param = h
    # getting samples
    def get_sample(self):
        return self.samples

    def _pinv(self, A, b, reltol=1e-6):
        # Compute the SVD of the input matrix A
        s, u, v = tf.svd(A)
        # Invert s, clear entries lower than reltol*s[0].
        atol = tf.reduce_max(s) * reltol
        s = tf.boolean_mask(s, s > atol)
        s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))
        # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
        return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))

    # computing residual
    def _get_residual(self, sample, mu, sigma):
        sample = sample - mu
        sample = tf.reshape(sample, shape=(self.num_units, 1))
        sample = self._pinv(sigma, sample)
        return sample
    # getting residual
    def get_residual(self):
        return self.residual
    # getting logits
    def get_param(self):
        return self.param
    # getting trainable variable
    def get_all(self):
        return tf.trainable_variables(scope=self.scope)
    def get_trainable(self, i):
        return tf.trainable_variables(scope=self.scope + "/layer" + str(i))
    def get_matrix(self, i):
        return tf.trainable_variables(scope=self.scope + "/G" + str(i))
