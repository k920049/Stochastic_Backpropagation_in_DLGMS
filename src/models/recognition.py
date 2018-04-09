import tensorflow as tf
import numpy as np

class Recognition:
    # initializer
    def __init__(self, V, num_layers, num_units, scope = "recognition"):
        self.scope = scope
        self.num_layers = num_layers
        self.num_units = num_units
        self.V = V
        self.mu_stack = None
        self.sigma_stack = None

        self._build_network()
    # create a new network
    def _build_network(self):
        with tf.variable_scope(self.scope):
            mu_list = []
            sigma_list = []
            v1 = self.V
            v2 = self.V
            v3 = self.V
            for i in range(self.num_layers):
                with tf.variable_scope("mu"):
                    for j in range(3):
                        current_scope = "layer" + str(self.num_layers - i - 1) + "/depth" + str(j)
                        v1 = tf.layers.dense(inputs=v1,
                                             units=self.num_units,
                                             name=current_scope, reuse=tf.AUTO_REUSE)
                    mu = v1
                with tf.variable_scope("diag"):
                    for j in range(3):
                        current_scope = "layer" + str(self.num_layers - i - 1) + "/depth" + str(j)
                        v2 = tf.layers.dense(inputs=v2,
                                             units=self.num_units,
                                             name=current_scope, reuse=tf.AUTO_REUSE)
                    d = v2
                with tf.variable_scope("u"):
                    for j in range(3):
                        current_scope = "layer" + str(self.num_layers - i - 1) + "/depth" + str(j)
                        v3 = tf.layers.dense(inputs=v3,
                                             units=self.num_units,
                                             name=current_scope, reuse=tf.AUTO_REUSE)
                    u = v3

                sigma = self._get_sigma(d[0], u[0])
                mu_list.append(mu[0])
                sigma_list.append(sigma)

            mu_list.reverse()
            sigma_list.reverse()
            self.mu_stack = mu_list
            self.sigma_stack = sigma_list
    # getting sigma
    def _get_sigma(self, d, u):
        u = tf.reshape(u, shape=(1, self.num_units))
        # preprocess
        nu_inverse = tf.matmul(tf.diag(d), tf.transpose(u))
        nu_inverse = tf.matmul(u, nu_inverse)
        nu = 1 / (nu_inverse + 1)
        # lhs
        lhs = tf.diag(d)
        lhs = tf.matrix_inverse(lhs)
        lhs = tf.sqrt(lhs)
        #rhs
        rhs = tf.matmul(u, lhs)
        rhs = tf.matmul(tf.transpose(u), rhs)
        rhs = tf.matmul(tf.matrix_inverse(tf.diag(d)), rhs)
        rhs = ((1 - tf.sqrt(nu)) / nu_inverse) * rhs

        return lhs - rhs
    # getting what's computed above
    def get_param(self):
        return self.mu_stack, self.sigma_stack
    # getting trainable variable
    def get_all(self):
        return tf.trainable_variables(scope=self.scope)
    def get_trainable(self, sub_scope, i, j):
        return tf.trainable_variables(scope=self.scope + "/" + sub_scope + "/layer" + str(i) + "/depth" + str(j))
