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
        self.logdet_stack = None
        self.act = tf.nn.elu

        self._build_network()
    # create a new network
    def _build_network(self):
        with tf.variable_scope(self.scope):
            mu_list = []
            sigma_list = []
            logdet_list = []
            v = self.V
            for i in range(self.num_layers):
                with tf.variable_scope("v"):
                    current_scope = "layer" + str(self.num_layers - i - 1)
                    v = tf.layers.dense(inputs=v, units=self.num_units, activation=self.act,  name=current_scope, reuse=tf.AUTO_REUSE)
                with tf.variable_scope("mu"):
                    current_scope = "layer" + str(self.num_layers - i - 1)
                    v1 = tf.layers.dense(inputs=v, units=self.num_units, activation=self.act, name=current_scope, reuse=tf.AUTO_REUSE)
                    mu = v1
                with tf.variable_scope("diag"):
                    current_scope = "layer" + str(self.num_layers - i - 1)
                    v2 = tf.layers.dense(inputs=v, units=self.num_units, activation=self.act,  name=current_scope, reuse=tf.AUTO_REUSE)
                    d = v2
                with tf.variable_scope("u"):
                    current_scope = "layer" + str(self.num_layers - i - 1)
                    v3 = tf.layers.dense(inputs=v, units=self.num_units, activation=self.act, name=current_scope, reuse=tf.AUTO_REUSE)
                    u = v3

                sigma, logdet = self._get_sigma(d[0], u[0])
                mu_list.append(mu[0])
                sigma_list.append(sigma)
                logdet_list.append(logdet)

            mu_list.reverse()
            sigma_list.reverse()
            logdet_list.reverse()
            self.mu_stack = mu_list
            self.sigma_stack = sigma_list
            self.logdet_stack = logdet_list

    def _pinv(self, A, reltol=1e-6):
        # Compute the SVD of the input matrix A
        s, u, v = tf.svd(A)
        # Invert s, clear entries lower than reltol*s[0].
        atol = tf.reduce_max(s) * reltol
        s = tf.boolean_mask(s, s > atol)
        s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.shape(A)[0] - tf.size(s)])], 0))
        # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
        return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))

    def _approx_matrix(self, A, reltol=1e-6):
        s, u, v = tf.svd(A)

        atol = tf.reduce_max(s) * reltol
        s = tf.boolean_mask(s, s > atol)
        s = tf.diag(tf.concat([s, tf.zeros([tf.shape(A)[0] - tf.size(s)])], 0))

        return tf.matmul(u, tf.matmul(s, tf.transpose(v)))

    # getting sigma
    def _get_sigma(self, d, u):
        u = tf.reshape(u, shape=(1, self.num_units))
        # preprocess
        nu_inverse = tf.matmul(tf.diag(d), tf.transpose(u))
        nu_inverse = tf.matmul(u, nu_inverse)
        nu = 1 / (nu_inverse + 1)
        # LHS
        lhs = tf.diag(d)
        lhs = self._pinv(lhs)
        lhs = tf.sqrt(lhs)
        # RHS
        rhs = tf.matmul(u, lhs)
        rhs = tf.matmul(tf.transpose(u), rhs)
        rhs = tf.matmul(self._pinv(tf.diag(d)), rhs)
        rhs = ((1 - tf.sqrt(nu)) / nu_inverse) * rhs
        # logdet of C
        logdet = tf.log(nu) - tf.linalg.logdet(self._approx_matrix(tf.diag(d)))

        return lhs - rhs, logdet
    # getting what's computed above
    def get_param(self):
        return self.mu_stack, self.sigma_stack, self.logdet_stack
    # getting trainable variable
    def get_all(self):
        return tf.trainable_variables(scope=self.scope)
    def get_trainable(self, sub_scope, i):
        return tf.trainable_variables(scope=self.scope + "/" + sub_scope + "/layer" + str(i))
