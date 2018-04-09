import tensorflow as tf
import numpy as np

import models.generative
import models.recognition

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

n_inputs = 28 * 28
n_outputs = 10
learning_rate = tf.constant(0.001)
k = tf.constant(0.1)

def initialize():
    X = tf.placeholder(tf.float32, shape=(1, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(1), name="y")

    with tf.name_scope("network"):
        r = Recognition(X, 2, n_inputs)
        mu_stack, sigma_stack = r.get_param()
        g = Generative(mu_stack, sigma_stack, 2, n_inputs, n_outputs)
        logits = g.get_param()
        prob = tf.nn.softmax(logits=logits)
        l_prob = tf.log(prob)

    with tf.name_scope("loss"):
        # gradients of generative network
        ops = []
        # optimizing parameters in fully connected layer
        for i in range(1, 3):
            for j in range(3):
                sub_theta = g.get_trainable(i, j)
                sub_grad = tf.gradients(l_prob, sub_theta)
                for l in range(len(sub_grad)):
                    sub_grad[l] = -sub_grad[l]
                    sub_grad[l] = sub_grad[l] + k * sub_theta[l]
                    # assign
                    training_op = tf.assign(sub_theta[l], sub_theta[l] - learning_rate * sub_grad[l])
                    ops.append(training_op)
        # optimizing parameters in matrices
        for i in range(0, 2):
            matrix = g.get_matrix(i)[0]
            sub_grad = tf.gradients(l_prob, matrix)[0]
            sub_grad = -sub_grad
            sub_grad = sub_grad + (k * matrix)
            # assign
            training_op = tf.assign(matrix, matrix - learning_rate * sub_grad)
            ops.append(training_op)

        # gradients of mu and sigma
        residual = g.get_residual()
        samples = g.get_sample()
        sub_scope = ["mu", "diag", "u"]
        for i in range(2):
            # for mu
            gradient_mu = tf.gradients(l_prob, samples[i])[0]
            gradient_mu = -gradient_mu
            gradient_mu = gradient_mu + mu_stack[i]
            # for sigma
            gradient_sample = tf.gradients(l_prob, samples[i])[0]
            lhs = tf.transpose(residual[i]) * gradient_sample
            lhs = - (lhs / 2)
            covariance = tf.matmul(sigma_stack[i], tf.transpose(sigma_stack[i]))
            rhs = tf.linalg.trace(covariance) - tf.linalg.logdet(covariance)
            rhs = tf.gradients(rhs, sigma_stack[i])[0]
            rhs = (rhs / 2)
            gradient_sigma = lhs + rhs

            for j in range(3):
                for each_scope in sub_scope:
                    sub_theta = r.get_trainable(each_scope, i, j)
                    if each_scope == "mu":
                        gradient_wrt_mu = tf.gradients(mu_stack[i], sub_theta)
                        for l in range(len(gradient_wrt_mu)):
                            lhs = gradient_mu * tf.transpose(gradient_wrt_mu[l])
                            # assign
                            training_op = tf.assign(sub_theta[l], sub_theta[l] - learning_rate * lhs)
                            ops.append(training_op)
                    else:
                        gradient_wrt_sigma = tf.gradients(sigma_stack[i], sub_theta)
                        gradient_wrt_sigma[1] = tf.reshape(gradient_wrt_sigma[1], shape=(n_inputs, 1))
                        for l in range(len(gradient_wrt_sigma)):
                            rhs = tf.matmul(gradient_sigma, gradient_wrt_sigma[l])
                            rhs = tf.linalg.trace(rhs)
                            print(rhs)
                            # assign
                            training_op = tf.assign(sub_theta[l], sub_theta[l] - learning_rate * rhs)
                            ops.append(training_op)

def main():


if __name__ == "__main__":
    main()
