import tensorflow as tf
import numpy as np

from models.generative import Generative
from models.recognition import Recognition

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

n_inputs = 28 * 28
n_outputs = 10
learning_rate = tf.constant(0.001)
k = tf.constant(0.1)
eps = tf.constant(0.0000000001)

s = {}
def RMSProp(theta, beta, gradient, scope):
    # updating momentum
    if scope not in s.keys():
        s[scope] = tf.multiply(gradient, gradient)
    s[scope] = beta * s[scope]
    rhs = tf.multiply(gradient, gradient)
    rhs = (1 - beta) * gradient
    s[scope] = s[scope] + rhs
    # updating theta
    rhs = tf.sqrt(s[scope]) + eps
    rhs = tf.divide(gradient, rhs)
    rhs = learning_rate * rhs
    return tf.assign(theta, theta - rhs)

def initialize():
    X = tf.placeholder(tf.float32, shape=(1, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(1), name="y")
    lhs_sum = tf.zeros(shape=(1, n_outputs), dtype=tf.float32)
    rhs = 0

    with tf.name_scope("network"):
        r = Recognition(X, 2, n_inputs)
        mu_stack, sigma_stack, logdet_stack = r.get_param()
        g = Generative(mu_stack, sigma_stack, 2, n_inputs, n_outputs)
        logits = g.get_param()
        print("logits: ", logits)
        prob = tf.nn.softmax(logits=logits)
        print("Prob: ", prob)
        l_prob = tf.log(prob)

    with tf.name_scope("loss"):
        with tf.name_scope("generative"):
            lhs_sum = lhs_sum + l_prob
            lhs = 0
            for i in range(1, 3):
                sub_theta = g.get_trainable(i)
                for l in range(len(sub_theta)):
                    norm = tf.square(tf.norm(sub_theta[l]))
                    lhs = lhs + norm
            for i in range(0, 2):
                matrix = g.get_matrix(i)[0]
                norm = tf.square(tf.norm(matrix))
                lhs = lhs + norm

        with tf.name_scope("recognition"):
            for i in range(len(mu_stack)):
                norm = tf.square(tf.norm(mu_stack[i]))
                rhs = rhs + norm
            for i in range(len(sigma_stack)):
                covariance = tf.matmul(sigma_stack[i], tf.transpose(sigma_stack[i]))
                norm = tf.linalg.trace(covariance)
                rhs = rhs + norm
                rhs = rhs - logdet_stack[i]
                rhs = rhs - 1
        loss = -lhs_sum
        loss = loss + 0.5 * k * lhs
        loss = loss + 0.5 * rhs

    with tf.name_scope("backpropagation"):
        ops = []
        with tf.name_scope("generative"):
            # optimizing parameters in fully connected layer
            for i in range(1, 3):
                sub_theta = g.get_trainable(i)
                # computing gradients
                sub_grad = tf.gradients(l_prob, sub_theta)
                for l in range(len(sub_grad)):
                    sub_grad[l] = -sub_grad[l]
                    sub_grad[l] = sub_grad[l] + k * sub_theta[l]
                    # assign
                    opt_scope = "G" + "Dense" + str(i) + str(l)
                    training_op = RMSProp(theta=sub_theta[l], beta=0.9, gradient=sub_grad[l], scope=opt_scope)
                    ops.append(training_op)
            # optimizing parameters in matrices
            for i in range(0, 2):
                matrix = g.get_matrix(i)[0]
                sub_grad = tf.gradients(l_prob, matrix)[0]
                sub_grad = -sub_grad
                sub_grad = sub_grad + (k * matrix)
                # assign
                opt_scope = "G" + "Matrix" + str(i)
                training_op = RMSProp(theta=matrix, beta=0.9, gradient=sub_grad, scope=opt_scope)
                ops.append(training_op)

        with tf.name_scope("recognition"):
            # gradients of mu and sigma
            residual = g.get_residual()
            samples = g.get_sample()
            sub_scope = ["v", "mu", "diag", "u"]
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
                rhs = tf.linalg.trace(covariance) - logdet_stack[i]
                rhs = tf.gradients(rhs, sigma_stack[i])[0]
                rhs = (rhs / 2)
                gradient_sigma = lhs + rhs

                for each_scope in sub_scope:
                    sub_theta = r.get_trainable(each_scope, i)
                    if each_scope == "mu":
                        gradient_wrt_mu = tf.gradients(mu_stack[i], sub_theta)
                        for l in range(len(gradient_wrt_mu)):
                            lhs = gradient_mu * tf.transpose(gradient_wrt_mu[l])
                            # assign
                            opt_scope = "R" + str(i) + str(l) + each_scope
                            training_op = RMSProp(theta=sub_theta[l], beta=0.9, gradient=lhs, scope=opt_scope)
                            ops.append(training_op)
                    elif each_scope == "diag" or each_scope == "u":
                        gradient_wrt_sigma = tf.gradients(sigma_stack[i], sub_theta)
                        gradient_wrt_sigma[1] = tf.reshape(gradient_wrt_sigma[1], shape=(n_inputs, 1))
                        for l in range(len(gradient_wrt_sigma)):
                            rhs = tf.matmul(gradient_sigma, gradient_wrt_sigma[l])
                            rhs = tf.linalg.trace(rhs)
                            # assign
                            opt_scope = "R" + str(i) + str(l) + each_scope
                            training_op = RMSProp(theta=sub_theta[l], beta=0.9, gradient=rhs, scope=opt_scope)
                            ops.append(training_op)
                    else:
                        gradient_wrt_mu = tf.gradients(mu_stack[i], sub_theta)
                        gradient_wrt_sigma = tf.gradients(sigma_stack[i], sub_theta)

                        if len(gradient_wrt_mu) == len(gradient_wrt_sigma):
                            gradient_wrt_sigma[1] = tf.reshape(gradient_wrt_sigma[1], shape=(n_inputs, 1))
                            for l in range(len(gradient_wrt_sigma)):
                                lhs = gradient_mu * tf.transpose(gradient_wrt_mu[l])
                                rhs = tf.matmul(gradient_sigma, gradient_wrt_sigma[l])
                                rhs = tf.linalg.trace(rhs)
                                # assign
                                opt_scope = "R" + str(i) + str(l) + each_scope
                                training_op = RMSProp(theta=sub_theta[l], beta=0.9, gradient=lhs + rhs, scope=opt_scope)
                                ops.append(training_op)
                        else:
                            print("Error : the number of elements in a gradient array doesn't match")
    return X, prob, loss, ops
