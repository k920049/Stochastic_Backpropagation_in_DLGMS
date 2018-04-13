from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data

import control

n_epochs = 10
batch_size = 16

def main():
    mnist = input_data.read_data_sets("../resource")
    X, prob, loss, ops = control.initialize()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                for j in range(batch_size):
                    print("Epoch: " + str(epoch) + ", batch: " + str(iteration) + ", iteration: " + str(j))
                    input = np.reshape(X_batch[j], (1, 784))
                    sess.run(ops, feed_dict={X : input})
                input = np.reshape(X_batch[0], (1, 784))
                energy = loss.eval(feed_dict={X : input})
                probability = prob.eval(feed_dict={X : input})
                print(probability.shape)

        save_path = saver.save(sess, "./my_model_final.ckpt")

if __name__ == "__main__":
    main()
