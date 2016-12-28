#!/usr/bin/env python

# Neural Network for MNIST classification
#
# Standard MNIST: 98.34%
# Kaggle Digit Recognizer: 98.02%

import tensorflow as tf
import numpy as np
import pandas as pd


def next_batch(data, labels, batch_size):
    i = 0
    n = len(data)
    while True:
        yield (data[i:i+batch_size], labels[i:i+batch_size])
        i += batch_size
        if i >= n:
            i = 0


BATCH_SIZE = 100

mnist = pd.read_csv('train.csv', sep=',')
mnist_test = pd.read_csv('test.csv', sep=',')

itrain = mnist.drop(['label'], axis=1).values.astype(np.float32)
ltrain = mnist['label'].values.astype(np.uint8)

itest = mnist_test.values

# itest = itrain[60000:]
# ltest = ltrain[60000:]

# itrain = itrain[:60000]
# ltrain = ltrain[:60000]

itrain = itrain/255.0
itest = itest/255.0


# Controls batch normalization
is_train = tf.placeholder(tf.bool)

# Ratio of zeroed neurons
dropout_rate = tf.placeholder(tf.float32)

# Normalize distribution of inputs between layers
# When training mean and variance are computed on current
# batch, for testing average of all of them is used instead
def normalize(inp):
    beta = tf.Variable(tf.fill(inp.get_shape()[1:], 0.0))
    gamma = tf.Variable(tf.fill(inp.get_shape()[1:], 1.0))
    eps = 0.0001

    mean, var = tf.nn.moments(inp, axes=[0])

    amean = tf.Variable(tf.fill(inp.get_shape()[1:], 0.0), trainable=False)
    avar = tf.Variable(tf.fill(inp.get_shape()[1:], 1.0), trainable=False)

    train_amean = tf.assign(amean, (amean+mean)/2)
    train_avar = tf.assign(avar, (avar+var)/2)

    with tf.control_dependencies([train_amean, train_avar]):
        return tf.cond(
            is_train,
            lambda: tf.nn.batch_normalization(inp, mean, var, beta, gamma, eps),
            lambda: tf.nn.batch_normalization(inp, amean, avar, beta, gamma, eps)
        )


# Maxout layer
def maxout(inp, shape):
    W = tf.Variable(tf.random_uniform(shape, -0.005, 0.005))
    o = tf.einsum('ij,jka->ika', inp, W)
    on = normalize(o)
    return (tf.reduce_max(on, reduction_indices=[2]), W)

# Relu layer
def relu(inp, shape):
    W = tf.Variable(tf.random_uniform(shape, -0.005, 0.005))
    o = tf.matmul(inp, W)
    on = normalize(o)
    return (tf.nn.relu(on), W)

# PRelu layer
def prelu(inp, shape):
    W = tf.Variable(tf.random_uniform(shape, -0.005, 0.005))
    alpha = tf.Variable(tf.fill(shape[1:], 0.001))
    o = tf.matmul(inp, W)
    on = normalize(o)
    return (tf.maximum(0.0, on) + alpha*tf.minimum(0.0, on), W)


# Build NN:
#
# Input (784,) in batches of 100 (BN)
# PRelu 800 (BN)
# Dropout 0.5
# PRelu 400 (BN)
# Dropout 0.5
# Softmax 10

x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
xn = normalize(x)

l1, W1 = prelu(xn, [784, 800])
dropout1 = tf.nn.dropout(l1, dropout_rate)

l2, W2 = prelu(dropout1, [800, 400])
dropout2 = tf.nn.dropout(l2, dropout_rate)

classW = tf.Variable(tf.random_uniform([400, 10], -0.005, 0.005))
classB = tf.Variable(tf.random_uniform([10], -0.01, 0.01))

o = tf.matmul(dropout2, classW) + classB
y = tf.argmax(o, axis=1)
y_ = tf.placeholder(tf.int64, [None,])
y_oh = tf.one_hot(y_, 10)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(o, y_oh))

accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(y, y_), tf.float32))

c = tf.summary.scalar('cost', cost)
acc = tf.summary.scalar('accuracy', accuracy)

train = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('log/', sess.graph)
    sess.run(init)

    #Training
    print('Training...')
    next_btch = next_batch(itrain, ltrain, BATCH_SIZE)
    for i in range(0, 12000):
        batch_xs, batch_ys = next(next_btch)

        feed_dict = {
            x: batch_xs,
            y_: batch_ys,
            is_train: True,
            dropout_rate: 0.5
        }

        _, summary = sess.run([train, merged], feed_dict)
        writer.add_summary(summary, i)

        # Random shuffle training data
        if i % 500 == 0:
            perm = np.random.permutation(itrain.shape[0])
            itrain = itrain[perm]
            ltrain = ltrain[perm]

        # Print stats
        if i % 100 == 0:
            a_, c_ = sess.run([accuracy, cost], feed_dict)
            print(
"""Iterations: {}
Accuracy: {}
Cost: {}
""".format(i, a_, c_)
            )

    # next_btch = next_batch(itest, ltest, BATCH_SIZE)
    # acc = np.zeros([100])
    #
    # for i in range(0, 100):
    #     batch_xs, batch_ys = next(next_btch)
    #     feed_dict = {
    #         x: batch_xs,
    #         y_: batch_ys,
    #         is_train: False,
    #         dropout_rate: 1.0
    #     }
    #     acc = acc + sess.run(accuracy, feed_dict)
    #
    # print("Test: {:.2f} accuracy.".format(np.average(acc)/100))


    classified = np.zeros([28000])

    for i in range(0, 280):
        batch_xs = itest[i*100:i*100+100]
        feed_dict = {
            x: batch_xs,
            is_train: False,
            dropout_rate: 1.0
        }
        classified[i*100:i*100+100] = sess.run(y, feed_dict)

    df = pd.DataFrame(classified.astype(np.int), columns=['Label'], index=range(1,28001))
    df.index.name = 'ImageId'
    df.to_csv('classified.csv')
