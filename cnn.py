#!/usr/bin/env python

# CNN for MNIST Classification
#
# Standard MNIST: 99.22%
# Kaggle Digit Recognizer: 99.04%

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


mnist = pd.read_csv('mnist.csv', sep=',')
mnist_test = pd.read_csv('test.csv', sep=',')

itrain = mnist.drop(['label'], axis=1).values.astype(np.float32)
ltrain = mnist['label'].values.astype(np.uint8)

# itest = mnist_test.values

itest = itrain[60000:]
ltest = ltrain[60000:]

itrain = itrain[:60000]
ltrain = ltrain[:60000]

itrain = itrain/255.0
itest = itest/255.0


BATCH_SIZE = 100


is_train = tf.placeholder(tf.bool)

def he(shape):
    fan_in = shape[0] if len(shape)==2 else np.prod(shape[1:])
    return np.sqrt(6.0 / fan_in)

# Normalize distributions between layers
def normalize(inp, isConv):
    beta = tf.Variable(tf.fill(inp.get_shape()[-1:], 0.0))
    gamma = tf.Variable(tf.fill(inp.get_shape()[-1:], 1.0))
    eps = 0.0001

    if isConv:
        mean, var = tf.nn.moments(inp, axes=[0,1,2])
    else:
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


def prelu(inp, biases, isConv):
    alpha = tf.Variable(tf.fill(biases, 0.001))
    y = normalize(inp, isConv)
    return tf.maximum(0.0, y) + alpha*tf.minimum(0.0, y)

def conv(inp, strides, weights):
    W = tf.Variable(tf.random_uniform(weights, -he(weights), he(weights)))
    return tf.nn.conv2d(inp, W, strides, padding='SAME')

def pool(inp, ksize, strides):
    return tf.nn.max_pool(inp, ksize=ksize, strides=strides, padding='SAME')

def fc(inp, weights):
    W = tf.Variable(tf.random_uniform(weights, -he(weights), he(weights)))
    return tf.matmul(inp, W)


# Build CNN:
#
# Input (28,28,1,) in batches of 100 (BN)
# Conv [5,5,1]x20
# PRelu (BN)
# MaxPool [2,2]
# Conv [5,5,20]x40
# PRelu (BN)
# MaxPool [2,2]
# FC-PRelu 1600 (BN)
# Dropout 0.2
# FC-PRelu 400 (BN)
# Dropout 0.2
# Softmax 10

dropout_rate = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
x2 = normalize(tf.reshape(x, [-1,28,28,1]), True)

conv1 = conv(x2, [1,1,1,1], [5,5,1,20])
relu1 = prelu(conv1, [20], True)
pool1 = pool(relu1, [1,2,2,1], [1,2,2,1])

conv2 = conv(pool1, [1,1,1,1], [5,5,20,40])
relu2 = prelu(conv2, [40], True)
pool2 = pool(relu2, [1,2,2,1], [1,2,2,1])

flat = tf.reshape(pool2, [-1, 7*7*40])
relu3 = prelu(fc(flat, [7*7*40,1600]), [1600], False)
dropout = tf.nn.dropout(relu3, dropout_rate)

relu4 = prelu(fc(dropout, [1600,400]), [400], False)
dropout2 = tf.nn.dropout(relu4, dropout_rate)

b = tf.Variable(tf.random_uniform([10], -he([10,1]), he([10,1])))
o = fc(dropout2, [400, 10]) + b
y = tf.argmax(o, axis=1)

y_ = tf.placeholder(tf.int64, [None,])
y_oh = tf.one_hot(y_, 10)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(o, y_oh))
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(y, y_), tf.float32))

train = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('log/', sess.graph)
    sess.run(init)

    print('Training...')
    next_btch = next_batch(itrain, ltrain, BATCH_SIZE)
    for i in range(0, 4000):
        batch_xs, batch_ys = next(next_btch)

        feed_dict = {
            x: batch_xs,
            y_: batch_ys,
            dropout_rate: 0.2,
            is_train: True
        }

        sess.run(train, feed_dict)
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

    next_btch = next_batch(itest, ltest, BATCH_SIZE)
    acc = np.zeros([100])
    for i in range(0, 100):
        batch_xs, batch_ys = next(next_btch)
        feed_dict = {
            x: batch_xs,
            y_: batch_ys,
            dropout_rate: 1.0,
            is_train: False
        }
        acc = acc + sess.run(accuracy, feed_dict)

    print("Test: {:.2f} accuracy.".format(np.average(acc)/100))

    # classified = np.zeros([28000])
    #
    # for i in range(0, 280):
    #     batch_xs = itest[i*100:i*100+100]
    #     feed_dict = {
    #         x: batch_xs,
    #         is_train: False,
    #         dropout_rate: 1.0
    #     }
    #     classified[i*100:i*100+100] = sess.run(y, feed_dict)
    #
    # df = pd.DataFrame(classified.astype(np.int), columns=['Label'], index=range(1,28001))
    # df.index.name = 'ImageId'
    # df.to_csv('classified.csv')
