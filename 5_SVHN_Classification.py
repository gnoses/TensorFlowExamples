#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import input_data
from TrainingPlot import *
import scipy.io as sio
from batch_norm import batch_norm
from activations import lrelu
from connections import conv2d, linear
from batch_norm import *
from connections import *
weights = []

# labels_dense : m x 1
# output : m x num_classes
def DenseToOneHot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def CreateWeight(kernelSize, inputSize, outputSize):
    name = 'w%d' % len(weights)
    # return (tf.get_variable(name, shape=[kernelSize, kernelSize, inputSize, outputSize]), initializer=tf.contrib.layers.xavier_initializer()))
    return (tf.get_variable(name, shape=[kernelSize, kernelSize, inputSize, outputSize]))


def ConvBNRelu(input, kernelSize, outputSize,is_training):
    inputSize = input.get_shape()[3].value
    # print type(inputSize)
    weights.append(CreateWeight(kernelSize, inputSize, outputSize))
    conv = tf.nn.conv2d(input, weights[-1], strides=[1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.batch_normalization(conv, 0.001, 1.0, 0, 1, 0.0001)
    conv = batch_norm(conv,is_training)
    return tf.nn.relu(conv)

def FCRelu(input,outputSize):
    size = input.get_shape().as_list()
    inputSize = np.uint16(np.prod(size[1:]))
    shape = [inputSize, outputSize]
    # print shape
    weight = tf.Variable(tf.random_normal(shape, stddev=0.01))
    inputFlat = tf.reshape(input, [-1,inputSize ])  # reshape to (?, 2048)

    bias = tf.Variable(tf.random_normal([outputSize], stddev=0.01))
    fc = tf.matmul(inputFlat, weight) + bias
    return tf.nn.relu(fc)

def ModelSimple(X, is_training):
    h_1 = lrelu(batch_norm(conv2d(X, 32, name='conv1'),
                           is_training, scope='bn1'), name='lrelu1')
    h_2 = lrelu(batch_norm(conv2d(h_1, 64, name='conv2'),
                           is_training, scope='bn2'), name='lrelu2')
    h_3 = lrelu(batch_norm(conv2d(h_2, 64, name='conv3'),
                           is_training, scope='bn3'), name='lrelu3')
    h_3_flat = tf.reshape(h_3, [-1, 64 * 4 * 4])
    return linear(h_3_flat, 10)

def ModelVGGLike(X,is_training):
    conv1 = ConvBNRelu(X, 3, 256, is_training)
    conv1 = ConvBNRelu(conv1, 3, 256, is_training)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv2 = ConvBNRelu(conv1, 3, 128, is_training)
    conv2 = ConvBNRelu(conv1, 3, 128, is_training)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv3 = ConvBNRelu(conv2, 3, 128, is_training)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
    # print conv3
    fc = FCRelu(conv3, 1024)
    fc = tf.nn.dropout(fc, 0.5)
    return linear(fc, 10)

# gpu memory restricion
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

train = sio.loadmat('data/train_32x32.mat')
trX = train['X'].transpose([3,0,1,2]) / 255.0
trY = train['y'] - 1
# trX = trX[:5000,:,:,:]
# trY = trY[:5000,:]
# import matplotlib.pyplot as plt
# plt.ioff()
# plt.imshow(trX[0,:,:,:])
# plt.hist(trY)
# plt.show()
trY = DenseToOneHot(trY,10)

val = sio.loadmat('data/test_32x32.mat')
teX = val['X'].transpose([3, 0, 1, 2]) / 255.0
teY = val['y'] - 1
teY = DenseToOneHot(teY,10)

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

is_training = tf.placeholder(tf.bool, name='is_training')

# p_keep_conv = tf.placeholder("float")
# pred = ModelSimple(X,is_training)
pred = ModelVGGLike(X,is_training)

# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,Y))
# cost = -tf.reduce_sum(Y * tf.log(pred))

# optimizer
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(pred, 1)

# acc operation
correct_pred = tf.equal(tf.argmax(Y, 1), predict_op) # Count correct predictions
acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average


plot = TrainingPlot()
batchSize = 128
sampleCount = len(trX)
totalIter = 10000
plot.SetConfig(batchSize, sampleCount, totalIter)
resumeTraining = True
savePath = 'snapshot'
m = len(trX)
print 'Training data : %d ea' % m

# Launch the graph in a session
with tf.Session() as sess:
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(savePath)
    if resumeTraining == False:
        print "Start from scratch"
    elif checkpoint:
        print "Restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "Couldn't find checkpoint to restore from. Starting over."
    tf.initialize_all_variables().run()

    for i in range(totalIter):
        trainLoss = []
        trainAcc = []

        for start, end in zip(range(0, len(trX), batchSize), range(batchSize, len(trX), batchSize)):
            sess.run(train_op, feed_dict={X: trX[start:end],Y:trY[start:end],is_training:True})
            loss = sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],is_training:True})
            acc = sess.run(acc_op, feed_dict={X: trX[start:end], Y: trY[start:end],is_training:True})
            trainLoss.append(loss)
            trainAcc.append(acc)
            # print 'train %d : %f, %f' % (start, loss, acc)

        trainLoss = np.mean(trainLoss)
        trainAcc = np.mean(trainAcc)

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:batchSize]
        valLoss = sess.run(cost, feed_dict={X: teX[test_indices],Y:teY[test_indices],is_training:False})
        valAcc = sess.run(acc_op, feed_dict={X: teX[test_indices],Y:teY[test_indices],is_training:False})

        plot.Add(i, trainLoss, valLoss, trainAcc, valAcc)
        # plot.Add(i, trainLoss, valLoss, 0, 0)
        plot.Show()

        # save snapshot
        if (resumeTraining):
            saver.save(sess, savePath + '/progress', global_step=i)
