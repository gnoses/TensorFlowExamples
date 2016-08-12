# SVHN Training
# Sejin Park
#
# We need Caffe to load lmdb prior to Tensorflow
import os
from LoadData import *
classes = 2
# X : m x w x h x 3
# Y : m x classes
db = 0
# cool lmdb
if db == 0:
    trXFull, trYFull, teX, teY = LoadLMDBData('data/CroppedFullLMDB/', classes)
    savePath = 'snapshotFull'
else:
    trXFull, trYFull, teX, teY = LoadLMDBData('data/CroppedSmall%dLMDB/' % db, classes)
    savePath = 'snapshot%d' % db
try:
    os.makedirs(savePath)
except:
    pass

# trXFull, trYFull, teX, teY = LoadDB('data/CroppedSmall1000/')
# split negative samples into n times of positives
# bootstrap + batch learning
trXList, trYList = PickNegativeSample(trXFull, trYFull)
# trXList = []
# trXList.append(trXFull)
# trYList = []
# trYList.append(trYFull)
# import matplotlib.pyplot as plt
# trX1 = trXList[1]
# trY1 = trYList[1]
# print trX1.shape
# index = trY1[:,0]==1
#
# imgPos = trX1[index,:,:,:]
# # imgNeg = trX1[trY1>0]
# # plt.hist(trYFull)
# print imgPos.shape
# plt.ioff()
# for i in range(imgPos.shape[0]):
#     plt.clf()
#     plt.imshow(imgPos[i])
#     plt.pause(0.01)
# exit(0)

#!/usr/bin/env python
import tensorflow as tf
import numpy as np
# import input_data
from TrainingPlot import *
# import scipy.io as sio
from batch_norm import batch_norm
from activations import lrelu
from connections import conv2d, linear
from batch_norm import *
from connections import *
import PIL.Image as Image
import cPickle as pkl

# import matplotlib.pyplot as plt
weights = []

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

def ModelVGGLikeSmall(X,is_training):
    conv1 = ConvBNRelu(X, 3, 256, is_training)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv2 = ConvBNRelu(conv1, 3, 128, is_training)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    fc = FCRelu(conv2, 512)
    fc = tf.nn.dropout(fc, 0.5)
    return linear(fc, classes)

def ModelVGGLike(X,is_training):
    conv1 = ConvBNRelu(X, 3, 256, is_training)
    conv1 = ConvBNRelu(conv1, 3, 256, is_training)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv2 = ConvBNRelu(conv1, 3, 128, is_training)
    conv2 = ConvBNRelu(conv2, 3, 128, is_training)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv3 = ConvBNRelu(conv2, 3, 128, is_training)
    conv3 = ConvBNRelu(conv3, 3, 128, is_training)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv4 = ConvBNRelu(conv3, 3, 128, is_training)
    conv4 = ConvBNRelu(conv4, 3, 128, is_training)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
    # print conv3
    fc = FCRelu(conv4, 1024)
    fc = tf.nn.dropout(fc, 0.5)
    return linear(fc, classes)

def Prediction(testX, testY, batchSize):
    valLoss = []
    valAcc = []
    i = 0
    for start, end in zip(range(0, len(testX), batchSize), range(batchSize, len(testX) + batchSize, batchSize)):
        if (end >= len(testX)):
            end = len(testX)
        loss, acc = sess.run((cost, acc_op), feed_dict={X: testX[start:end], Y: testY[start:end], is_training: False})
        # acc = sess.run(acc_op, feed_dict={X: testX[start:end], Y: testY[start:end], is_training: False})
        # print 'Predict batch# %d : %f' % (i, acc)
        i += 1
        # correct = sess.run(correct_pred, feed_dict={X: testX[start:end], Y: testY[start:end], is_training: False})
        # print correct.shape
        # correctPos = correct[testY[start:end][0] != 0]
        # valAccPos.append(np.mean(correctPos))
        valLoss.append(loss)
        valAcc.append(acc)

    return np.mean(valLoss), np.mean(valAcc)

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, classes])

is_training = tf.placeholder(tf.bool, name='is_training')

# p_keep_conv = tf.placeholder("float")
# pred = ModelSimple(X,is_training)
# pred = ModelVGGLikeSmall(X,is_training)
output = ModelVGGLike(X,is_training)

# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output,Y))
# cost = -tf.reduce_sum(Y * tf.log(pred))

# optimizer
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(output, 1)

# acc operation
correct_pred = tf.equal(tf.argmax(Y, 1), predict_op) # Count correct predictions
acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average

plot = TrainingPlot()
batchSize = 1000
sampleCount = len(trXFull)
totalIter = 1000000
plot.SetConfig(batchSize, sampleCount, totalIter)
resumeTraining = True

print 'Training data : %d ea' % len(trXList[0])

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(savePath)
    if resumeTraining == False:
        print "Start from scratch"
        # tf.initialize_all_variables().run()
    elif checkpoint:
        print "Restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "Couldn't find checkpoint to restore from. Starting over."
        # tf.initialize_all_variables().run()
    negIndex = 0
    valLoss = 0
    valAcc = 0

    for i in range(totalIter):
        trainLoss = []
        trainAcc = []

        # train with sampled negative
        trX = trXList[negIndex]
        trY = trYList[negIndex]
        negIndex += 1
        if negIndex == len(trXList):
            negIndex = 0
        k = 0
        # print 'batch data count : %d' % len(trX)
        # print range(0, len(trX), batchSize)
        for start, end in zip(range(0, len(trX), batchSize), range(batchSize, len(trX) + batchSize, batchSize)):
            if (end >= len(trX)):
                end = len(trX)
            sess.run(train_op,feed_dict={X: trX[start:end], Y: trY[start:end], is_training: True})
            loss, acc = sess.run((cost, acc_op), feed_dict={X: trX[start:end], Y: trY[start:end],is_training:False})
            trainLoss.append(loss)
            trainAcc.append(acc)
            # print 'train iter %d neg# %d batch# %d : %f' % (i, negIndex, k, acc)
            k += 1

        trainLoss = np.mean(trainLoss)
        trainAcc = np.mean(trainAcc)

        # Prediction(teX, teY, batchsize)

        # print 'test time ', (time.time() - start)
        if (i % 50 == 0):
            valLoss, valAcc = Prediction(teX, teY, batchSize)

        # save snapshot
        if (resumeTraining  and (i + 1) % 50 == 0):
            # valLoss = []
            # valAcc = []
            # valAccPos = []
            # for k in range(0,len(trXList)):
            #     loss, acc = Prediction(trXList[k], trYList[k], batchSize)
            #     # print 'val iter %d neg# %d : %f' % (i,k,acc)
            #     valLoss.append(loss)
            #     valAcc.append(acc)
            # plot.Add(i, trainLoss, np.mean(valLoss), trainAcc, np.mean(valAcc))
            saver.save(sess, savePath + '/progress_%d' % (i + 1))

        plot.Add(i, trainLoss, valLoss, trainAcc, valAcc)
        plot.Show()
