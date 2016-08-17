# SVHN negative sampling refinement
# Sejin Park
#
# We need Caffe to load lmdb prior to Tensorflow
import os
from LoadData import *
import shutil
from ROCCurve import *
from LMDBTool import LMDBTool

classes = 2
# X : m x w x h x 3
# Y : m x classes

# cool lmdb
trXFull, trYFull, teX, teY = LoadLMDBData('data/CroppedSmall1000LMDB/', classes)

savePath = 'snapshot100'
try:
    os.makedirs(savePath)
except:
    pass


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

def ModelVGGLikeSmall(X,is_training):
    conv1 = ConvBNRelu(X, 3, 32, is_training)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv2 = ConvBNRelu(conv1, 3, 32, is_training)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    fc = FCRelu(conv2, 256)
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
    # valAccPos = []

    for start, end in zip(range(0, len(testX), batchSize), range(batchSize, len(testX), batchSize)):
        loss = sess.run(cost, feed_dict={X: testX[start:end], Y: testY[start:end], is_training: False})
        acc = sess.run(acc_op, feed_dict={X: testX[start:end], Y: testY[start:end], is_training: False})
        # correct = sess.run(correct_pred, feed_dict={X: testX[start:end], Y: testY[start:end], is_training: False})
        # print correct.shape
        # correctPos = correct[testY[start:end][0] != 0]
        # valAccPos.append(np.mean(correctPos))
        # print '%.5f' % acc
        valLoss.append(loss)
        valAcc.append(acc)

    return np.mean(valLoss), np.mean(valAcc) # , np.mean(valAccPos)

class ResultManager:
    count = 0
    imgList = np.array([]).reshape(0,32,32,3)
    scoreList = np.array([]).reshape(0,1)
    labelList = np.array([]).reshape(0,1)
    npImages = None
    def Add(self, batchData, batchProb, batchLabel):
        # no elements
        if (batchLabel.shape[0] > 0):
            self.imgList = np.vstack([self.imgList, batchData])
            self.scoreList = np.vstack([self.scoreList, batchProb.reshape(-1,1)])
            self.labelList = np.vstack([self.labelList, batchLabel.reshape(-1, 1)])

    def SaveImage(self, path):
        try:
            shutil.rmtree(path)
        except:
            pass
        try:
            os.makedirs(path)
        except:
            print 'Can\'t make ', path
            return
        # images, scores = self.ConvertListToNdarray()
        images = self.imgList
        scores = self.scoreList
        print images.shape
        self.npImages = images
        self.count = images.shape[0]
        for i in range(images.shape[0]):
            npImage = self.imgList[i, :, :, :].astype(np.uint8)
            img = Image.fromarray(npImage)
            confidence = np.int8(scores[i] * 100)
            if (self.scoreList[i,0] >= 0.5):
                img.save(path + '/%05d_%d_PosPred.png' % (confidence, i))
            else:
                img.save(path + '/%05d_%d_NegData.png' % (confidence, i))

    def SaveLMDB(self, path):
        images = self.imgList
        scores = self.scoreList
        label = self.labelList
        print images.shape
        self.npImages = images
        self.count = images.shape[0]
        lmdbTool = LMDBTool(path, 1000, True)
        for i in range(images.shape[0]):
            npImage = self.imgList[i, :, :, :].astype(np.uint8)
            # img = Image.fromarray(npImage)
            lmdbTool.Put(npImage, label[i])
        lmdbTool.Flush()

# collect hard negative training samples
# 1. collect false positive as negatives
# 2. collect true negative with low confidence (<0.2) as negatives
# result : list of m x
def HardNegativeMining(testX, testY, batchSize):
    result = {}
    result['PositivePrediction'] = ResultManager()
    result['HardNegative'] = ResultManager()

    imgCount = 0
    rocData = ROCData()

    for start, end in zip(range(0, testX.shape[0], batchSize), range(batchSize, (testX.shape[0] + batchSize), batchSize)):
        if (end > len(testX)):
            end = len(testX)
        batchData = testX[start:end]
        batchLabel = testY[start:end]
        imgCount += batchData.shape[0]
        prob, pred, correct = sess.run((prob_op, predict_op, correct_pred), feed_dict={X: batchData, Y: batchLabel, is_training: False})
        # correct = sess.run(correct_pred, feed_dict={X: testX[start:end], Y: testY[start:end], is_training: False})
        # print prob, pred, correct

        # 1. false positives
        # index1 = np.logical_and((correct == False), (pred == 1))

        # 2. rare negatives
        thr = 1.0
        index = np.logical_and((batchLabel[:,0] == 1), (prob[:,0] < thr))
        result['HardNegative'].Add(batchData[index,:,:,:], batchLabel[index,0], prob[index,0])
        # tp only
        # rocData.Add(batchLabel[index, 1], prob[index, 1])

        index = (prob[:,1] >= 0.5)
        # print index.shape
        result['PositivePrediction'].Add(batchData[index,:,:,:], batchLabel[index,1], prob[index, 1])
        # print batchLabel[index, 1]
        # print prob[index, 1]


        # all data
        rocData.Add(batchLabel[:,1], prob[:,1])

        if (0):
            sel = 0
            plt.ion()
            plt.imshow(negImg[sel,:,:,:])
            print prob[index]
            # plt.imshow(falsePos[0,:,:,:])
            # plt.show()
            plt.pause(0.1)

            # pick hard negative
            # correctPos = correct[testY[start:end][0] != 0]
    # result['HardNegative'].SaveImage('ResultImage/HardNegative')
    # result['PositivePrediction'].SaveImage('ResultImage/PositivePrediction')
    result['HardNegative'].SaveLMDB('data/HardNegative')
    rocData.PlotROCCurve()
    # print result['HardNegative'].count
    # hardNegatives, hardNegativeScores = ConvertListToNdarray(hardNegativeList, hardNegativeScoreList)
    # positives, positiveScores= ConvertListToNdarray(positiveList, positiveScoreList)

    print 'Pos %d, HardNeg %d, All %d' % (result['PositivePrediction'].count, result['HardNegative'].count, imgCount)

    # SaveImage(hardNegatives, hardNegativeScores, 'ResultImage/HardNegative')
    # SaveImage(positives, positiveScores, 'ResultImage/TruePositives')
    # hardNegatives : m x w x h x c
    return result['HardNegative'].npImages, result['HardNegative'].count, imgCount


X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, classes])

is_training = tf.placeholder(tf.bool, name='is_training')

# p_keep_conv = tf.placeholder("float")
# pred = ModelSimple(X,is_training)
# pred = ModelVGGLikeSmall(X,is_training)
output = ModelVGGLike(X,is_training)

# prob
prob_op = tf.nn.softmax(output)

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

# print 'Training data : %d ea' % len(trXList[0])

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(savePath)
    if resumeTraining == False:
        print "No pre trained snapshot"
        exit(0)
    elif checkpoint:
        print "Restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "Couldn't find checkpoint to restore from. Starting over."
        exit(0)

    # trainLoss, trainAcc = Prediction(trXFull, trYFull, batchSize)
    hardNegatives, negCount, imgCount = HardNegativeMining(trXFull, trYFull, batchSize)

    print hardNegatives.shape
    print 'Mined %d, all %d' % (negCount, imgCount)
    # valLoss, valAcc = Prediction(teX, teY, batchSize)

    # plot.Add(0, trainLoss, valLoss, trainAcc, valAcc)

    # plot.Add(i, trainLoss, valLoss, 0, 0)
    # plot.Show()
    # saver.save(sess, savePath + '/progress_%d' % i)
