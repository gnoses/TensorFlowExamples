"""
Convolutional Encoder Decoder Net
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np
import math
from TrainingPlot import *
from PIL import Image
from batch_norm import batch_norm
import cPickle as pkl
import time
width = 320
height = 224

featureSize = [64,32,1]
filterSize = [9,1,5]

weights = {
    'ce1': tf.get_variable("ce1",shape= [filterSize[0],filterSize[0], 1, featureSize[0] ], initializer=tf.contrib.layers.xavier_initializer()),
    'ce2': tf.get_variable("ce2",shape= [filterSize[1],filterSize[1], featureSize[0], featureSize[1]], initializer=tf.contrib.layers.xavier_initializer()),
    'ce3': tf.get_variable("ce3",shape= [filterSize[2],filterSize[2], featureSize[1], 1], initializer=tf.contrib.layers.xavier_initializer()),
    # 'ce4': tf.get_variable("ce4",shape= [filterSize[0],filterSize, featureSize[2], 1], initializer=tf.contrib.layers.xavier_initializer()),
    'b1': tf.get_variable("b1",shape= [1]),
    'b2': tf.get_variable("b2",shape= [1]),
    'b3': tf.get_variable("b3",shape= [1])
}


def Model(X, W):

    # Encoder
    encoder1 = tf.add(tf.nn.conv2d(X, W['ce1'], strides=[1, 1, 1, 1], padding='SAME'),W['b1'])
    encoder1 = tf.nn.relu(encoder1)

    encoder2 = tf.add(tf.nn.conv2d(encoder1, W['ce2'], strides=[1, 1, 1, 1], padding='SAME'), W['b2'])
    encoder2 = tf.nn.relu(encoder2)

    output = tf.add(tf.nn.conv2d(encoder2, W['ce3'], strides=[1, 1, 1, 1], padding='SAME'), W['b3'])

    return output


def LoadStanfordBG(filename):
    class DataSets(object):
        pass

    datalistFile = open(filename, "rt")
    fileList = datalistFile.readlines()
    # print len(fileList)
    data = None
    label = None
    # for i in range(0,len(fileList),2):
    for i in range(0,100,2):
        file = fileList[i].replace('\n','')
        # print ('%d / %d' % (i, len(fileList)))
        img = Image.open(file)
        # img = img.resize((224, 224))
        rgb = np.array(img).reshape(1,height,width,3)

        # pixels = np.concatenate((np.array(rgb[0]).flatten(),np.array(rgb[1]).flatten(),np.array(rgb[2]).flatten()),axis=0)
        # pixels = pixels.reshape(pixels.shape[0], 1)
        if i == 0:
            data = rgb
        else:
            data = np.concatenate((data, rgb),axis=0)

        # file = fileList[i * 2 + 1].replace('\n', '')
        # label = Image.open(file)
    label = (np.copy(np.float32(data[:,:,:,0]) / 255)).reshape(-1,height,width,1)
    dataLow = np.copy(label)

    # data = np.array(data[:,:,0])
    for i in range(data.shape[0]):
        imgHigh = Image.fromarray(data[i,:,:,0].reshape([height,width]))
        scale = 3.0
        imgLow = imgHigh.resize((np.uint8(width/3.0),np.uint8(height/3.0)), Image.BICUBIC)
        imgLow = imgLow.resize((width, height), Image.BICUBIC)


        dataLow[i,:,:,0] = np.array(np.float32(imgLow) / 255.0)

        # plt.subplot(1, 2, 1)
        # plt.imshow(dataLow[i,:,:,0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(label[i,:,:,0])
        # plt.show()

    return dataLow, label

# 1. data preparation for mnist

startTime = time.time()
print('Start data loading')
trainData, trainLabel = LoadStanfordBG('./StanfordBG/train.txt')
print('Finished in %d sec' % (time.time() - startTime))

# Define functions
x = tf.placeholder(tf.float32, [None, height,width,1])
y = tf.placeholder(tf.float32, [None, height,width,1])

pred = Model(x, weights)

# cost
cost = tf.reduce_mean(tf.pow(pred - y, 2))

learning_rate = 0.001

optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Fit all training data
batch_size = 5
n_epochs = 10000
m = trainData.shape[0]
print("Strart training..")

trainingPlot = TrainingPlot()
trainingPlot.SetConfig(batch_size, 500, n_epochs)

# you need to initialize all variables
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(".")
    if checkpoint:
        print "Restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "Couldn't find checkpoint to restore from. Starting over."

    for epoch_i in range(n_epochs):
        trainLoss = []
        trainAcc = []

        # print epoch_i
        for start, end in zip(range(0, m, batch_size), range(batch_size, m, batch_size)):
            # sess.run(train_op, feed_dict={X: trX[start:end], p_keep_conv: 0.8})
            # print start,end
            batchData = trainData[start:end]
            batchLabel = trainLabel[start:end]
            # print trainData.shape
            # print batchData.shape
            sess.run(optm, feed_dict={x: batchData, y: batchLabel})
            trainLoss.append(sess.run(cost, feed_dict={x: batchData, y: batchLabel}))
            # trainAcc.append(sess.run(acc_op, feed_dict={x: batchData, y: batchLabel}))
            # print 'pred', sess.run(pred, feed_dict={x: batchData, y: batchLabel})
            # print 'logPred', sess.run(logPred, feed_dict={x: batchData, y: batchLabel})

        trainLoss = np.mean(trainLoss)
        # trainAcc = np.mean(trainAcc)
        #

        # n_examples = 100
        # test_xs, _ = mnist.test.next_batch(n_examples)
        # test_xs_norm = np.array([img - mean_img for img in test_xs])
        # valLoss = sess.run(cost, feed_dict={x: test_xs_norm})
        # trainAcc = 0
        # valAcc = 0

        # print (epoch_i,trainLoss)
        trainingPlot.Add(epoch_i, trainLoss, 0, 0, 0)
        plt.figure(1)
        trainingPlot.Show()

        # save snapshot
        if epoch_i % 100 == 0:
            print "training on image #%d" % epoch_i
            saver.save(sess, 'progress', global_step=epoch_i)

        # show debugging image
        if (epoch_i % 10 == 0):
            batchData = trainData[np.random.randint(0,m)]
            batchLabel = trainLabel[np.random.randint(0,m)]

            # predMaxOut = sess.run(predMax, feed_dict={x: batchData, y: batchLabel})
            # yMaxOut = sess.run(yMax, feed_dict={x: batchData, y: batchLabel})
            # print trainData.shape
            predOut = sess.run(pred, feed_dict={x: batchData, y: batchLabel})

            # for i in range(22):
            # show predicted image

            plt.figure(2)
            plt.clf()
            plt.subplot(2, 2, 1)
            img = trainData[0, :, :, :].reshape(height, width, 1)
            # print img.shape
            img = np.uint8(img.reshape(height,width) * 255)
            plt.imshow(Image.fromarray(img), cmap='Greys_r')
            plt.subplot(2, 2, 2)
            img2 = predOut[0, :, :].reshape(height, width)
            img2 = np.uint8(img2.reshape(height, width) * 255)

            plt.imshow(Image.fromarray(img2), cmap='Greys_r')
            plt.subplot(2, 2, 3)
            plt.imshow(np.abs(img - img2))
            # plt.pause(0.01)
            # plt.subplot(2, 2, 4)
            # plt.imshow(img - predMaxOut[0, :, :].reshape(height, width))
            plt.pause(0.01)

        # print(epoch_i, "/", n_epochs, loss)

print("Training done. ")
