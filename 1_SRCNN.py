"""
SRCNN (http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) implementation using Tensorflow

by Sejin Park

dataset :Stanford BG (http://dags.stanford.edu/data/iccv09Data.tar.gz)

"""
from PIL import Image
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np
import math
from TrainingPlot import *

width = 128
height = 128

weights = []

# input : [m x h x w x c]
def Unpooling(inputOrg, size, mask=None):
    # m, c, h, w order
    # print 'start unpooling'
    # size = tf.shape(inputOrg)
    m = size[0]
    h = size[1]
    w = size[2]
    c = size[3]
    input = tf.transpose(inputOrg, [0, 3, 1, 2])
    # print input.get_shape()
    x = tf.reshape(input, [-1, 1])
    k = np.float32(np.array([1.0, 1.0]).reshape([1,-1]))
    # k = tf.Variable([1.0, 1.0],name="weights")
    # k = tf.reshape(k,[1,-1])
    # k = np.array(k).reshape([1, -1])
    output = tf.matmul(x, k)
    output = tf.reshape(output,[-1, c, h, w * 2])
    # m, c, w, h
    xx = tf.transpose(output, [0, 1, 3, 2])
    xx = tf.reshape(xx,[-1, 1])
    # print xx.shape

    output = tf.matmul(xx, k)
    # m, c, w, h
    output = tf.reshape(output, [-1, c, w * 2, h * 2])
    output = tf.transpose(output, [0, 3, 2, 1])
    # print mask
    outshape = tf.pack([m, h * 2, w * 2, c])

    if mask != None:
        dense_mask = tf.sparse_to_dense(mask, outshape, output, 0)
        # print dense_mask
        # print 'output',output
        # print 'mask',mask
        # print dense_mask
            # output = tf.mul(output, mask)

        return output, dense_mask
    else:
        return output


def CreateWeight(kernelSize, inputSize, outputSize):
    name = 'w%d' % len(weights)
    return (tf.get_variable(name, shape=[kernelSize, kernelSize, inputSize, outputSize]))

# input : [
def ConvBNRelu(input, kernelSize, outputSize):
    inputSize = input.get_shape()[3].value
    # print type(inputSize)
    weights.append(CreateWeight(kernelSize, inputSize, outputSize))
    conv = tf.nn.conv2d(input, weights[-1], strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.batch_normalization(conv, 0.001, 1.0, 0, 1, 0.0001)
    return tf.nn.relu(conv)

def ModernModel(X, W):
    conv1 = ConvBNRelu(X, 3, 64)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    # encoder1 = tf.nn.dropout(encoder1, 0.5)

    conv2 = ConvBNRelu(conv1, 3, 64)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
    # encoder2 = tf.nn.dropout(encoder2, 0.5)

    conv3 = Unpooling(conv2, [tf.shape(X)[0], height / 4, width / 4, 64])
    conv3 = ConvBNRelu(conv3, 3, 64)
    # encoder3 = tf.nn.dropout(encoder3, 0.5)

    conv4 = Unpooling(conv3, [tf.shape(X)[0], height / 2, width / 2, 64])
    conv4 = ConvBNRelu(conv4, 3, 64)
    # encoder4 = tf.nn.dropout(encoder4, 0.5)

    weights.append(CreateWeight(3, 64, 1))
    output = tf.nn.conv2d(conv4, weights[-1], strides=[1, 1, 1, 1], padding='SAME')

    return output

def LoadStanfordBG(path, scale):
    class DataSets(object):
        pass
    fileList = glob.glob(path + '/*.jpg')
    # print len(fileList)


    for i, file in enumerate(fileList):
        # if (i>10):
        #     break
        # print ('%d / %d' % (i, len(fileList)))
        img = Image.open(file)
        img = img.resize((width, height))

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
        imgLow = imgHigh.resize((np.uint16(width/scale),np.uint16(height/scale)), Image.BICUBIC)
        imgLow = imgLow.resize((width, height), Image.BICUBIC)


        dataLow[i,:,:,0] = np.array(np.float32(imgLow) / 255.0)

        # plt.subplot(1, 2, 1)
        # plt.imshow(dataLow[i,:,:,0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(label[i,:,:,0])
        # plt.show()
    residual = label - dataLow
    return dataLow, residual, label

startTime = time.time()
print('Start data loading')
trainData, trainLabel, trainLabelOriginal = LoadStanfordBG('StanfordBG/images/', scale = 2.0)
print('Finished in %d sec' % (time.time() - startTime))

# Define functions
x = tf.placeholder(tf.float32, [None, height,width,1])
y = tf.placeholder(tf.float32, [None, height,width,1])

# pred = Model(x, weights)
pred = ModernModel(x, weights)

# cost
cost = tf.reduce_mean(tf.pow(pred - y, 2))

learning_rate = 0.0001

optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Fit all training data
batch_size = 5
n_epochs = 10000
m = trainData.shape[0]
print("Strart training..")

trainingPlot = TrainingPlot()
trainingPlot.SetConfig(batch_size, 500, n_epochs)


def PSNR(labelOriginal, reconst):
    mse = np.sqrt(np.mean((labelOriginal - reconst).flatten() ** 2))
    psnr = 20 * np.log10(255 / mse)
    plt.title(psnr)
    return psnr


def ShowDebugImage():
    selected = np.random.randint(0, m)
    batchData = trainData[selected].reshape([1, height, width, 1])
    batchLabel = trainLabel[selected].reshape([1, height, width, 1])
    batchLabelOriginal = trainLabelOriginal[selected].reshape([1, height, width, 1])
    # predMaxOut = sess.run(predMax, feed_dict={x: batchData, y: batchLabel})
    # yMaxOut = sess.run(yMax, feed_dict={x: batchData, y: batchLabel})
    # print trainData.shape
    predOut = sess.run(pred, feed_dict={x: batchData, y: batchLabel})
    # for i in range(22):
    # show predicted image
    plt.figure(2)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.title('Input')
    src = batchData
    src = np.uint8(src.reshape(height, width) * 255)
    plt.imshow(Image.fromarray(src), cmap='Greys_r')
    plt.subplot(2, 2, 2)
    plt.title('Label')
    labelOriginal = batchLabelOriginal
    labelOriginal = np.uint8(labelOriginal.reshape(height, width) * 255)
    plt.imshow(Image.fromarray(labelOriginal), cmap='Greys_r')
    plt.subplot(2, 2, 3)
    plt.title('Output')
    output = predOut[0, :, :].reshape(height, width)
    output = np.uint8(output.reshape(height, width) * 255)
    plt.imshow(Image.fromarray(output), cmap='Greys_r')
    plt.subplot(2, 2, 4)
    # plt.title('Error')
    # plt.imshow(np.abs(img - img2))
    plt.title('Reconst')
    reconst = np.uint8(output + src)
    plt.imshow(Image.fromarray(reconst), cmap='Greys_r')
    print 'cnn : ', PSNR(labelOriginal, reconst), 'bicubic : ', PSNR(labelOriginal, src)


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

            if (0):
                print batchData.shape
                predOut = sess.run(pred, feed_dict={x: batchData, y: batchLabel})

                for i in range(batch_size):
                    src = batchData[i]
                    src = np.uint8(src.reshape(height, width) * 255)

                    output = predOut[i, :, :].reshape(height, width)
                    output = np.uint8(output.reshape(height, width) * 255)
                    filename = 'result/output%d_%d.png' % (epoch_i, i)
                    Image.fromarray(output + src).save(filename)
                    print filename + ' saved'

                continue



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
        if epoch_i % 500 == 0:
            print "training on image #%d" % epoch_i
            saver.save(sess, 'progress', global_step=epoch_i)



        # show debugging image
        if (epoch_i % 10 == 0):
            ShowDebugImage()


            # plt.imshow(Image.fromarray(src).resize((width,height),Image.BICUBIC), cmap='Greys_r')


            # plt.pause(0.01)
            # plt.subplot(2, 2, 4)
            # plt.imshow(img - predMaxOut[0, :, :].reshape(height, width))
            plt.pause(0.01)

        # print(epoch_i, "/", n_epochs, loss)

print("Training done. ")

