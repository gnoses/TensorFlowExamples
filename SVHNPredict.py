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
classes = 11

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
    conv2 = ConvBNRelu(conv1, 3, 128, is_training)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv3 = ConvBNRelu(conv2, 3, 128, is_training)
    conv3 = ConvBNRelu(conv2, 3, 128, is_training)
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


# load training data list file
# load training data list file
def LoadTrainingData(filename, negativeSampleCount=None):
    class DataSets(object):
        pass

    datalistFile = open(filename, "rt")
    fileList = datalistFile.readlines()
    np.random.shuffle(fileList)
    # print len(fileList)
    data = []
    label = []
    # if sampleCount == None:

    sampleCount = len(fileList)
    hist = np.zeros(classes)
    for i in range(0, sampleCount, 2):
        # for i in range(0,50,2):
        str = fileList[i].replace('\n', '')
        str = str.split(' ')
        file = str[0]
        classId = int(str[1])
        if (negativeSampleCount != None and classId == 0):
            if negativeSampleCount < 0:
                continue
            negativeSampleCount -= 1

        hist[classId] += 1
        if (i % 10000 == 0):
            print '%d / %d : %s = %d' % (i, sampleCount, file, classId)

        img = Image.open(file)

        data.append(np.array(img))
        label.append(classId)

    # label : m x 1 nparray
    # oneHot : m x classes nparray of one hot code
    data = np.array(data).reshape(-1, 32, 32, 3)
    label = np.array(label)
    # print '1'
    oneHot = DenseToOneHot(label, classes)
    # print '2'
    print hist
    print hist / np.sum(hist) * 100
    # for i in range(22):
    #     plt.subplot(1,2,1)
    #     plt.imshow(data[0,:,:,:].reshape(height,width,3))
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(labelOneHot[0,:,:,i].reshape(height,width))
    #     plt.show()
    return [data.astype(np.float32) / 255, oneHot.astype(np.float32)]


# gpu memory restricion
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def LoadDB(pickleLoad = False):
    # startTime = time.time()
    if (pickleLoad == False):
        trX, trY = LoadTrainingData('data/CroppedSmall/train.txt')
        teX, teY = LoadTrainingData('data/CroppedSmall/val.txt')

        with open('data/Cropped/db.pkl','wb') as fp:
            pkl.dump(trX, fp)
            pkl.dump(trY, fp)
            pkl.dump(teX, fp)
            pkl.dump(teY, fp)
    else:
        with open('data/Cropped/db.pkl','rb') as fp:
            trX = pkl.load(fp)
            trY = pkl.load(fp)
            teX = pkl.load(fp)
            teY = pkl.load(fp)
    # print trX.shape, teX.shape

    # print (time.time() - startTime), ' sec'
    return trX, trY, teX, teY

# random pick some negative samples
# return split list of pos + partial neg samples
def PickNegativeSample(trXFull, trYFull, split):
    pos = trXFull[trYFull[:,0]==0,:,:,:]
    poslabel = trYFull[trYFull[:, 0] == 0, :]
    neg = trXFull[trYFull[:,0]==1,:,:,:]
    neglabel = trYFull[trYFull[:,0]==1,:]
    negCount = neg.shape[0]
    batchSize = negCount / split
    pickIndex = np.random.permutation(neg.shape[0])

    trXList = []
    trYList = []
    for start, end in zip(range(0, negCount, batchSize), range(batchSize, negCount, batchSize)):
        batchIndex = pickIndex[start:end]
        negPick = neg[batchIndex, :,:,:]
        neglabelPick = neglabel[batchIndex,:]
        trXList.append(np.concatenate((pos, negPick), axis=0))
        trYList.append(np.concatenate((poslabel, neglabelPick), axis=0))

    return trXList, trYList

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
        print acc
        valLoss.append(loss)
        valAcc.append(acc)

    return np.mean(valLoss), np.mean(valAcc) # , np.mean(valAccPos)


# X : m x w x h x 3
# Y : m x classes
trXFull, trYFull, teX, teY = LoadDB(True)
# trXList, trYList = PickNegativeSample(trXFull, trYFull, 10)

print 'Full Train data :', trXFull.shape
print 'Val data :', teX.shape

# exit(0)
# trX = trX[:5000,:,:,:]
# trY = trY[:5000,:]
# print trX.shape

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, classes])

is_training = tf.placeholder(tf.bool, name='is_training')

# p_keep_conv = tf.placeholder("float")
# pred = ModelSimple(X,is_training)
# pred = ModelVGGLikeSmall(X,is_training)
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
batchSize = 10
sampleCount = len(trXFull)
totalIter = 1000000
plot.SetConfig(batchSize, sampleCount, totalIter)
resumeTraining = True
savePath = 'snapshotSmall2'
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

    trainLoss, trainAcc = Prediction(trXFull, trYFull, batchSize)
    valLoss, valAcc = Prediction(teX, teY, batchSize)

    plot.Add(0, trainLoss, valLoss, trainAcc, valAcc)

    # plot.Add(i, trainLoss, valLoss, 0, 0)
    # plot.Show()
    # saver.save(sess, savePath + '/progress_%d' % i)
