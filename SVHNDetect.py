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
import PIL.ImageDraw as ImageDraw
import cPickle as pkl
import scipy.io as sio

# import matplotlib.pyplot as plt

weights = []
classes = 2
count = [0,0,0,0,0,0,0,0,0,0,0,0]
stride = 5

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


def ExtractPatch(imgData, roi):
    # count[classId] = count[classId] + 1
    croppedImg = imgData.crop(roi)
    croppedImg = croppedImg.resize((32, 32), Image.BICUBIC)
    # print imgPath
    # croppedImg.save(imgPath)
    # print savePath
    return croppedImg
# calulate IoU
def CalcIoU(a, b):

    width = min((a[2],b[2])) - max((a[0],b[0]))
    if width <= 0:
        return 0
    height  = min((a[3], b[3])) - max((a[1], b[1]))
    if height <= 0:
        return 0
    areaA = a[4] * a[5]
    areaB = b[4] * b[5]
    intersect = width * height
    return float(intersect) / float(areaA + areaB - intersect)

# crop negative samples with same size of gt in specific stride
def ExaustiveSearch(sess, imgData, gtList, stride):
    width = gtList[0][4]
    height = gtList[0][5]
    # print gtList
    # data = None
    # label = []
    imgDetect = imgData.copy()
    drawDetect = ImageDraw.Draw(imgDetect)
    for row in range(0,imgData.size[1]-height-1,stride):
        for col in range(0, imgData.size[0]-width-1, stride):
            imgDisp = imgData.copy()
            hit = False

            draw = ImageDraw.Draw(imgDisp)

            b = (col, row, col + width, row + height, width, height)
            iou = 0.0
            classId = 0
            for gt in gtList:

                # print col, row, width, height, CalcIoU(a,b)
                draw.rectangle([(col, row), (col + width, row + height)], outline=(0,0,255))
                iou = CalcIoU(b, gt)

                if (iou > 0.5):
                    hit = True
                    classId = gt[6]
                    # print iou
                    break


            # if hit == False:
            # if hit:
            #     draw.rectangle([(col,row),(col+width,row+height)],outline=(0,0,255))
            # else:
            #     draw.rectangle([(col, row), (col + width, row + height)], outline=(255, 255, 255))
            # crop and resize target patch
            patch = ExtractPatch(imgData, (col, row, col + width, row + height))
            rgb = np.array(patch).reshape(1, patch.size[1], patch.size[0], 3).astype(np.float32) / 255
            # if data == None:
            #     data = rgb
            # else:
            #     data = np.concatenate((data, rgb), axis=0)
            #
            # label.append(classId)

            pred = sess.run(predict_op, feed_dict={X: rgb, is_training: False})
            # acc = sess.run(acc_op, feed_dict={X: teX, Y: teY, is_training: False})
            if pred > 0:
                draw.rectangle([(col,row),(col+width,row+height)],outline=(0,255,0))
            else:
                draw.rectangle([(col, row), (col + width, row + height)], outline=(255, 255, 255))
            if pred > 0:
                drawDetect.rectangle([(col,row),(col+width,row+height)],outline=(0,255,0))
                # drawDetect.text((col,row), '%d' % pred)

            if (0):
                plt.clf()
                plt.ioff()
                plt.subplot(1,2,1)
                plt.imshow(imgDisp)
                # plt.title(classId)
                plt.title(pred)
                plt.subplot(1, 2, 2)
                plt.imshow(patch)
                plt.title(iou)
                plt.pause(0.01)

    if (1):
        plt.clf()
        plt.ioff()
        plt.subplot(1,2,1)
        plt.imshow(imgData)
        plt.subplot(1, 2, 2)
        plt.imshow(imgDetect)
        # plt.title(iou)
        plt.pause(1)

    # label = np.array(label)
    # oneHot = DenseToOneHot(label, classes)
    # return [data.astype(np.float32) / 255, oneHot.astype(np.float32)]

def LoadImage(digitStruct, i, sess):
    info = (digitStruct['digitStruct'][0])[i]
    filename = info[0][0]
    rois = info[1]
    # print filename
    try:
        imgData = Image.open(pathLoad + filename)
    except:
        # print pathLoad + filename + ' failed'
        return [None, None]
    # if filename == '10727.png':
    #     print '-------------------------------------------------------'
    # print pathLoad + filename + ' ok'
    gtList = []
    imgDisp = imgData

    for j in range(rois.shape[1]):
        # draw = ImageDraw.Draw(imgDisp)
        # height, left, top, width, label
        left = int(rois[0,j][1])
        top = int(rois[0,j][2])
        right = int(left) + int(rois[0,j][3])
        bottom = top + int(rois[0,j][0])
        width = (right - left - 1)
        height = (bottom - top - 1)

        classId = int(rois[0,j][4])
        gt = (left, top, right, bottom, width, height, classId)
        # ExtractPatch(imgData, roi)
        # patch = ExtractPatch(imgData, (left, top, right, bottom))
        gtList.append(gt)
        # print savePath, count
        if (0):
            draw.rectangle([(left, top), (right, bottom)])
            plt.subplot(1,2,1)
            plt.imshow(imgDisp)
            plt.subplot(1, 2, 2)
            plt.imshow(patch)
            plt.pause(0.01)

    # X : m x w x h x 3
    # Y : m x classes
    return ExaustiveSearch(sess, imgData, gtList, stride)

# load full resolution image for detection
pathLoad = 'data/Original/train/'
savePath = 'snapshot10000'
digitStruct = sio.loadmat(pathLoad + 'digitStruct.mat')
sampleCount = digitStruct['digitStruct'].shape[1]
print 'Load %d data' % sampleCount

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
# sampleCount = len(trXFull)
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
    # valLoss, valAcc = Prediction(teX, teY, batchSize)

    for i in range(sampleCount):
        # X : m x w x h x 3
        # Y : m x classes
        # print digitStruct, i
        # teX, teY = LoadImage(digitStruct, i, sess)
        LoadImage(digitStruct, i, sess)
        # if teX == None:
        #     continue

        # print i, acc


    # plot.Add(0, trainLoss, valLoss, trainAcc, valAcc)

    # plot.Add(i, trainLoss, valLoss, 0, 0)
    # plot.Show()
    # saver.save(sess, savePath + '/progress_%d' % i)
