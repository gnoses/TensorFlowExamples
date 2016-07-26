"""
SVHN Semantic Segmentation training

SejinPark 2016
"""

import tensorflow as tf
from TrainingPlot import *
from PIL import Image
import time

# modeling configuration
width = 128 # 320
height = 128 # 224
classes = 10
kernelSize = 3
featureSize = 32
resumeTraining = False

# sovling configuration
sampleCount = 1000
# sampleCount = None
batch_size = 10
n_epochs = 10000
modelType = 1       # 0 : no pooling, 1 : max pool + mean unpool
mode = 0            # 0 : training, 1 : predict and visualize, 2 : predict and evaluate accuracy
resultText = []
if (mode == 1 or mode == 2):
    resumeTraining = True


weights = {
    'ce1': tf.get_variable("ce1",shape= [kernelSize,kernelSize, 3, featureSize]),
    'ce2': tf.get_variable("ce2",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'ce3': tf.get_variable("ce3",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'ce4': tf.get_variable("ce4",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd4': tf.get_variable("cd4",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd3': tf.get_variable("cd3",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd2': tf.get_variable("cd2",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'cd1': tf.get_variable("cd1",shape= [kernelSize,kernelSize, featureSize, featureSize]),
    'dense_inner_prod': tf.get_variable("dense_inner_prod",shape= [1, 1, featureSize,classes])
}

def CreateDB(categoryName):
    pathLoad1 = categoryName
    pathLoad2 = categoryName + 'annot'
    curPath = os.path.dirname(os.path.abspath(__file__))
    print curPath
    fileList1 = glob.glob(curPath + '/' + pathLoad1 + '/*.png')

    #fileList2 = glob.glob(pathLoad2 + '/*.png')
    #plt.ion()
    trainFile = open(categoryName + '.txt','wt')
    count = 0
    occupancyList = []
    for filename in fileList1:
        img1 = Image.open(filename)
        #filename2 = filename.replace('_IPMImg', '_IPMLabel')
        filename2 = curPath + '/' + pathLoad2 + '/' + os.path.basename(filename)
        img2 = Image.open(filename2)

        print >> trainFile, filename
        print >> trainFile, filename2

        #cropimg.save(pathSave + '/' + filename)
        #plt.title(pathCity)
        #plt.imshow(cropimg)
        #plt.show()
        count += 1

    print ('%d data list created' % count)
    trainFile.close()
    return occupancyList

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

# max pool + stride 2 transpose conv
def ModelNoPool(X, W):

    # Encoder
    encoder1 = tf.nn.conv2d(X, W['ce1'], strides=[1, 1, 1, 1], padding='SAME')
    encoder1 = tf.nn.batch_normalization(encoder1,0.001,1.0,0,1,0.0001)
    encoder1 = tf.nn.relu(encoder1)
    # encoder1 = tf.nn.dropout(encoder1, 0.5)

    encoder2 = tf.nn.conv2d(encoder1, W['ce2'], strides=[1, 1, 1, 1], padding='SAME')
    encoder2 = tf.nn.batch_normalization(encoder2, 0.001, 1.0, 0, 1, 0.0001)
    encoder2 = tf.nn.relu(encoder2)
    # encoder2 = tf.nn.dropout(encoder2, 0.5)

    encoder3 = tf.nn.conv2d(encoder2, W['ce3'], strides=[1, 1, 1, 1], padding='SAME')
    encoder3 = tf.nn.batch_normalization(encoder3, 0.001, 1.0, 0, 1, 0.0001)
    encoder3 = tf.nn.relu(encoder3)
    # encoder3 = tf.nn.dropout(encoder3, 0.5)

    encoder4 = tf.nn.conv2d(encoder3, W['ce4'], strides=[1, 1, 1, 1], padding='SAME')
    encoder4 = tf.nn.batch_normalization(encoder4, 0.001, 1.0, 0, 1, 0.0001)
    encoder4 = tf.nn.relu(encoder4)
    # encoder4 = tf.nn.dropout(encoder4, 0.5)

    # Decoder
    decoder4 = tf.nn.conv2d(encoder4, W['cd4'], strides=[1, 1, 1, 1], padding='SAME')
    decoder4 = tf.nn.batch_normalization(decoder4, 0.001, 1.0, 0, 1, 0.0001)
    decoder4 = tf.nn.relu(decoder4)
    # decoder4 = tf.nn.dropout(decoder4, 0.5)

    decoder3 = tf.nn.conv2d(decoder4, W['cd3'], strides=[1, 1, 1, 1], padding='SAME')
    decoder3 = tf.nn.batch_normalization(decoder3, 0.001, 1.0, 0, 1, 0.0001)
    decoder3 = tf.nn.relu(decoder3)
    # decoder3 = tf.nn.dropout(decoder3, 0.5)

    decoder2 = tf.nn.conv2d(decoder3, W['cd2'], strides=[1, 1, 1, 1], padding='SAME')
    decoder2 = tf.nn.batch_normalization(decoder2, 0.001, 1.0, 0, 1, 0.0001)
    decoder2 = tf.nn.relu(decoder2)
    decoder2 = tf.nn.dropout(decoder2, 0.5)

    decoder1 = tf.nn.conv2d(decoder2, W['cd1'], strides=[1, 1, 1, 1], padding='SAME')
    decoder1 = tf.nn.batch_normalization(decoder1, 0.001, 1.0, 0, 1.0, 0.0001)
    decoder1 = tf.nn.relu(decoder1)
    decoder1 = tf.nn.dropout(decoder1, 0.5)

    output = tf.nn.conv2d(decoder1, W['dense_inner_prod'], strides=[1, 1, 1, 1], padding='SAME')

    return output

# max pool + mean unpool
def ModelMeanUnpool(X, W):

    # Encoder
    encoder1 = tf.nn.conv2d(X, W['ce1'], strides=[1, 1, 1, 1], padding='SAME')
    encoder1 = tf.nn.batch_normalization(encoder1,0.001,1.0,0,1,0.0001)
    encoder1 = tf.nn.relu(encoder1)
    encoder1 = tf.nn.max_pool(encoder1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # encoder1 = tf.nn.dropout(encoder1, 0.5)

    encoder2 = tf.nn.conv2d(encoder1, W['ce2'], strides=[1, 1, 1, 1], padding='SAME')
    encoder2 = tf.nn.batch_normalization(encoder2, 0.001, 1.0, 0, 1, 0.0001)
    encoder2 = tf.nn.relu(encoder2)
    encoder2 = tf.nn.max_pool(encoder2, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    # encoder2 = tf.nn.dropout(encoder2, 0.5)

    encoder3 = tf.nn.conv2d(encoder2, W['ce3'], strides=[1, 1, 1, 1], padding='SAME')
    encoder3 = tf.nn.batch_normalization(encoder3, 0.001, 1.0, 0, 1, 0.0001)
    encoder3 = tf.nn.relu(encoder3)
    encoder3 = tf.nn.max_pool(encoder3, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    encoder3 = tf.nn.dropout(encoder3, 0.5)

    encoder4 = tf.nn.conv2d(encoder3, W['ce4'], strides=[1, 1, 1, 1], padding='SAME')
    encoder4 = tf.nn.batch_normalization(encoder4, 0.001, 1.0, 0, 1, 0.0001)
    encoder4 = tf.nn.relu(encoder4)
    encoder4 = tf.nn.max_pool(encoder4, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    encoder4 = tf.nn.dropout(encoder4, 0.5)
    # Decoder
    decoder4 = Unpooling(encoder4, [tf.shape(X)[0], height / 16, width / 16, featureSize])
    decoder4 = tf.nn.conv2d(decoder4, W['cd4'], strides=[1, 1, 1, 1], padding='SAME')
    decoder4 = tf.nn.batch_normalization(decoder4, 0.001, 1.0, 0, 1, 0.0001)
    decoder4 = tf.nn.relu(decoder4)
    decoder4 = tf.nn.dropout(decoder4, 0.5)

    decoder3 = Unpooling(encoder3, [tf.shape(X)[0], height/8, width/8, featureSize])
    decoder3 = tf.nn.conv2d(decoder3, W['cd3'], strides=[1, 1, 1, 1], padding='SAME')
    decoder3 = tf.nn.batch_normalization(decoder3, 0.001, 1.0, 0, 1, 0.0001)
    decoder3 = tf.nn.relu(decoder3)
    decoder3 = tf.nn.dropout(decoder3, 0.5)

    decoder2 = Unpooling(decoder3, [tf.shape(X)[0], height/4, width/4, featureSize])
    decoder2 = tf.nn.conv2d(decoder2, W['cd2'], strides=[1, 1, 1, 1], padding='SAME')
    decoder2 = tf.nn.batch_normalization(decoder2, 0.001, 1.0, 0, 1, 0.0001)
    decoder2 = tf.nn.relu(decoder2)
    # decoder2 = tf.nn.dropout(decoder2, 0.5)

    decoder1 = Unpooling(decoder2, [tf.shape(X)[0], height / 2, width / 2, featureSize])
    decoder1 = tf.nn.conv2d(decoder1, W['cd1'], strides=[1, 1, 1, 1], padding='SAME')
    decoder1 = tf.nn.batch_normalization(decoder1, 0.001, 1.0, 0, 1.0, 0.0001)
    decoder1 = tf.nn.relu(decoder1)
    # decoder1 = tf.nn.dropout(decoder1, 0.5)

    output = tf.nn.conv2d(decoder1, W['dense_inner_prod'], strides=[1, 1, 1, 1], padding='SAME')

    # return output, mask1, mask2, mask3
    return output

def DenseToOneHot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def LoadTrainingData(filename,sampleCount=None):
    class DataSets(object):
        pass

    datalistFile = open(filename, "rt")
    fileList = datalistFile.readlines()
    # print len(fileList)
    data = None
    label = None
    if sampleCount == None:
        sampleCount = len(fileList)
    print 'Load %d samples' % sampleCount
    for i in range(0,sampleCount,2):
    # for i in range(0,50,2):
    #     print i, fileList[i]
        file = fileList[i].replace('\n','')
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

        file = fileList[i+1].replace('\n', '')
        # print i,file
        img = Image.open(file)
        img = img.resize((width, height), Image.NEAREST)
        labelImage = np.array(img).reshape(1, height, width,1)

        if i == 0:
            label = labelImage
        else:
            # print data.shape
            label = np.concatenate((label, labelImage), axis=0)
    labelOneHot = np.zeros((label.shape[0],label.shape[1], label.shape[2], classes))
    for row in range(height):
        for col in range(width):
            single = label[:, row, col, 0]
            # print single.shape
            # exit(0)
            # print index
            oneHot = DenseToOneHot(single, classes)
            labelOneHot[:, row, col, :] = oneHot
    # for i in range(22):
    #     plt.subplot(1,2,1)
    #     plt.imshow(data[0,:,:,:].reshape(height,width,3))
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(labelOneHot[0,:,:,i].reshape(height,width))
    #     plt.show()
    return [data.astype(np.float32)/255, label, labelOneHot.astype(np.float32)]

def ShowDebuggingPlot(sess, inputData, tensor):
    x = tensor['x']
    y = tensor['y']
    trainData = inputData['trainData']
    trainLabelOneHot = inputData['trainLabelOneHot']
    index = np.random.randint(trainData.shape[0])
    batchData = trainData[index:index+1]
    batchLabel = trainLabelOneHot[index:index+1]
    predMaxOut = sess.run(tensor['predMax'], feed_dict={x: batchData, y: batchLabel})
    yMaxOut = sess.run(tensor['yMax'], feed_dict={x: batchData, y: batchLabel})
    # for i in range(22):
    # show predicted image
    plt.figure(2)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.title('Input')
    img = trainData[index, :, :, :].reshape(height, width, 3)
    plt.imshow(img * 255)
    plt.subplot(2, 2, 2)
    plt.title('Ground truth')
    img = yMaxOut[0, :, :].reshape(height, width)
    plt.imshow(img)
    plt.subplot(2, 2, 3)
    plt.title('Prediction')
    plt.imshow(predMaxOut[0, :, :].reshape(height, width))
    plt.subplot(2, 2, 4)
    plt.title('Error')
    plt.imshow(img - predMaxOut[0, :, :].reshape(height, width))
    plt.pause(0.01)


def Train(inputData, tensor, savePath):
    print('Start data loading mode : ', mode)
    # global batchData, batchLabel
    trainingPlot = TrainingPlot()
    trainingPlot.SetConfig(batch_size, 500, n_epochs)
    x = tensor['x']
    y = tensor['y']
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()

        checkpoint = tf.train.latest_checkpoint(savePath)
        # checkpoint = tf.train.get_checkpoint_state(savePath)
        if resumeTraining == False:
            print "Start from scratch"
        elif checkpoint:
            print "Restoring from checkpoint", checkpoint
            saver.restore(sess, checkpoint)
        else:
            print "Couldn't find checkpoint to restore from. Starting over."

        for epoch_i in range(n_epochs+1):
            trainLoss = []
            trainAcc = []
            for start, end in zip(range(0, len(inputData['trainData']), batch_size),
                                  range(batch_size, len(inputData['trainData']), batch_size)):
                batchData = inputData['trainData'][start:end]
                batchLabel = inputData['trainLabelOneHot'][start:end]

                sess.run(tensor['optm'], feed_dict={x: batchData, y: batchLabel})
                trainLoss.append(sess.run(tensor['cost'], feed_dict={x: batchData, y: batchLabel}))
                trainAcc.append(sess.run(tensor['acc_op'], feed_dict={x: batchData, y: batchLabel}))

            trainLoss = np.mean(trainLoss)
            trainAcc = np.mean(trainAcc)

            # run validation
            valLoss = sess.run(tensor['cost'], feed_dict={x: inputData['valData'], y: inputData['valLabelOneHot']})
            valAcc = sess.run(tensor['acc_op'], feed_dict={x: inputData['valData'], y: inputData['valLabelOneHot']})

            trainingPlot.Add(epoch_i, trainLoss, valLoss, trainAcc, valAcc)
            plt.figure(1)
            trainingPlot.Show()
            # trainingPlot.Save('trainignPlot.png')

            # save snapshot
            if resumeTraining and epoch_i % 10 == 0 and epoch_i > 0:
                print "training on image #%d" % epoch_i
                saver.save(sess, savePath + '/progress', global_step=epoch_i)
            # saver.save(sess, savePath + '/progress', global_step=epoch_i)
            # return
            # show debugging image
            # startTime = time.time()
            ShowDebuggingPlot(sess, inputData,tensor)
            # print 'ShowDebug Image in %f sec' % (time.time() - startTime)

            # print(epoch_i, "/", n_epochs, loss)

def PredictShow(inputData, tensor, savePath):
    x = tensor['x']
    y = tensor['y']

    # you need to initialize all variables
    sess = tf.Session()
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    # if (resumeTraining):
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(savePath)
    # checkpoint = tf.train.get_checkpoint_state(savePath)
    if checkpoint:
        print "Restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "Couldn't find checkpoint to restore from. Prediction over."
        return

    for i in range(0, len(inputData['trainData'])):
        batchData = inputData['trainData'][i:i + 1]
        batchLabel = inputData['trainLabelOneHot'][i:i + 1]

        # sess.run(optm, feed_dict={x: batchData, y: batchLabel})
        # trainAcc = sess.run(acc_op, feed_dict={x: batchData, y: batchLabel})

        predMaxOut = sess.run(tensor['predMax'], feed_dict={x: batchData, y: batchLabel})
        # show predicted image
        plt.clf()
        plt.ioff()
        plt.subplot(1, 2, 1)
        plt.title('Input')
        img = batchData[0, :, :, :].reshape(height, width, 3)
        plt.imshow(img * 255)
        plt.subplot(1, 2, 2)
        plt.title('Prediction')
        # print predMaxOut.shape
        plt.imshow(img * 255)
        plt.imshow(predMaxOut[0, :, :].reshape(height, width), alpha=0.4)
        plt.show()
        # plt.pause(0.01)


        # print(epoch_i, "/", n_epochs, loss)

def PredictEvaluate(inputData, tensor, savePath):
    x = tensor['x']
    y = tensor['y']

    # you need to initialize all variables

    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
        # if (resumeTraining):
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(savePath)
        # checkpoint = tf.train.get_checkpoint_state(savePath)
        # print checkpoint
        if checkpoint:
            print "Restoring from checkpoint", checkpoint
            saver.restore(sess, checkpoint)
        else:
            print "Couldn't find checkpoint to restore from. Prediction over."
            return

        trainAccList = []
        for start, end in zip(range(0, len(inputData['trainData']), batch_size),
                              range(batch_size, len(inputData['trainData']), batch_size)):
            batchData = inputData['trainData'][start:end]
            batchLabel = inputData['trainLabelOneHot'][start:end]

            sess.run(tensor['optm'], feed_dict={x: batchData, y: batchLabel})
            trainAccList.append(sess.run(tensor['acc_op'], feed_dict={x: batchData, y: batchLabel}))

        trainAcc = np.mean(trainAccList)

        valAccList = []
        for start, end in zip(range(0, len(inputData['valData']), batch_size),
                              range(batch_size, len(inputData['valData']), batch_size)):
            batchData = inputData['valData'][start:end]
            batchLabel = inputData['valLabelOneHot'][start:end]

            sess.run(tensor['optm'], feed_dict={x: batchData, y: batchLabel})
            valAccList.append(sess.run(tensor['acc_op'], feed_dict={x: batchData, y: batchLabel}))

        valAcc = np.mean(valAccList)
        result = 'Train : %f (%d), Val : %f (%d)' % (trainAcc, len(trainAccList), valAcc, len(valAccList))
        # print (result)
        resultText.append(result)


def Process(mode, modelType, dataPath, savePath):

    startTime = time.time()
    print('Start data loading')
    inputData = {}
    inputData['trainData'], inputData['trainLabel'], inputData['trainLabelOneHot'] = LoadTrainingData(dataPath + '/train.txt', sampleCount)
    inputData['valData'], inputData['valLabel'], inputData['valLabelOneHot'] = LoadTrainingData(dataPath + '/val.txt', sampleCount / 2)

    # print('Finished in %d sec' % (time.time() - startTime))

    tensor = {}

    # Define functions
    x = tf.placeholder(tf.float32, [None, height, width, 3])
    y = tf.placeholder(tf.float32, [None, height, width, classes])

    tensor['x'] = x
    tensor['y'] = y

    # output : m x height x width x classes
    if (modelType == 0):
        pred = ModelNoPool(x, weights)
    elif (modelType == 1):
        pred = ModelMeanUnpool(x, weights)
    else:
        print ('Wrong model type')
        return


    linearizePred = tf.reshape(pred, shape=[-1, classes])
    linearizeY = tf.reshape(y, shape=[-1, classes])
    tensor['cost'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(linearizePred, linearizeY))
    # accuracy
    # print yNumber, tf.argmax(pred, 3)
    tensor['predMax'] = tf.argmax(pred, 3)
    tensor['yMax'] = tf.argmax(y, 3)
    correct_pred = tf.equal(tf.argmax(y, 3), tf.argmax(pred, 3))  # Count correct predictions
    tensor['acc_op'] = tf.reduce_mean(tf.cast(correct_pred, "float"))  # Cast boolean to float to average
    learning_rate = 0.0001
    tensor['optm'] = tf.train.AdamOptimizer(learning_rate).minimize(tensor['cost'])
    # Fit all training data

    if (mode == 0):
        Train(inputData, tensor, savePath)
    elif (mode == 1):
        PredictShow(inputData, tensor, savePath)
    elif (mode == 2):
        PredictEvaluate(inputData, tensor, savePath)

    return resultText

# main body
if __name__ == '__main__':
    # Create DB (run once)
    if (0):
        CreateDB('CamVid/train')
        CreateDB('CamVid/val')
        exit(0)

    # mode = 0 : training, 1 : predict and visualize, 2 : predict and evaluate accuracy
    # modelType = 0 : no pooling, 1 : max pool + mean unpool
    resultText = []
    savePath = 'snapshot'
    dataPath = './data/PixelLabel/'
    Process(0, modelType, dataPath, savePath)
