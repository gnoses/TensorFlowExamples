from LMDBTool import LMDBTool
import numpy as np

# labels_dense : m x 1
# output : m x num_classes
def DenseToOneHot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def LoadData(pathLoad, classes):
    lmdbToolReadTrain = LMDBTool(pathLoad + '/train', 1000, False)
    trXFull, label = lmdbToolReadTrain.Get(32,32,3)
    trYFull = DenseToOneHot(label, classes)
    lmdbToolReadVal = LMDBTool('data/CroppedSmall1000LMDB/val', 1000, False)
    teX, label = lmdbToolReadVal.Get(32,32,3)
    teY = DenseToOneHot(label, classes)
    print 'Load data train : ', trXFull.shape, trYFull.shape
    print 'Load data val : ', teX.shape, teY.shape
    return trXFull, trYFull, teX, teY



# load training data list file
def LoadTrainingData(filename):
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
    for i in range(sampleCount):
        # for i in range(0,50,2):
        str = fileList[i].replace('\n', '')
        str = str.split(' ')
        file = str[0]
        classId = int(str[1])

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

def LoadDB(path,pickleLoad = False):
    # startTime = time.time()
    if (pickleLoad == False):
        trX, trY = LoadTrainingData(path + '/train.txt')
        teX, teY = LoadTrainingData(path + '/val.txt')

        with open(path + '/db.pkl','wb') as fp:
            pkl.dump(trX, fp)
            pkl.dump(trY, fp)
            pkl.dump(teX, fp)
            pkl.dump(teY, fp)
    else:
        with open(path + '/db.pkl','rb') as fp:
            trX = pkl.load(fp)
            trY = pkl.load(fp)
            teX = pkl.load(fp)
            teY = pkl.load(fp)
    # print trX.shape, teX.shape

    # print (time.time() - startTime), ' sec'
    return trX, trY, teX, teY

# random pick some negative samples
# return split list of pos + partial neg samples
def PickNegativeSample(trXFull, trYFull):
    pos = trXFull[trYFull[:,0]==0,:,:,:]
    poslabel = trYFull[trYFull[:, 0] == 0, :]
    neg = trXFull[trYFull[:,0]==1,:,:,:]
    neglabel = trYFull[trYFull[:,0]==1,:]
    posCount = pos.shape[0]
    negCount = neg.shape[0]
    print posCount, negCount
    assert(posCount + negCount == trXFull.shape[0])
    # set negative batch to pos * 2
    batchSize = posCount * 2


    pickIndex = np.random.permutation(neg.shape[0])

    trXList = []
    trYList = []
    for start, end in zip(range(0, negCount, batchSize), range(batchSize, negCount, batchSize)):
        batchIndex = pickIndex[start:end]
        negPick = neg[batchIndex, :,:,:]
        neglabelPick = neglabel[batchIndex,:]
        trXList.append(np.concatenate((pos, negPick), axis=0))
        trYList.append(np.concatenate((poslabel, neglabelPick), axis=0))

    print 'Pick negatives : pos %d, neg (%d / %d) * %d' % (posCount, batchSize, negCount, len(trXList))
    return trXList, trYList

